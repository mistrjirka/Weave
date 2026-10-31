"""Code-owned hooks retain real PyTorch behavior, data history and snapshots."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import uuid
import zipfile

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.training.api import router
from backend.training.recipe import starter_script
from backend.training.service import (
    TrainingService,
    TrainingConfig,
    Stage,
    Launch,
    events,
    build_snapshot,
)
from backend.training.weave_training import atomic_json, instrumentation_warnings
from backend.training.workspaces import WorkspaceStore
from test_training import tiny_project


def filled_starter(data, **settings):
    """Tests supply their own data hook; production starters contain no dataset logic."""
    source = starter_script()
    defaults = {
        "micro_batch_size": 2,
        "max_epochs": None,
        "max_optimizer_steps": 4,
        "checkpoint_every": 2,
    }
    defaults.update(settings)
    import ast

    tree = ast.parse(source)
    call = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
    )
    lines = source.splitlines(keepends=True)
    for kw in reversed(call.keywords):
        if kw.arg in defaults:
            line = lines[kw.value.lineno - 1]
            lines[kw.value.lineno - 1] = (
                line[: kw.value.col_offset]
                + repr(defaults[kw.arg])
                + line[kw.value.end_col_offset :]
            )
    source = "".join(lines)
    body = f"""from torch.utils.data import DataLoader
    from weave_support.data_stream import TokenStream
    def encode(row):
        return row['input_ids'], row.get('labels', row['input_ids'])
    dataset = TokenStream({str(data)!r}, encode, seq_len=4)
    return DataLoader(dataset, batch_size=config.micro_batch_size, num_workers=0), None"""
    return source.replace(
        'raise NotImplementedError("Add your data loaders here.")', body
    ).replace("tokenizer=load_tokenizer,", "tokenizer=None,")


def make_hook_run(tmp_path, script=None):
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "\n".join(json.dumps({"input_ids": [i % 7 + 1] * 17}) for i in range(12))
    )
    if script is None:
        script = filled_starter(data)
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    project = tiny_project()
    project["id"] = "project-" + uuid.uuid4().hex
    stage = Stage(
        id="hooks",
        name="Hooks",
        config=TrainingConfig(execution="monitor"),
        script=script,
    )
    stage = service.workspaces.save_stage(project["id"], stage)
    info = service.launch(Launch(project=project, stage=stage))
    return service, service.run_dir(info["id"]), stage


def execute(directory, *args):
    env = {**os.environ, "WEAVE_RUN_DIR": str(directory), "USE_TORCH": "1"}
    env.pop("WEAVE_REPORT_CONFIG", None)
    result = subprocess.run(
        [sys.executable, "train.py", *args],
        cwd=directory,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_hooks_no_user_json_real_training_snapshot_and_exact_resume(tmp_path):
    service, directory, stage = make_hook_run(tmp_path)
    assert not (Path(stage.script_path).parent / "config.json").exists()
    assert stage.config_path.endswith("weave_support/launch.json")
    assert not instrumentation_warnings(stage.script)
    # Snapshot requests must still be fulfilled at a safe optimizer boundary.
    atomic_json(
        directory / "commands/request.json", {"id": "request", "kind": "snapshot"}
    )
    execute(directory)
    records = events(directory)
    metrics = [e["payload"] for e in records if e["kind"] == "metrics"]
    assert all("train/loss" in e and "lr/group_0" in e for e in metrics)
    assert any(e["kind"] == "configuration" for e in records)
    manifest = json.loads((directory / "datasets.json").read_text())
    assert manifest["train-0"]["fingerprint"] == "stat"
    assert "sha256" not in manifest["train-0"]
    trace = [
        json.loads(line)
        for line in (directory / "consumed-rank-0.jsonl").read_text().splitlines()
    ]
    assert len(trace) == 4 and all(len(t["batches"]) == 2 for t in trace)
    assert json.loads(trace[0]["batches"][0]["source"][0])[0]["row"] == 0
    checkpoints = [e for e in records if e["kind"] == "checkpoint"]
    first = next(e for e in checkpoints if e["step"] == 1)
    assert first["payload"]["metadata"]["snapshot_requests"] == ["request"]
    assert first["payload"]["metadata"]["exact_resume"]
    snapshot = build_snapshot(directory, first, True, "test")
    with zipfile.ZipFile(directory / snapshot) as archive:
        assert "weave_support/trainer.py" in archive.namelist()
        assert any(
            name.startswith("source-data/train-0/") for name in archive.namelist()
        )
        history = [
            json.loads(line)
            for line in archive.read("consumed-rank-0.jsonl").decode().splitlines()
        ]
        assert all(row["step"] <= 1 for row in history)
    final_path = directory / checkpoints[-1]["payload"]["path"]
    expected = torch.load(final_path, map_location="cpu", weights_only=True)
    original_trace = trace
    # Same code and data, new output directory, restoring a mid-run data cursor.
    resumed = tmp_path / "resumed"
    env = {**os.environ, "WEAVE_RUN_DIR": str(resumed), "USE_TORCH": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(directory / "train.py"),
            "--checkpoint",
            str(directory / first["payload"]["path"]),
            "--resume",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    last = [e for e in events(resumed) if e["kind"] == "checkpoint"][-1]
    actual = torch.load(
        resumed / last["payload"]["path"], map_location="cpu", weights_only=True
    )
    assert actual["step"] == expected["step"] == 4
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
    replayed = [
        json.loads(line)
        for line in (resumed / "consumed-rank-0.jsonl").read_text().splitlines()
    ]
    assert replayed == original_trace[1:]


def test_downloaded_package_has_code_settings_and_runs_from_other_cwd(tmp_path):
    service, directory, stage = make_hook_run(tmp_path)
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        response = client.get(f"/api/training/runs/{directory.name}/package")
        assert response.status_code == 200, response.text
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert "config.json" not in archive.namelist()
            assert "weave_support/launch.json" in archive.namelist()
            extracted = tmp_path / "extracted"
            archive.extractall(extracted)
    result = subprocess.run(
        [sys.executable, str(extracted / "train.py")],
        cwd=tmp_path,
        env={**os.environ, "USE_TORCH": "1"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((extracted / "runs").glob("*/checkpoints/*.pt"))
    assert not (extracted / "events.jsonl").exists()


PARALLEL_SCRIPT = """
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from weave_support.model import Model
from weave_support.trainer import Trainer, TrainingConfig, StepResult

config = TrainingConfig(devices=[], strategy=STRATEGY, max_epochs=None, max_optimizer_steps=2,
    gradient_accumulation_steps=2, gradient_clip=1000,
    checkpoint_every=2, eval_every=2, sample_prompts=[])

class Pairs(Dataset):
    def __len__(self): return 2
    def __getitem__(self, index):
        x = torch.tensor([1, 2, 3, 4] if index == 0 else [4, 3, 2, 1])
        y = torch.tensor([2, 3, 4, 5] if index == 0 else [3, -100, -100, -100])
        return {'input_ids': x, 'labels': y}

def data(tokenizer, config):
    dataset = Pairs()
    parallel = int(os.environ.get('WORLD_SIZE', '1')) > 1
    sampler = DistributedSampler(dataset, shuffle=False) if parallel else None
    return DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler), None

def optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=0.1)

def step(model, batch, device):
    logits = model(batch['input_ids'].to(device))
    targets = batch['labels'].to(device)
    loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(), ignore_index=-100)
    return StepResult(loss, {'custom': 3.0}, weight=(targets != -100).sum().item())

if __name__ == '__main__':
    Trainer(model=Model, config=config, dataloaders_fn=data,
            optimizer_fn=optimizer, training_step_fn=step).train()
"""


def test_custom_hooks_cpu_ddp_matches_single_with_uneven_label_counts(tmp_path):
    states = []
    for name, strategy in [("single", "single"), ("parallel", "ddp")]:
        path = tmp_path / name
        path.mkdir()
        script = textwrap.dedent(PARALLEL_SCRIPT).replace("STRATEGY", repr(strategy))
        service, directory, _ = make_hook_run(path, script)
        execute(directory)
        checkpoint = [e for e in events(directory) if e["kind"] == "checkpoint"][-1]
        states.append(
            torch.load(
                directory / checkpoint["payload"]["path"],
                weights_only=True,
                map_location="cpu",
            )
        )
        metric = [e["payload"] for e in events(directory) if e["kind"] == "metrics"][-1]
        assert metric["train/custom"] == pytest.approx(3.0)
        assert metric["lr/group_0"] == 0.1
    for key, value in states[0]["model"].items():
        torch.testing.assert_close(states[1]["model"][key], value, rtol=2e-5, atol=2e-6)


def test_named_project_rename_and_base_directory_migration_preserve_files(
    tmp_path, monkeypatch
):
    first = tmp_path / "projects"
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(first))
    store = WorkspaceStore(tmp_path / "data/training")
    a = store.project_directory("project-a", "Hello there")
    a.mkdir(parents=True)
    atomic_json(a / "project.json", {"id": "project-a", "name": "Hello there"})
    (a / "notes.txt").write_text("my SSH edits")
    b = store.project_directory("project-b", "Hello there")
    assert b.name != a.name and b.name.startswith("hello-there-")
    renamed = store.project_directory("project-a", "Renamed model")
    atomic_json(renamed / "project.json", {"id": "project-a", "name": "Renamed model"})
    assert renamed.name == "renamed-model" and not a.exists()
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(tmp_path / "other"))
    moved = store.project_directory("project-a")
    assert (moved / "notes.txt").read_text() == "my SSH edits"
    assert moved.name == "renamed-model" and not renamed.exists()


@pytest.mark.skipif(
    not os.environ.get("WEAVE_CUDA_PYTHON"),
    reason="Set WEAVE_CUDA_PYTHON for real GPU hooks acceptance",
)
def test_hook_trainer_split_ddp_and_separate_eval_on_real_gpus(tmp_path):
    python = os.environ["WEAVE_CUDA_PYTHON"]
    states = []
    for strategy, devices in [
        ("single", [0]),
        ("split", [0, 1]),
        ("ddp", [0, 1]),
        ("train_eval", [0]),
    ]:
        target = tmp_path / strategy
        target.mkdir()
        script = (
            textwrap.dedent(PARALLEL_SCRIPT)
            .replace("STRATEGY", repr(strategy))
            .replace("devices=[]", f"devices={devices!r}")
        )
        if strategy == "split":
            script = script.replace(
                "max_epochs=None, max_optimizer_steps=2,",
                "max_epochs=None, max_optimizer_steps=2, placement={'embedding': 0, 'shared_transformer': 1, 'vocabulary': 0},",
            )
        if strategy == "train_eval":
            script = script.replace(
                "max_epochs=None, max_optimizer_steps=2,",
                "max_epochs=None, max_optimizer_steps=2, evaluation_device=1,",
            )
            script = script.replace(
                "return DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler), None",
                "loader = DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler)\n    return loader, loader",
            )
        service, directory, _ = make_hook_run(target, script)
        result = subprocess.run(
            [python, "train.py"],
            cwd=directory,
            env={
                **os.environ,
                "WEAVE_RUN_DIR": str(directory),
                "USE_TORCH": "1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
            capture_output=True,
            text=True,
            timeout=150,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        checkpoint = events(directory, kind="checkpoint")[-1]
        states.append(
            torch.load(
                directory / checkpoint["payload"]["path"],
                map_location="cpu",
                weights_only=True,
            )
        )
        if strategy == "train_eval":
            evaluation = [
                e
                for e in events(directory)
                if e.get("channel") == "evaluation" and e["kind"] == "metrics"
            ]
            assert evaluation, (directory / "evaluation.log").read_text()
            assert "eval/loss" in evaluation[-1]["payload"]
    for state in states[1:]:
        for key, value in states[0]["model"].items():
            torch.testing.assert_close(state["model"][key], value, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(
    not os.environ.get("WEAVE_CUDA_PYTHON"),
    reason="Set WEAVE_CUDA_PYTHON for FP16 acceptance",
)
def test_hook_fp16_and_optional_data_and_batch_hashes(tmp_path):
    from backend.training.weave_training import file_hash

    script = filled_starter(
        tmp_path / "rows.jsonl", devices=[1], precision="fp16", max_optimizer_steps=2
    )
    script = script.replace(
        "sample_prompts=[],",
        'sample_prompts=[], data_fingerprint="sha256", trace_batches="sha256",',
    )
    service, directory, _ = make_hook_run(tmp_path, script)
    result = subprocess.run(
        [os.environ["WEAVE_CUDA_PYTHON"], "train.py"],
        cwd=directory,
        env={
            **os.environ,
            "WEAVE_RUN_DIR": str(directory),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "USE_TORCH": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((directory / "datasets.json").read_text())
    assert manifest["train-0"]["sha256"] == file_hash(tmp_path / "rows.jsonl")
    trace = json.loads(
        (directory / "consumed-rank-0.jsonl").read_text().splitlines()[-1]
    )
    assert len(trace["batches"][0]["sha256"]) == 64
    checkpoint = events(directory, kind="checkpoint")[-1]
    state = torch.load(
        directory / checkpoint["payload"]["path"], weights_only=True, map_location="cpu"
    )
    assert state["step"] == 2
    assert state["scaler"] and all(
        torch.isfinite(value).all() for value in state["model"].values()
    )


def test_event_readers_close_sqlite_files_without_waiting_for_gc(tmp_path):
    import gc
    from backend.training.weave_training import Run

    if not Path("/proc/self/fd").is_dir():
        pytest.skip("Linux descriptor inspection")
    with Run(tmp_path) as run:
        run.log({"loss": 1.0}, step=1)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(20):
            assert events(tmp_path)
        opened = []
        for descriptor in Path("/proc/self/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if str(tmp_path / "event-index.sqlite") in target:
                opened.append(target)
        assert opened == []
    finally:
        if was_enabled:
            gc.enable()


def test_epoch_limits_partial_accumulation_cosine_and_resume(tmp_path):
    script = filled_starter(
        tmp_path / "rows.jsonl",
        micro_batch_size=16,
        gradient_accumulation_steps=3,
        max_epochs=3,
        max_optimizer_steps=None,
        checkpoint_every=999,
        eval_every=999,
    )
    # The test loader owns both its chunk size and its length estimate.
    script = script.replace(
        "dataset = TokenStream(",
        "class SizedStream(TokenStream):\n        def __len__(self): return 50\n    dataset = SizedStream(",
    )
    service, directory, _ = make_hook_run(tmp_path, script)
    atomic_json(
        directory / "commands/epoch-snapshot.json",
        {"id": "epoch-snapshot", "kind": "snapshot"},
    )
    execute(directory)
    records = events(directory)
    progress = [e["payload"] for e in records if e["kind"] == "progress"][-1]
    assert progress["epochs_completed"] == 3
    assert progress["optimizer_updates"] == 6
    assert progress["batches_processed"] == 12
    assert progress["tokens_processed"] == 600
    assert progress["total"] == 6
    checkpoints = events(directory, kind="checkpoint")
    expected = torch.load(
        directory / checkpoints[-1]["payload"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    assert expected["scheduler"]["_last_lr"][0] == pytest.approx(3e-5)
    assert expected["rank_state"][0]["loader"]["epoch"] == 3
    history = [
        json.loads(line)
        for line in (directory / "consumed-rank-0.jsonl").read_text().splitlines()
    ]
    assert [len(entry["batches"]) for entry in history] == [3, 1] * 3
    assert [entry["epoch"] for entry in history] == [0, 0, 1, 1, 2, 2]
    resumed = tmp_path / "epoch-resume"
    result = subprocess.run(
        [
            sys.executable,
            str(directory / "train.py"),
            "--checkpoint",
            str(directory / checkpoints[0]["payload"]["path"]),
            "--resume",
        ],
        cwd=tmp_path,
        env={**os.environ, "WEAVE_RUN_DIR": str(resumed), "USE_TORCH": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    last = events(resumed, kind="checkpoint")[-1]
    actual = torch.load(
        resumed / last["payload"]["path"], map_location="cpu", weights_only=True
    )
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
    assert actual["rank_state"][0]["batches_processed"] == 12


def test_unsized_epoch_loader_requires_duration_for_default_cosine(tmp_path):
    script = filled_starter(
        tmp_path / "rows.jsonl", max_epochs=2, max_optimizer_steps=None
    )
    service, directory, _ = make_hook_run(tmp_path, script)
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=directory,
        capture_output=True,
        text=True,
        env={**os.environ, "USE_TORCH": "1"},
        timeout=120,
    )
    assert result.returncode != 0
    assert (
        "Cosine decay needs len(train_loader) or max_optimizer_steps" in result.stderr
    )
    assert not list(directory.glob("checkpoints/*.pt"))


def test_fresh_starter_has_no_project_data_or_sequence_policy():
    source = starter_script()
    assert "encode_record" not in source and "DATA_PATTERN" not in source
    assert "sequence_length" not in source and "no_decay" not in source
    assert "Add your data loaders here." in source
    assert "max_epochs=3" in source and "max_optimizer_steps=None" in source
    assert "min_learning_rate=" in source and "CosineAnnealingLR" in source


def test_unsized_finite_loader_can_train_by_epochs_without_step_budget(tmp_path):
    script = filled_starter(
        tmp_path / "rows.jsonl",
        micro_batch_size=16,
        gradient_accumulation_steps=3,
        max_epochs=2,
        max_optimizer_steps=None,
        checkpoint_every=999,
        eval_every=999,
    ).replace("scheduler_fn=configure_scheduler,", "scheduler_fn=None,")
    service, directory, _ = make_hook_run(tmp_path, script)
    execute(directory)
    progress = events(directory, kind="progress")[-1]["payload"]
    assert progress["total"] is None
    assert progress["epochs_completed"] == 2
    assert progress["optimizer_updates"] == 4
    assert progress["batches_processed"] == 8
    assert progress["tokens_processed"] == 400


def test_background_cpu_validation_overlaps_updates_and_keeps_latest_job(tmp_path):
    script = filled_starter(
        tmp_path / "rows.jsonl",
        max_optimizer_steps=5,
        checkpoint_every=999,
        eval_every=1,
    )
    script = script.replace("async_evaluation=False", "async_evaluation=True")
    script = script.replace(
        "return DataLoader(dataset, batch_size=config.micro_batch_size, num_workers=0), None",
        "return DataLoader(dataset, batch_size=config.micro_batch_size, num_workers=0), [next(iter(DataLoader(dataset, batch_size=1)))]",
    )
    # A deterministic gate: validation cannot finish until training has reached
    # the final update. An inline implementation would fail this test.
    script = script.replace(
        "def training_step(model, batch, device):",
        """def validation_step(model, batch, device):
    import json, os, time
    from pathlib import Path
    root = Path(os.environ['WEAVE_RUN_DIR'])
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        lines = (root / 'events.jsonl').read_text().splitlines()
        if any(json.loads(line)['kind'] == 'metrics' and json.loads(line)['step'] == 5 for line in lines):
            assert device == 'cpu'
            return training_step(model, batch, device)
        time.sleep(0.05)
    raise RuntimeError('Training stalled waiting for validation')


def training_step(model, batch, device):""",
    )
    script = script.replace(
        "training_step_fn=training_step,",
        "training_step_fn=training_step,\n        evaluation_step_fn=validation_step,",
    )
    service, directory, _ = make_hook_run(tmp_path, script)
    execute(directory)
    records = events(directory)
    phases = [e for e in records if e["kind"] == "evaluation"]
    assert not [e for e in phases if e["payload"]["state"] == "failed"]
    running = [
        e["payload"]["checkpoint_step"]
        for e in phases
        if e["payload"]["state"] == "running"
    ]
    assert running == [1, 5], running
    evaluations = [
        e for e in records if e["channel"] == "evaluation" and e["kind"] == "metrics"
    ]
    assert [e["step"] for e in evaluations] == [1, 5]
    assert all("eval/loss" in e["payload"] for e in evaluations)
    assert not list((directory / "evaluation-weights").glob("*.pt"))
    checkpoints = events(directory, kind="checkpoint")
    assert (
        len(checkpoints) == 1
    ), "Validation should not serialize the optimizer at every update"
    assert events(directory, kind="progress")[-1]["payload"]["optimizer_updates"] == 5


def test_async_validation_is_independent_of_cpu_ddp_and_gui_override(tmp_path):
    script = textwrap.dedent(PARALLEL_SCRIPT).replace("STRATEGY", repr("ddp"))
    script = script.replace("eval_every=2", "eval_every=1")
    script = script.replace(
        "parallel = int(os.environ.get('WORLD_SIZE', '1')) > 1",
        "parallel = config.resolved_strategy == 'ddp'",
    )
    script = script.replace(
        "return DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler), None",
        "loader = DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler)\n    return loader, loader",
    )
    service, directory, _ = make_hook_run(tmp_path, script)
    # The launch override enables validation without editing the Python recipe.
    launch_path = directory / "weave_support/launch.json"
    launch = json.loads(launch_path.read_text())
    launch.update(evaluation_mode="async", evaluation_device="cpu")
    atomic_json(launch_path, launch)
    execute(directory)
    records = events(directory)
    assert [e["step"] for e in records if e["kind"] == "progress"][-1] == 2
    validations = [
        e for e in records if e["channel"] == "evaluation" and e["kind"] == "metrics"
    ]
    assert validations and validations[-1]["step"] == 2
    assert "eval/loss" in validations[-1]["payload"]
    assert not [
        e
        for e in records
        if e["kind"] == "warning" and e["payload"].get("code") == "evaluation-failed"
    ]


@pytest.mark.skipif(
    not os.environ.get("WEAVE_CUDA_PYTHON"), reason="Requires CUDA training environment"
)
@pytest.mark.parametrize("validation_device", ["cpu", 1])
def test_fp16_training_can_validate_asynchronously_on_cpu_or_selected_gpu(
    tmp_path, validation_device
):
    script = (
        textwrap.dedent(PARALLEL_SCRIPT)
        .replace("STRATEGY", repr("single"))
        .replace("devices=[]", "devices=[0]")
    )
    script = script.replace(
        "gradient_accumulation_steps=2,",
        f'gradient_accumulation_steps=2, precision="fp16", async_evaluation=True, evaluation_device={validation_device!r},',
    )
    script = script.replace("eval_every=2", "eval_every=1")
    script = script.replace(
        "return DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler), None",
        "loader = DataLoader(dataset, batch_size=1 if parallel else 2, sampler=sampler)\n    return loader, loader",
    )
    service, directory, _ = make_hook_run(tmp_path, script)
    result = subprocess.run(
        [os.environ["WEAVE_CUDA_PYTHON"], "train.py"],
        cwd=directory,
        env={
            **os.environ,
            "WEAVE_RUN_DIR": str(directory),
            "USE_TORCH": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = events(directory)
    assert not [
        e
        for e in records
        if e["kind"] == "evaluation" and e["payload"]["state"] == "failed"
    ], (directory / "evaluation.log").read_text()
    results = [
        e for e in records if e["kind"] == "metrics" and e["channel"] == "evaluation"
    ]
    assert results and results[-1]["step"] == 2
    config_files = list(directory.glob("evaluation-*/resolved-config.json"))
    assert config_files
    for path in config_files:
        resolved = json.loads(path.read_text())
        assert resolved["devices"] == (
            [] if validation_device == "cpu" else [validation_device]
        )
        assert resolved["precision"] == (
            "fp32" if validation_device == "cpu" else "fp16"
        )
