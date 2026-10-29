"""Real training and provenance checks, not mocked optimizer behavior."""

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest
import torch

from backend.templates import node, edge, transformer
from backend.training.service import (
    TrainingService,
    TrainingConfig,
    Stage,
    Launch,
    events,
    build_snapshot,
    import_snapshot,
    model_metadata,
)
from backend.training.weave_training import Run, file_hash, instrumentation_warnings

ROOT = Path(__file__).resolve().parents[1]


def tiny_project():
    nodes = [
        node(
            "ids",
            "input",
            "Token IDs",
            0,
            0,
            {"index": 0, "shape": ["B", "T"], "dtype": "int64"},
        ),
        node("emb", "embedding", "Embedding", 100, 0, {"vocab": 8, "width": 8}),
        node(
            "body",
            "loop",
            "Shared transformer",
            200,
            0,
            {"iterations": 2, "shared": True},
            transformer(heads=2, kv=2),
        ),
        node("head", "head", "Vocabulary", 300, 0, {"out": 8}),
        node("out", "output", "Logits", 400, 0),
    ]
    return {
        "version": 1,
        "id": "training-test",
        "name": "Tiny training model",
        "description": "",
        "config": {"vocab": 8, "width": 8, "heads": 2, "ffn_width": 16},
        "graph": {
            "nodes": nodes,
            "edges": [
                edge("ids", "emb"),
                edge("emb", "body"),
                edge("body", "head"),
                edge("head", "out"),
            ],
        },
    }


def make_run(tmp_path, *, steps=8, config=None, script=None):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    dataset = tmp_path / "train.jsonl"
    if not dataset.exists():
        dataset.write_text(
            "\n".join(
                json.dumps({"input_ids": [1, 2, 3, 4, 1, 2, 3, 4]}) for _ in range(8)
            )
        )
    settings = TrainingConfig(
        dataset=str(dataset),
        max_steps=steps,
        checkpoint_every=steps,
        learning_rate=0.02,
        warmup_steps=0,
        batch_size=2,
        sequence_length=8,
        **(config or {}),
    )
    source = script or (ROOT / "backend/training/template.py").read_text()
    # Monitor mode lets this test own process lifetime and interpreter.
    settings.execution = "monitor"
    info = service.launch(
        Launch(
            project=tiny_project(),
            stage=Stage(id="stage", name="Tiny stage", config=settings, script=source),
        )
    )
    return service, service.run_dir(info["id"])


def run_script(directory, python=sys.executable, extra=None):
    environment = dict(
        os.environ, WEAVE_RUN_DIR=str(directory), CUDA_DEVICE_ORDER="PCI_BUS_ID"
    )
    if extra:
        environment.update(extra)
    result = subprocess.run(
        [python, "train.py", "--config", "config.json"],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return events(directory)


def test_real_training_and_snapshot_roundtrip(tmp_path):
    service, directory = make_run(tmp_path)
    records = run_script(directory)
    losses = [
        e["payload"]["train/loss"]
        for e in records
        if e["kind"] == "metrics" and "train/loss" in e["payload"]
    ]
    assert losses[-1] < losses[0]
    checkpoints = [e for e in records if e["kind"] == "checkpoint"]
    assert len(checkpoints) == 1
    checkpoint = torch.load(
        directory / checkpoints[-1]["payload"]["path"], weights_only=True
    )
    assert checkpoint["step"] == 8
    assert checkpoint["optimizer"]["state"]
    assert len((directory / "consumed-rank-0.jsonl").read_text().splitlines()) == 8
    snapshot = build_snapshot(directory, checkpoints[-1], True)
    with (directory / snapshot).open("rb") as stream:
        restored = import_snapshot(service, stream)
    imported = service.run_dir(restored["id"])
    assert (
        file_hash(imported / checkpoints[-1]["payload"]["path"])
        == checkpoints[-1]["payload"]["sha256"]
    )
    assert (imported / "source-data/dataset/train.jsonl").exists()


def test_resume_matches_uninterrupted_and_new_stage_resets(tmp_path):
    # Separate source roots preserve immutable run records.
    (tmp_path / "part").mkdir()
    (tmp_path / "full").mkdir()
    service, first = make_run(tmp_path / "part", steps=3)
    part_events = run_script(first)
    checkpoint = next(
        e["payload"]["path"] for e in part_events if e["kind"] == "checkpoint"
    )
    source = json.loads((first / "stage.json").read_text())
    config = source["config"]
    config.update(
        start_mode="resume",
        parent_run=first.name,
        parent_checkpoint=checkpoint,
        max_steps=6,
    )
    resumed_info = service.launch(Launch(project=tiny_project(), stage=Stage(**source)))
    resumed = service.run_dir(resumed_info["id"])
    resumed_events = run_script(resumed)
    provenance = resumed / "lineage" / first.name
    assert (provenance / "train.py").read_text() == (first / "train.py").read_text()
    assert (
        json.loads((provenance / "selected-checkpoint.json").read_text())["step"] == 3
    )
    assert (provenance / "consumed-rank-0.jsonl").is_file()
    _, full = make_run(tmp_path / "full", steps=6)
    full_events = run_script(full)

    def last_state(directory, records):
        path = [e["payload"]["path"] for e in records if e["kind"] == "checkpoint"][-1]
        return torch.load(directory / path, weights_only=True)

    resumed_state, full_state = last_state(resumed, resumed_events), last_state(
        full, full_events
    )
    for name, value in full_state["model"].items():
        torch.testing.assert_close(resumed_state["model"][name], value, rtol=0, atol=0)
    config.update(start_mode="weights", max_steps=1)
    weights_info = service.launch(Launch(project=tiny_project(), stage=Stage(**source)))
    weights = service.run_dir(weights_info["id"])
    records = run_script(weights)
    assert last_state(weights, records)["step"] == 1


def test_no_supervised_tokens_reports_clear_failure(tmp_path):
    service, directory = make_run(tmp_path)
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"input_ids": [1, 2, 3], "labels": [-100, -100, -100]}))
    config = json.loads((directory / "config.json").read_text())
    config["dataset_sha256"] = file_hash(path)
    (directory / "config.json").write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=directory,
        env=dict(os.environ, WEAVE_RUN_DIR=str(directory)),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "no supervised next-token targets" in result.stderr


def test_reporting_warnings_and_checksum_failure(tmp_path):
    with Run(tmp_path / "run") as run:
        run.log({"loss": float("nan")}, step=1)
        run.sample(prompt="hello", output="world", checkpoint="checkpoints/test.pt")
    assert any(e["kind"] == "warning" for e in events(tmp_path / "run"))
    assert instrumentation_warnings("import weave_training")
    assert not instrumentation_warnings(
        "r.log({}); r.sample(); r.checkpoint(); r.poll_commands()"
    )
    service, directory = make_run(tmp_path)
    snapshot = build_snapshot(directory)
    damaged = io.BytesIO()
    with zipfile.ZipFile(directory / snapshot) as original, zipfile.ZipFile(
        damaged, "w"
    ) as output:
        for name in original.namelist():
            output.writestr(
                name, b"changed" if name == "train.py" else original.read(name)
            )
    damaged.seek(0)
    with pytest.raises(ValueError, match="checksum"):
        import_snapshot(service, damaged)


def test_data_changes_are_detected_before_training(tmp_path):
    _, directory = make_run(tmp_path)
    (tmp_path / "train.jsonl").write_text('{"input_ids":[2,2,2]}')
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=directory,
        env=dict(os.environ, WEAVE_RUN_DIR=str(directory)),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Dataset contents changed" in result.stderr


def test_sft_masks_real_chat_and_records_generated_outputs(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from backend.tokenizers import TokenizerService

    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "user": 1,
                "assistant": 2,
                "hello": 3,
                "world": 4,
                "end": 5,
                "tools": 6,
                "done": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = "{% for message in messages %}{{ message['role'] }} {% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }} {% endgeneration %}{% else %}{{ message['content'] }} {% endif %}{% endfor %}"
    local = tmp_path / "tokenizer-source"
    fast.save_pretrained(local)
    tokenizers = TokenizerService(tmp_path / "tokenizers")
    info = tokenizers.load(local)
    project = tiny_project()
    project["graph"]["nodes"][0]["data"]["tokenizer"] = info
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            }
        )
    )
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    config = TrainingConfig(
        recipe="sft",
        execution="monitor",
        dataset=str(dataset),
        eval_dataset=str(dataset),
        max_steps=2,
        checkpoint_every=1,
        generation_tokens=3,
        prompts=["hello"],
    )
    result = service.launch(
        Launch(
            project=project,
            stage=Stage(
                id="sft",
                name="SFT",
                config=config,
                script=(ROOT / "backend/training/template.py").read_text(),
            ),
        )
    )
    directory = service.run_dir(result["id"])
    records = run_script(directory)
    prepared = json.loads(
        (directory / "prepared-train.jsonl").read_text().splitlines()[0]
    )
    assert prepared["labels"][:3] == [-100, -100, -100]
    assert prepared["labels"][3] == 4
    samples = [event for event in records if event["kind"] == "sample"]
    assert len(samples) == 2
    assert all(sample["payload"]["checkpoint"] for sample in samples)
    assert all(isinstance(sample["payload"]["output"], str) for sample in samples)
    assert any(
        "eval/loss" in event["payload"]
        for event in records
        if event["kind"] == "metrics"
    )


def test_snapshot_request_uses_safe_checkpoint_and_filters_consumed_trace(tmp_path):
    from backend.training.weave_training import atomic_json

    service, directory = make_run(tmp_path, steps=2)
    atomic_json(directory / "status.json", {"state": "running"})
    request = service.command(directory.name, "snapshot")
    records = run_script(directory)
    service.complete_snapshots(directory)
    snapshot = directory / "snapshots" / f"snapshot-{request['id']}.zip"
    assert snapshot.exists()
    with zipfile.ZipFile(snapshot) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        checkpoint = torch.load(
            io.BytesIO(archive.read(manifest["checkpoint"])),
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["step"] == 1
        consumed = [
            json.loads(line)
            for line in archive.read("consumed-rank-0.jsonl").splitlines()
        ]
        assert all(record["step"] <= 1 for record in consumed)
        assert "prepared-train.jsonl" not in archive.namelist()
        assert not any(name.startswith("source-data/") for name in archive.namelist())
    assert any(
        request["id"] in e["payload"].get("metadata", {}).get("snapshot_requests", [])
        for e in records
        if e["kind"] == "checkpoint"
    )


def test_event_index_recovers_partial_and_malformed_records(tmp_path):
    with Run(tmp_path) as run:
        run.log({"loss": 1.0}, step=1)
    first = events(tmp_path)
    path = tmp_path / "events.jsonl"
    with path.open("a") as stream:
        stream.write('{"version":1,"kind":"metrics","payload":{},"id":[]}\n')
        stream.write('{"version":1,')
    assert events(tmp_path) == first
    with path.open("a") as stream:
        stream.write(
            '"id":"later","time":1,"kind":"metrics","step":2,"tokens":4,"payload":{"loss":0.5}}\n'
        )
    assert events(tmp_path, after=first[-1]["id"])[-1]["id"] == "later"
