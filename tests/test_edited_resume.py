"""Edited-script resumes preserve provenance and reject incompatible state."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from backend.training.service import events
from backend.training.weave_training import Run, atomic_json, file_hash
from test_training_hooks import execute, make_hook_run


def stopped_run(tmp_path, monkeypatch):
    service, source, stage = make_hook_run(tmp_path)
    execute(source)
    checkpoints = events(source, kind="checkpoint")
    expected = torch.load(
        source / checkpoints[-1]["payload"]["path"], weights_only=True
    )
    first = checkpoints[0]
    retained = [
        e for e in events(source) if e["kind"] != "checkpoint" or e["id"] == first["id"]
    ]
    (source / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in retained)
    )
    for index in source.glob("event-index.sqlite*"):
        index.unlink()
    info = json.loads((source / "run.json").read_text())
    atomic_json(source / "run.json", {**info, "execution": "local"})
    atomic_json(source / "status.json", {"state": "stopped"})
    launch = service.launch

    def prepare(request, **kwargs):
        request.stage.config.execution = "monitor"
        return launch(request, **kwargs)

    monkeypatch.setattr(service, "launch", prepare)
    return service, source, stage, expected, first


def failed_restore(directory):
    env = {**os.environ, "WEAVE_RUN_DIR": str(directory)}
    env.pop("WEAVE_REPORT_CONFIG", None)
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=directory,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode != 0, result.stdout
    assert not events(directory, kind="metrics")
    assert not events(directory, kind="checkpoint")
    atomic_json(directory / "status.json", {"state": "failed"})
    info = json.loads((directory / "run.json").read_text())
    atomic_json(directory / "run.json", {**info, "execution": "local"})
    return json.loads((directory / "resume-problem.json").read_text())


def test_edited_resume_snapshot_and_restored_optimizer_match_uninterrupted(
    tmp_path, monkeypatch
):
    service, source, stage, expected, first = stopped_run(tmp_path, monkeypatch)
    original = (source / "train.py").read_text()
    edited = original.replace(
        "StepResult(loss, weight=count)", "StepResult(loss + 0.0, weight=count)"
    )
    assert edited != original
    Path(stage.script_path).write_text(edited)
    helper = Path(stage.script_path).parent / "user_helper.py"
    helper.write_text("# frozen editable helper\n")
    result = service.restart(source.name, script_source="edited")
    resumed = service.run_dir(result["id"])
    revision = json.loads((resumed / "revision.json").read_text())
    assert revision["script_source"] == "edited"
    assert revision["checkpoint"]["run_id"] == source.name
    for name, digest in revision["files"].items():
        assert file_hash(resumed / name) == digest
    assert (resumed / "user_helper.py").read_text() == helper.read_text()
    assert (resumed / "train.py").read_text() == edited
    # Later SSH edits cannot change the frozen attempt.
    Path(stage.script_path).write_text("# later edit\n")
    execute(resumed)
    assert events(resumed, kind="metrics")[0]["step"] == first["step"] + 1
    actual = torch.load(
        resumed / events(resumed, kind="checkpoint")[-1]["payload"]["path"],
        weights_only=True,
    )
    for name in expected["model"]:
        torch.testing.assert_close(
            actual["model"][name], expected["model"][name], rtol=0, atol=0
        )
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert actual["scheduler"] == expected["scheduler"]
    assert (source / "train.py").read_text() == original


@pytest.mark.parametrize(
    "edit, message, retry_source",
    [
        (
            lambda s: s.replace("model=Model,", "model=lambda: torch.nn.Linear(4, 8),"),
            "state_dict",
            "previous",
        ),
        (
            lambda s: s.replace(
                "model.parameters(),", "reversed(list(model.parameters())),"
            ),
            "parameter names",
            "edited",
        ),
    ],
)
def test_incompatible_attempt_can_retry_previous_script_without_overwriting_edits(
    tmp_path, monkeypatch, edit, message, retry_source
):
    service, source, stage, _, _ = stopped_run(tmp_path, monkeypatch)
    original = (source / "train.py").read_text()
    edited = edit(original)
    assert edited != original
    Path(stage.script_path).write_text(edited)
    result = service.restart(source.name, script_source="edited")
    failed = service.run_dir(result["id"])
    problem = failed_restore(failed)
    assert message in problem["message"]
    assert service.restart_context(failed.name)["source_run"] == source.name
    retry_script = original
    if retry_source == "edited":
        retry_script = original + "\n# Fixed optimizer ordering after failed restore.\n"
        Path(stage.script_path).write_text(retry_script)
    recovered = service.restart(failed.name, script_source=retry_source)
    directory = service.run_dir(recovered["id"])
    assert (directory / "train.py").read_text() == retry_script
    assert Path(stage.script_path).read_text() == (
        edited if retry_source == "previous" else retry_script
    )
    assert (
        json.loads((directory / "revision.json").read_text())["requested_from"]
        == failed.name
    )
    execute(directory)
    assert events(directory, kind="metrics")[0]["step"] == 3


def test_rejected_syntax_is_snapshotted_and_does_not_block_correction(
    tmp_path, monkeypatch
):
    service, source, stage = make_hook_run(tmp_path)
    # The snapshot policy also covers rejected attempts before process startup.
    with Run(source) as run:
        path = source / "checkpoint.pt"
        path.write_bytes(b"placeholder")
        run.checkpoint(path, metadata={"exact_resume": True})
    info = json.loads((source / "run.json").read_text())
    atomic_json(source / "run.json", {**info, "execution": "local"})
    atomic_json(source / "status.json", {"state": "stopped"})
    Path(stage.script_path).write_text("def broken(:\n")
    with pytest.raises((ValueError, SyntaxError)):
        service.restart(source.name, script_source="edited")
    attempts = list((source / "resume-attempts").glob("*/attempt.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["state"] == "rejected"
    assert (attempts[0].parent / "requested.py").read_text() == "def broken(:\n"
    assert not (source / "restart.json").exists()


def test_older_checkpoint_requires_named_optimizer_mapping_for_edited_script(
    tmp_path, monkeypatch
):
    service, source, stage, _, first = stopped_run(tmp_path, monkeypatch)
    path = source / first["payload"]["path"]
    state = torch.load(path, weights_only=True)
    state.pop("optimizer_structure")
    torch.save(state, path)
    records = events(source)
    for event in records:
        if event["id"] == first["id"]:
            event["payload"]["sha256"] = file_hash(path)
    (source / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in records))
    for index in source.glob("event-index.sqlite*"):
        index.unlink()
    Path(stage.script_path).write_text(
        (source / "train.py").read_text() + "\n# edited\n"
    )
    result = service.restart(source.name, script_source="edited")
    problem = failed_restore(service.run_dir(result["id"]))
    assert "Resume once with the previous script" in problem["message"]
