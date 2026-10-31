"""Run controls use safe trainer boundaries and preserve exact checkpoint state."""

import json
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.training.api import router
from backend.training.run_controls import (
    ValidationSettings,
    VALIDATION_UPGRADE_FROM,
    code_hash,
    validation_state,
)
from backend.training.service import events
from backend.training.weave_training import Run, atomic_json, file_hash
from test_training_hooks import make_hook_run, execute


def test_validation_commands_are_validated_pending_and_acknowledged(tmp_path):
    service, directory, _ = make_hook_run(tmp_path)
    atomic_json(directory / "status.json", {"state": "running"})
    with Run(directory) as run:
        run.capabilities(commands=True, live_validation=True)
        run.emit(
            "configuration",
            {
                "eval_every": 500,
                "eval_batches": 20,
                "async_evaluation": True,
                "evaluation_device": "cpu",
            },
        )
        settings = ValidationSettings(eval_every=10, eval_batches=2).model_dump()
        command = service.set_validation(directory.name, settings)
        state = validation_state(
            directory, events(directory), service.status(directory)
        )
        assert state["current"]["eval_every"] == 500
        assert state["pending"]["id"] == command["id"]
        with pytest.raises(ValueError, match="acknowledge"):
            service.set_validation(directory.name, settings)
        run.emit(
            "validation_settings",
            {
                **settings,
                "command_id": command["id"],
                "state": "applied",
                "effective_after_step": 17,
            },
        )
        state = validation_state(
            directory, events(directory), service.status(directory)
        )
        assert state["pending"] is None
        assert state["current"] == settings
        assert state["applied_at_step"] == 17
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        endpoint = f"/api/training/runs/{directory.name}/validation"
        assert (
            client.put(endpoint, json={**settings, "eval_every": 0}).status_code == 422
        )
        assert (
            client.put(endpoint, json={**settings, "device": True}).status_code == 422
        )
        assert client.put(endpoint, json={**settings, "device": 99}).status_code == 409
        assert (
            client.post(
                f"/api/training/runs/{directory.name}/restart", json={}
            ).status_code
            == 403
        )


def test_old_trainers_require_restart_instead_of_silent_command_drop(tmp_path):
    service, directory, _ = make_hook_run(tmp_path)
    atomic_json(directory / "status.json", {"state": "running"})
    with pytest.raises(ValueError, match="Restart from a checkpoint"):
        service.set_validation(directory.name, ValidationSettings().model_dump())
    assert list((directory / "commands").glob("*.json")) == []


def test_actual_trainer_applies_async_validation_command_and_records_settings(tmp_path):
    service, directory, _ = make_hook_run(tmp_path)
    settings = ValidationSettings(eval_every=2, eval_batches=1).model_dump()
    service.command(directory.name, "validation", settings=settings)
    execute(directory)
    applied = events(directory, kind="validation_settings")
    assert len(applied) == 1 and applied[0]["payload"]["effective_after_step"] == 0
    assert applied[0]["payload"]["eval_every"] == 2
    evaluations = events(directory, kind="evaluation")
    assert any(event["payload"]["state"] == "running" for event in evaluations)
    assert all(event["payload"]["device"] == "cpu" for event in evaluations)
    evaluator_configs = [
        event["payload"]
        for event in events(directory)
        if event["kind"] == "configuration" and event.get("channel") == "evaluation"
    ]
    assert evaluator_configs and all(
        cfg["eval_every"] == 2 and cfg["eval_batches"] == 1 and cfg["evaluation_only"]
        for cfg in evaluator_configs
    )


def test_checkpoint_restart_upgrades_only_trainer_and_preserves_exact_results(
    tmp_path, monkeypatch
):
    service, directory, stage = make_hook_run(tmp_path)
    frozen = (directory / "train.py").read_text()
    # Exercise a code-hash-changing framework migration with a known test revision.
    trainer = directory / "weave_support/trainer.py"
    trainer.write_text(
        trainer.read_text() + "\n# previous verified framework revision\n"
    )
    monkeypatch.setattr(
        "backend.training.run_controls.VALIDATION_UPGRADE_FROM",
        VALIDATION_UPGRADE_FROM | {file_hash(trainer)},
    )
    execute(directory)
    checkpoints = events(directory, kind="checkpoint")
    expected = torch.load(
        directory / checkpoints[-1]["payload"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    first = checkpoints[0]
    # Simulate a safe stop at the first checkpoint while retaining the full-run
    # weights above as an independent comparison for the resumed optimizer path.
    retained = [
        event
        for event in events(directory)
        if event["kind"] != "checkpoint" or event["id"] == first["id"]
    ]
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in retained)
    )
    for index in directory.glob("event-index.sqlite*"):
        index.unlink()
    info = json.loads((directory / "run.json").read_text())
    atomic_json(directory / "run.json", {**info, "execution": "local"})
    atomic_json(directory / "status.json", {"state": "stopped"})
    editable = Path(stage.script_path)
    editable.write_text("# User SSH edit: do not overwrite or execute this file\n")
    original_launch = service.launch

    def prepare(request, **kwargs):
        # Supervision is tested in the browser; here execute the real frozen
        # recipe synchronously so state_dict tensors can be compared exactly.
        request.stage.config.execution = "monitor"
        return original_launch(request, **kwargs)

    monkeypatch.setattr(service, "launch", prepare)
    result = service.restart(
        directory.name, ValidationSettings(mode="inline", eval_every=50).model_dump()
    )
    resumed = service.run_dir(result["id"])
    assert (resumed / "train.py").read_text() == frozen
    assert (
        editable.read_text()
        == "# User SSH edit: do not overwrite or execute this file\n"
    )
    proof = json.loads(
        (resumed / "weave_support/resume-compatibility.json").read_text()
    )
    assert proof["previous_code_sha256"] == code_hash(directory)
    assert proof["current_code_sha256"] == code_hash(resumed)
    assert proof["previous_code_sha256"] != proof["current_code_sha256"]
    assert service.restart(directory.name)["id"] == result["id"]
    execute(resumed)
    assert events(resumed, kind="metrics")[0]["step"] == first["step"] + 1
    last = events(resumed, kind="checkpoint")[-1]
    actual = torch.load(
        resumed / last["payload"]["path"], map_location="cpu", weights_only=True
    )
    assert actual["step"] == expected["step"] == 4
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], tensor, rtol=0, atol=0)
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert actual["scheduler"] == expected["scheduler"]
