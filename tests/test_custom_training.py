"""Custom loops keep their own algorithm while using portable run infrastructure."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import pytest
import torch

from backend.training.weave_training import Run, atomic_json
from backend.training.custom_training import (
    Checkpointer,
    launch_settings,
)
from backend.training.custom_evaluation import AsyncEvaluator, EvaluationResult
from backend.training.service import (
    events,
    Stage,
    TrainingConfig,
    TrainingService,
    Launch,
)
from backend.training.stage_settings import preview_stage_settings
from backend.training.chat import checkpoint_catalog
from test_training import tiny_project


def test_configure_counters_and_streamed_trajectory_table(tmp_path):
    with Run(tmp_path) as run:
        run.configure(
            {"devices": [], "precision": "fp32"},
            metrics={
                "eval/success": {
                    "label": "Task success",
                    "direction": "max",
                    "min": 0,
                    "max": 1,
                },
            },
            primary_metric="eval/success",
            step_unit="policy updates",
            headline_metrics=["eval/success"],
        )
        run.capabilities(metrics=True, commands=True)
        run.capabilities(checkpoint=False)
        run.log(
            {"train/reward": -0.3}, step=3, counters={"episodes": 20, "rollouts": 5}
        )
        path = run.table(
            "Trajectories",
            (
                {"task": index, "call": "<call/>", "reward": index % 2}
                for index in range(200)
            ),
            metadata={"catalog": "mobile-v1"},
        )
        assert len(path.read_text().splitlines()) == 200
        run.artifact(path, name="Raw trajectories")
    records = events(tmp_path)
    config = next(item for item in records if item["kind"] == "configuration")
    assert config["payload"]["reporting"]["step_unit"] == "policy updates"
    capabilities = [
        item["payload"] for item in records if item["kind"] == "capabilities"
    ][-1]
    assert capabilities == {"metrics": True, "commands": True, "checkpoint": False}
    table = next(item for item in records if item["kind"] == "table")
    assert table["counters"] == {"episodes": 20, "rollouts": 5}
    assert table["payload"]["rows"] == 200
    assert len(table["payload"]["preview"]) == 20


def test_bad_contracts_and_paths_reject_without_publishing_partial_tables(tmp_path):
    with Run(tmp_path) as run:
        with pytest.raises(ValueError, match="direction"):
            run.configure({}, primary_metric="reward")
        with pytest.raises(ValueError, match="nonnegative"):
            run.log({"reward": 1}, step=1, counters={"episodes": -2})
        with pytest.raises(ValueError, match="booleans"):
            run.capabilities(commands="yes")
        with pytest.raises(ValueError, match="dictionaries"):
            run.table("Broken", [{"task": "x"}, 3])
        assert not list((tmp_path / "artifacts/tables").glob("*.jsonl"))
        with pytest.raises(ValueError):
            run.artifact(tmp_path.parent / "outside")
        path = run.table(
            "Large preview",
            [{f"c{j}": "x" * 10000 for j in range(64)} for _ in range(5)],
        )
        assert path.is_file()
    assert (
        len(
            json.dumps(
                next(item for item in events(tmp_path) if item["kind"] == "table")[
                    "payload"
                ]
            ).encode()
        )
        < 100000
    )


class Cursor:
    def __init__(self, position=0):
        self.position = position

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def test_atomic_checkpoint_restores_next_update_optimizer_rng_and_cursor(tmp_path):
    torch.manual_seed(7)
    random.seed(9)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    cursor = Cursor(12)

    def update():
        optimizer.zero_grad()
        loss = model(torch.randn(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach().clone()

    with Run(tmp_path, optimizer=optimizer) as run:
        update()
        run.log_loss(1.0)
        run.set_counters(rollouts=4)
        checkpointer = Checkpointer(
            run,
            components={"model": model, "optimizer": optimizer, "scheduler": scheduler},
            rank_local={"cursor": cursor},
            exact_resume=True,
        )
        path = checkpointer.save(extra={"reward_version": "v1"})
        assert "model" in torch.load(path, weights_only=True)
        expected_random = random.random()
        expected_loss = update()
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        cursor.position = 999
        extra = checkpointer.load(path)
        assert cursor.position == 12
        assert extra == {"reward_version": "v1"}
        assert random.random() == expected_random
        assert run.step == 1
        actual_loss = update()
        assert run.step == 2  # Optimizer hook follows the restored counter.
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_shape_mismatch_rejects_before_mutation_and_previous_model_can_retry(tmp_path):
    with Run(tmp_path) as run:
        original = torch.nn.Linear(3, 2)
        path = Checkpointer(run, components={"model": original}).save()
        incompatible = torch.nn.Linear(4, 2)
        before = incompatible.weight.detach().clone()
        with pytest.raises(ValueError, match="tensor shapes"):
            Checkpointer(run, components={"model": incompatible}).load(path)
        torch.testing.assert_close(incompatible.weight, before)
        replacement = torch.nn.Linear(3, 2)
        Checkpointer(run, components={"model": replacement}).load(
            path, weights_only=True
        )
        torch.testing.assert_close(replacement.weight, original.weight)
        with pytest.raises(RuntimeError, match="Unsupported checkpoint value"):
            Checkpointer(run, components={"model": original}).save(
                "bad.pt", extra={"bad": object()}
            )
        assert not (tmp_path / "bad.pt").exists()


def test_stop_saves_and_acknowledges_snapshot_requests_only_when_handled(tmp_path):
    with Run(tmp_path) as run:
        model = torch.nn.Linear(2, 2)
        checkpointer = Checkpointer(run, components={"model": model}, exact_resume=True)
        atomic_json(tmp_path / "commands/stop.json", {"id": "stop", "kind": "stop"})
        atomic_json(
            tmp_path / "commands/snapshot.json", {"id": "snapshot", "kind": "snapshot"}
        )
        assert run.handle_commands(checkpointer.save)
        assert not run.handle_commands(checkpointer.save)
    records = events(tmp_path)
    checkpoint = next(item for item in records if item["kind"] == "checkpoint")
    assert set(checkpoint["payload"]["metadata"]["snapshot_requests"]) == {
        "stop",
        "snapshot",
    }
    assert len([item for item in records if item["kind"] == "command_ack"]) == 2
    assert any(item["kind"] == "stopped" for item in records)


_evaluation_calls = 0


def evaluate_snapshot(checkpoint, device):
    global _evaluation_calls
    _evaluation_calls += 1
    assert device == "cpu"
    return EvaluationResult(
        metrics={
            "eval/reward": float(checkpoint.read_text()),
            "eval/calls": _evaluation_calls,
        },
        samples=[{"prompt": "task", "output": "<no_call/>"}],
        tables={"Task results": [{"task": "test", "success": True}]},
    )


def stalled_evaluation(checkpoint, device):
    time.sleep(10)
    return {}


def wait_evaluation(evaluator):
    deadline = time.monotonic() + 20
    while evaluator.job is not None and time.monotonic() < deadline:
        result = evaluator.poll(raise_on_error=True)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError("Evaluation did not finish")


def test_persistent_async_evaluation_keeps_policy_step_and_training_counter(tmp_path):
    with Run(tmp_path) as run, AsyncEvaluator(run, evaluate_snapshot) as evaluator:
        run.log({"train/reward": 0.1}, step=4, counters={"episodes": 80})
        checkpoint = tmp_path / "policy.txt"
        checkpoint.write_text("0.75")
        assert evaluator.submit(checkpoint)
        run.log({"train/reward": 0.2}, step=10, counters={"episodes": 200})
        assert not evaluator.submit(checkpoint)
        first = wait_evaluation(evaluator)
        assert first.metrics["eval/calls"] == 1
        assert run.step == 10 and run.counters["episodes"] == 200
        worker_pid = evaluator.process.pid
        assert evaluator.submit(checkpoint, step=10)
        second = wait_evaluation(evaluator)
        assert second.metrics["eval/calls"] == 2 and evaluator.process.pid == worker_pid
    metrics = [
        item
        for item in events(tmp_path)
        if item["kind"] == "metrics" and item["channel"] == "evaluation"
    ]
    assert [item["step"] for item in metrics] == [4, 10]
    assert metrics[0]["counters"]["episodes"] == 80
    assert metrics[0]["checkpoint"] == "policy.txt"
    assert metrics[0]["policy_version"]


def test_async_timeout_is_visible_and_worker_can_be_closed(tmp_path):
    with (
        Run(tmp_path) as run,
        AsyncEvaluator(run, stalled_evaluation, timeout=0.05) as evaluator,
    ):
        path = tmp_path / "snapshot"
        path.write_text("frozen")
        evaluator.submit(path)
        time.sleep(0.1)
        with pytest.raises(RuntimeError, match="exceeded"):
            evaluator.poll(raise_on_error=True)
        assert evaluator.process is None
    assert any(
        item["kind"] == "evaluation_status" and item["payload"]["state"] == "failed"
        for item in events(tmp_path)
    )


def test_custom_preview_requires_explicit_supported_settings_and_launch_is_read_only(
    tmp_path, monkeypatch
):
    settings = TrainingConfig(recipe="custom")
    stage = Stage(id="custom", name="Custom", config=settings, script="print('hello')")
    assert preview_stage_settings(stage)["supported"] == []
    stage.script = 'WEAVE_SUPPORTED_SETTINGS = ("devices", "precision")\n'
    assert preview_stage_settings(stage)["supported"] == ["devices", "precision"]
    atomic_json(tmp_path / "config.json", {"devices": [1], "precision": "fp16"})
    monkeypatch.setenv("WEAVE_RUN_DIR", str(tmp_path))
    assert launch_settings({"devices": []}) == {"devices": [1], "precision": "fp16"}
    monkeypatch.delenv("WEAVE_RUN_DIR")
    assert launch_settings({"devices": []}) == {"devices": []}


def test_primary_metric_ranks_exact_checkpoints_and_ignores_wrong_policy_hash(tmp_path):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    run = service.launch(
        Launch(
            project=tiny_project(),
            stage=Stage(
                id="custom",
                name="Custom",
                config=TrainingConfig(recipe="custom", execution="monitor"),
                script="print('custom')",
            ),
        )
    )
    directory = service.run_dir(run["id"])
    with Run(directory) as report:
        report.configure(
            {},
            metrics={"eval/success": {"direction": "max"}},
            primary_metric="eval/success",
        )
        for step, reward in [(1, 0.8), (2, 0.4)]:
            path = directory / f"policy-{step}.pt"
            path.write_bytes(b"fake weights")
            report.log({"eval/success": reward}, step=step)
            report.checkpoint(path)
        # Same-step results for a different policy must never contaminate ranking.
        report.emit(
            "metrics",
            {"eval/success": 1.0},
            step=2,
            checkpoint="policy-2.pt",
            policy_version="wrong-hash",
            channel="evaluation",
        )
    assert checkpoint_catalog(directory)["recommended_checkpoint"] == "policy-1.pt"


def test_two_cpu_ranks_save_local_state_and_stop_together(tmp_path):
    script = tmp_path / "ddp.py"
    script.write_text("""
import os
from pathlib import Path
import torch
from backend.training.weave_training import Run, atomic_json
from backend.training.custom_training import distributed_context, Checkpointer
class Cursor:
    def __init__(self, rank): self.rank = rank
    def state_dict(self): return {"rank": self.rank}
    def load_state_dict(self, state): self.rank = state["rank"]
with distributed_context() as context, Run(os.environ["TEST_RUN_DIR"]) as run:
    model = context.wrap(torch.nn.Linear(2, 2))
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.zero_grad(); model(torch.ones(2, 2)).sum().backward(); optimizer.step()
    cursor = Cursor(context.rank)
    cp = Checkpointer(run, components={"model": model, "optimizer": optimizer}, rank_local={"cursor": cursor}, exact_resume=True)
    path = cp.save()
    # save has completed on rank zero before any rank attempts load.
    cursor.rank = 999
    cp.load(path)
    assert cursor.rank == context.rank
    if context.rank == 0: atomic_json(run.directory / "commands/stop.json", {"id": "stop", "kind": "stop"})
    torch.distributed.barrier()
    assert run.handle_commands(cp.save, commands=context.poll_commands(run))
""")
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "TEST_RUN_DIR": str(tmp_path / "run"),
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(script),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(item["kind"] == "stopped" for item in events(tmp_path / "run"))


def test_api_preserves_custom_contract_after_event_window_and_remote_tables(
    tmp_path, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.training.api import router
    from backend.training.reporting import ReportBatch, ReportStore

    monkeypatch.delenv("WEAVE_TRAINING_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/training/runs",
            json={
                "project": tiny_project(),
                "stage": {
                    "id": "custom",
                    "name": "Custom",
                    "script": "print('custom')",
                    "config": TrainingConfig(
                        recipe="custom", execution="monitor"
                    ).model_dump(),
                },
            },
        )
        assert response.status_code == 200, response.text
        directory = tmp_path / "training/runs" / response.json()["id"]
        with Run(directory) as run:
            run.capabilities(metrics=True, samples=False, commands=False)
            run.configure(
                {"devices": []},
                metrics={"eval/reward": {"direction": "max"}},
                primary_metric="eval/reward",
            )
            for step in range(10010):
                run.log({"train/reward": step / 10010}, step=step)
        detail = client.get(f'/api/training/runs/{response.json()["id"]}').json()
        assert detail["script_capabilities"]["commands"] is False
        assert any(item["kind"] == "configuration" for item in detail["events"])
        assert not any(
            "No generated examples" in message for message in detail["diagnostics"]
        )
        store = ReportStore()
        connection = store.connection(directory, "http://localhost:8173")
        with Run(tmp_path / "remote") as run:
            run.log({"reward": 1}, step=1, counters={"episodes": 2})
            run.table("Remote", [{"task": "a", "reward": 1}])
        remote = [
            json.loads(line)
            for line in (tmp_path / "remote/events.jsonl").read_text().splitlines()
        ]
        store.ingest(directory, connection["token"], ReportBatch(events=remote))
        remote_table = events(directory, kind="table")[-1]
        assert remote_table["payload"]["remote"] is True
        assert remote_table["counters"] == {"episodes": 2}


def test_custom_checkpoint_pause_resume_edited_failure_and_previous_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WEAVE_ALLOW_LOCAL_TRAINING", "1")
    fixture = (Path(__file__).parent / "fixtures/custom_run.py").read_text()
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    config = TrainingConfig(
        recipe="custom", execution="local", python=sys.executable, max_steps=4
    )
    stage = Stage(id="custom", name="Custom", config=config, script=fixture)
    run = service.launch(Launch(project=tiny_project(), stage=stage))
    directory = service.run_dir(run["id"])

    def finished(identifier):
        deadline = time.monotonic() + 45
        folder = service.run_dir(identifier)
        while time.monotonic() < deadline:
            state = service.status(folder)["state"]
            if state in ("completed", "failed", "stopped"):
                return state
            time.sleep(0.1)
        raise AssertionError((folder / "output.log").read_text())

    assert finished(run["id"]) == "completed", (directory / "output.log").read_text()
    assert service.restart_context(run["id"])["exact_resume"]
    # Save an edited script whose model has incompatible state, then try it.
    bad = stage.model_copy(
        update={
            "script": fixture.replace(
                "model = Model()", "model = torch.nn.Linear(3, 2)"
            )
        }
    )
    service.workspaces.save_stage(tiny_project()["id"], bad)
    child = service.restart(run["id"], script_source="edited")
    assert finished(child["id"]) == "failed"
    assert events(service.run_dir(child["id"]), kind="resume_problem")
    retried = service.restart(child["id"], script_source="previous")
    assert finished(retried["id"]) == "completed"
    child_folder = service.run_dir(retried["id"])
    assert (child_folder / "revision.json").exists()
    assert events(child_folder, kind="checkpoint")


def test_explicitly_unsupported_controls_fail_without_queuing_dead_requests(tmp_path):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    run = service.launch(
        Launch(
            project=tiny_project(),
            stage=Stage(
                id="custom",
                name="Custom",
                config=TrainingConfig(recipe="custom", execution="monitor"),
                script="print('custom')",
            ),
        )
    )
    directory = service.run_dir(run["id"])
    with Run(directory) as report:
        report.capabilities(commands=False, checkpoint=False)
        with pytest.raises(ValueError, match="cooperative stop"):
            service.command(run["id"], "stop")
        with pytest.raises(ValueError, match="snapshot handling"):
            service.command(run["id"], "snapshot")
        assert not list((directory / "commands").glob("*.json"))


def test_checkpoint_paths_are_unique_and_published_files_cannot_be_overwritten(
    tmp_path,
):
    with Run(tmp_path) as run:
        cp = Checkpointer(run, components={"model": torch.nn.Linear(2, 2)})
        first, second = cp.save(), cp.save()
        assert first != second
        original = first.read_bytes()
        with pytest.raises(RuntimeError, match="immutable"):
            cp.save(first)
        assert first.read_bytes() == original


@pytest.mark.parametrize("format", ["raw", "trainer"])
def test_new_custom_stage_can_initialize_from_existing_pytorch_weights(
    tmp_path, format
):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "pretrained.pt"
    torch.save(
        (
            model.state_dict()
            if format == "raw"
            else {"model": model.state_dict(), "optimizer": {"old": "state"}}
        ),
        path,
    )
    fresh = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(fresh.parameters(), lr=0.003)
    with Run(tmp_path) as run:
        Checkpointer(run, components={"model": fresh, "optimizer": optimizer}).load(
            path, weights_only=True
        )
        torch.testing.assert_close(fresh.weight, model.weight)
        assert not optimizer.state and optimizer.param_groups[0]["lr"] == 0.003
        assert run.step == 0


def test_nonimportable_evaluator_callback_fails_cleanly(tmp_path):
    with Run(tmp_path) as run:
        evaluator = AsyncEvaluator(run, lambda checkpoint, device: {})
        path = tmp_path / "snapshot"
        path.write_text("test")
        with pytest.raises((AttributeError, TypeError)):
            evaluator.submit(path)
        assert evaluator.process is None
        evaluator.close(wait=False)
