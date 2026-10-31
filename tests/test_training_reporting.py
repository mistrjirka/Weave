"""Independent reporting must preserve training and keep remote files honest."""

import io
import json
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.training.api import router
from backend.training.service import TrainingConfig, events
from backend.training.weave_training import Run, _Reporter
from test_training import tiny_project


def prepare(client):
    response = client.post(
        "/api/training/runs",
        json={
            "project": tiny_project(),
            "stage": {
                "id": "external",
                "name": "Independent",
                "script": 'print("custom")\n',
                "config": TrainingConfig(
                    recipe="custom", execution="monitor"
                ).model_dump(),
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_remote_reporting_key_dedup_commands_and_no_fake_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("WEAVE_TRAINING_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        run_id = prepare(client)
        directory = tmp_path / "training/runs" / run_id
        connection = json.loads((directory / "reporting.json").read_text())
        assert (
            connection["url"] == f"http://localhost:8173/api/training/reports/{run_id}"
        )
        assert (directory / ".report-token").stat().st_mode & 0o777 == 0o600
        connection = client.post(
            f"/api/training/runs/{run_id}/reporting",
            json={"base_url": "http://weave.example:8173"},
        ).json()
        assert connection["url"].startswith("http://weave.example:8173/")
        with Run(tmp_path / "remote") as report:
            report.log({"train/loss": 0.4, "train/learning_rate": 0.01}, step=3)
            report.sample(prompt="hello", output="there")
            checkpoint = report.directory / "checkpoints/final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"remote weights")
            report.checkpoint(checkpoint)
        records = [
            json.loads(line)
            for line in (tmp_path / "remote/events.jsonl").read_text().splitlines()
        ]
        endpoint = f"/api/training/reports/{run_id}"
        assert client.post(endpoint, json={"events": records}).status_code == 401
        headers = {"x-weave-report-token": connection["token"]}
        first = client.post(endpoint, json={"events": records}, headers=headers)
        assert first.status_code == 200, first.text
        assert first.json()["accepted"] == len(records)
        assert (
            client.post(endpoint, json={"events": records}, headers=headers).json()[
                "accepted"
            ]
            == 0
        )
        detail = client.get(f"/api/training/runs/{run_id}").json()
        assert detail["status"]["state"] == "completed"
        assert detail["reporting"]["remote"]
        checkpoint_event = next(
            e for e in detail["events"] if e["kind"] == "checkpoint"
        )
        assert checkpoint_event["payload"]["remote"]
        snapshot = client.post(
            f"/api/training/runs/{run_id}/commands", json={"kind": "snapshot"}
        ).json()
        data = client.get(
            f"/api/training/runs/{run_id}/artifact",
            params={"path": snapshot["artifact"]},
        ).content
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert not json.loads(archive.read("manifest.json"))["resumable_weights"]
            assert ".report-token" not in archive.namelist()
            assert "reporting.json" not in archive.namelist()
        command = client.post(
            f"/api/training/runs/{run_id}/commands", json={"kind": "stop"}
        ).json()
        result = client.post(endpoint, json={"events": []}, headers=headers).json()
        assert command["id"] in [item["id"] for item in result["commands"]]
        invalid = {**records[0], "time": "NaN"}
        assert (
            client.post(
                endpoint, json={"events": [invalid]}, headers=headers
            ).status_code
            == 422
        )


def test_reporter_outage_keeps_local_records_and_retries_same_batch(tmp_path):
    # Exercise the actual urllib request body and durable cursor across restarts.
    with Run(tmp_path / "local") as run:
        run.log({"train/loss": 0.7}, step=1)
    config = {
        "url": "http://localhost:8173/api/training/reports/run-test",
        "token": "scoped",
    }
    reporter = _Reporter(Run(tmp_path / "local"), config)
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        with pytest.raises(OSError):
            reporter.send()
    assert reporter.cursor == 0
    assert not reporter.cursor_path.exists()
    seen = []

    def receive(request, timeout):
        seen.extend(json.loads(request.data)["events"])
        assert request.get_header("X-weave-report-token") == "scoped"
        return io.BytesIO(
            json.dumps({"commands": [{"id": "abc123", "kind": "stop"}]}).encode()
        )

    with patch("urllib.request.urlopen", side_effect=receive):
        assert not reporter.send()
    assert len(seen) == 3
    assert (tmp_path / "local/commands/abc123.json").is_file()
    restored = _Reporter(Run(tmp_path / "local"), config)
    assert restored.cursor == (tmp_path / "local/events.jsonl").stat().st_size
    with patch("urllib.request.urlopen", side_effect=receive):
        restored.send()
    assert len(seen) == 3


def test_optimizer_hook_tracks_parameter_groups_and_removes_itself(tmp_path):
    import torch

    a = torch.nn.Parameter(torch.tensor(2.0))
    b = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD(
        [{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}]
    )
    with Run(tmp_path / "hook") as run:
        run.watch_optimizer(optimizer)
        loss = a.square() + b.square()
        loss.backward()
        optimizer.step()
        run.log({"train/loss": loss.item()}, step=1)
    metrics = [
        item["payload"]
        for item in events(tmp_path / "hook")
        if item["kind"] == "metrics"
    ]
    assert metrics[0] == {"lr/group_0": 0.1, "lr/group_1": 0.01}
    assert metrics[1] == {"train/loss": 13.0}
    assert not optimizer._optimizer_step_post_hooks


def test_real_http_transport_delivers_finished_run_and_remote_gpu(
    tmp_path, monkeypatch
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from backend.training.reporting import ReportStore, ReportBatch
    from backend.training.service import TrainingService, Launch, Stage

    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    info = service.launch(
        Launch(
            project=tiny_project(),
            stage=Stage(
                id="remote-http",
                name="HTTP integration",
                script='print("external")\n',
                config=TrainingConfig(recipe="custom", execution="monitor"),
            ),
        )
    )
    directory = service.run_dir(info["id"])
    reports = ReportStore()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content = self.rfile.read(int(self.headers["Content-Length"]))
            result = reports.ingest(
                directory,
                self.headers["X-Weave-Report-Token"],
                ReportBatch.model_validate_json(content),
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = reports.connection(directory, f"http://127.0.0.1:{server.server_port}")
    config_path = tmp_path / "connection.json"
    config_path.write_text(json.dumps(connection))
    monkeypatch.setenv("WEAVE_REPORT_CONFIG", str(config_path))
    monkeypatch.setattr(_Reporter, "hardware", lambda self: None)
    try:
        with Run(tmp_path / "independent") as report:
            report.log({"train/loss": 0.25}, step=2)
            report.sample(prompt="Hi", output="there")
            report.emit("hardware", {"time": time.time(), "gpus": []})
        received = events(directory)
        assert sum(item["kind"] == "metrics" for item in received) == 1
        assert any(item["kind"] == "sample" for item in received)
        assert any(item["kind"] == "hardware" for item in received)
        assert service.status(directory)["state"] == "completed"
        assert json.loads((tmp_path / "independent/reporting-status.json").read_text())[
            "connected"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_simple_reporting_preserves_training_and_tracks_real_updates(tmp_path):
    import torch

    def train(reporting):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

        # Simulate a resumed run and gradient accumulation: two backward calls
        # per optimizer update, with the explicitly chosen aggregate objective.
        def steps(run=None):
            for update in range(3):
                optimizer.zero_grad()
                losses = []
                for target in (0.0, 1.0):
                    loss = (parameter - target).square()
                    (loss / 2).backward()
                    losses.append(loss.detach())
                optimizer.step()
                scheduler.step()
                if run:
                    run.log_loss(sum(losses) / 2, tokens=(update + 1) * 8)

        if reporting:
            with Run(
                tmp_path / "simple", optimizer=optimizer, total_steps=8, start_step=5
            ) as run:
                steps(run)
            assert not optimizer._optimizer_step_post_hooks
        else:
            steps()
        return parameter.detach()

    assert torch.equal(train(False), train(True))
    records = events(tmp_path / "simple")
    losses = [
        event
        for event in records
        if event["kind"] == "metrics" and "train/loss" in event["payload"]
    ]
    assert [event["step"] for event in losses] == [6, 7, 8]
    assert [event["tokens"] for event in losses] == [8, 16, 24]
    assert losses[0]["payload"]["train/loss"] == pytest.approx(2.5)
    rates = [
        event["payload"]["lr/group_0"]
        for event in records
        if event["kind"] == "metrics" and "lr/group_0" in event["payload"]
    ]
    assert rates == pytest.approx([0.1, 0.05, 0.025])
    progress = [event["payload"] for event in records if event["kind"] == "progress"]
    assert [item["completed"] for item in progress] == [6, 7, 8]
    assert all(item["total"] == 8 for item in progress)
    assert not any(event["kind"] == "warning" for event in records)


def test_simple_reporting_warns_when_loss_hook_is_forgotten(tmp_path):
    import torch

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with Run(tmp_path / "missing", optimizer=optimizer, total_steps=4) as run:
        # Attaching the same optimizer twice must not double-count updates.
        run.watch_optimizer(optimizer)
        for _ in range(4):
            optimizer.zero_grad()
            parameter.square().backward()
            optimizer.step()
    records = events(tmp_path / "missing")
    warnings = [event for event in records if event["kind"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["payload"]["code"] == "missing-loss"
    assert "run.log_loss(loss)" in warnings[0]["payload"]["message"]
    assert max(event["step"] for event in records) == 4
    assert records[-1]["payload"]["success"]
    assert not optimizer._optimizer_step_post_hooks


def test_simple_reporting_interval_first_final_and_early_exit(tmp_path):
    import torch

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with Run(
        tmp_path / "interval", optimizer=optimizer, total_steps=3, log_every=10
    ) as run:
        for _ in range(3):
            optimizer.zero_grad()
            loss = parameter.square()
            loss.backward()
            optimizer.step()
            run.log_loss(loss)
    assert [
        event["step"]
        for event in events(tmp_path / "interval")
        if event["kind"] == "metrics" and "train/loss" in event["payload"]
    ] == [1, 3]
    with Run(tmp_path / "short", optimizer=optimizer, total_steps=100, log_every=10):
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()
    assert any(
        event["kind"] == "warning" and event["payload"]["code"] == "missing-loss"
        for event in events(tmp_path / "short")
    )
    assert not optimizer._optimizer_step_post_hooks


def test_clean_training_package_runs_offline_from_another_directory(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import sys
    from backend.training.service import Stage

    monkeypatch.delenv("WEAVE_TRAINING_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(router(tmp_path))
    data = tmp_path / "training-data.jsonl"
    data.write_text(json.dumps({"input_ids": [1, 2, 3, 4]}) + "\n")
    template = Path(__file__).parents[1] / "backend/training/template.py"
    stage = Stage(
        id="offline-package",
        name="Offline package",
        script=template.read_text(),
        config=TrainingConfig(
            execution="monitor",
            dataset=str(data),
            max_steps=1,
            batch_size=1,
            checkpoint_every=1,
            cpu_threads=1,
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/training/runs",
            json={"project": tiny_project(), "stage": stage.model_dump()},
        )
        assert response.status_code == 200, response.text
        run_id = response.json()["id"]
        client.post(
            f"/api/training/runs/{run_id}/reporting",
            json={"base_url": "http://127.0.0.1:9"},
        )
        package = client.get(f"/api/training/runs/{run_id}/package")
        assert package.status_code == 200
    extracted = tmp_path / "downloaded-project"
    extracted.mkdir()
    with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
        assert set(name for name in archive.namelist() if "/" not in name) == {
            "train.py",
            "config.json",
            "reporting.json",
            "README.md",
        }
        assert "weave_support/model.py" in archive.namelist()
        assert "weave_support/text_training.py" in archive.namelist()
        assert "weave_support/requirements.txt" in archive.namelist()
        assert len(archive.read("train.py").splitlines()) < 30
        archive.extractall(extracted)
    (extracted / "data.jsonl").write_bytes(data.read_bytes())
    config = json.loads((extracted / "config.json").read_text())
    config["dataset"] = "data.jsonl"
    (extracted / "config.json").write_text(json.dumps(config))
    environment = dict(os.environ)
    environment.pop("WEAVE_RUN_DIR", None)
    environment.pop("WEAVE_REPORT_CONFIG", None)
    result = subprocess.run(
        [sys.executable, str(extracted / "train.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = list((extracted / "runs").glob("run-*"))
    assert len(outputs) == 1
    records = events(outputs[0])
    assert not (extracted / "events.jsonl").exists()
    assert not (extracted / "checkpoints").exists()
    assert any(
        event["kind"] == "metrics" and "train/loss" in event["payload"]
        for event in records
    )
    assert any(event["kind"] == "checkpoint" for event in records)
    assert records[-1]["kind"] == "finished"
    assert (
        json.loads((outputs[0] / "reporting-status.json").read_text())["connected"]
        is False
    )

    # All torchrun workers must agree on one generated output directory.
    config["strategy"] = "ddp"
    (extracted / "config.json").write_text(json.dumps(config))
    distributed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(extracted / "train.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert distributed.returncode == 0, distributed.stdout + distributed.stderr
    outputs = list((extracted / "runs").glob("run-*"))
    assert len(outputs) == 2  # One single-device run and one shared DDP run.
    parallel = next(
        path for path in outputs if (path / "consumed-rank-1.jsonl").exists()
    )
    assert (parallel / "consumed-rank-0.jsonl").is_file()
    assert any(event["kind"] == "checkpoint" for event in events(parallel))


def test_slow_http_server_does_not_block_optimizer_updates(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import torch

    request_started = threading.Event()
    release_response = threading.Event()
    updates_finished = threading.Event()
    failures = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            request_started.set()
            release_response.wait(10)
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(b'{"commands": []}')
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = tmp_path / "reporting.json"
    connection.write_text(
        json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "token": "test"})
    )
    monkeypatch.setenv("WEAVE_REPORT_CONFIG", str(connection))
    monkeypatch.setattr(_Reporter, "hardware", lambda self: None)

    def train():
        try:
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.SGD([parameter], lr=0.01)
            with Run(tmp_path / "outbox", optimizer=optimizer) as run:
                assert request_started.wait(10)
                for _ in range(30):
                    optimizer.zero_grad()
                    loss = parameter.square()
                    loss.backward()
                    optimizer.step()
                    run.log_loss(loss)
                assert parameter.item() < 1.0
                updates_finished.set()
        except BaseException as error:
            failures.append(error)
            updates_finished.set()

    worker = threading.Thread(target=train)
    worker.start()
    try:
        assert request_started.wait(15), "Reporter did not start its HTTP request"
        assert updates_finished.wait(1.5), "Optimizer loop waited for the HTTP response"
        assert not release_response.is_set()
        assert not failures
    finally:
        release_response.set()
        worker.join(10)
        server.shutdown()
        server.server_close()
    assert not worker.is_alive()
    assert len(events(tmp_path / "outbox", kind="metrics")) >= 30


def test_remote_evaluation_cannot_finish_training(tmp_path, monkeypatch):
    monkeypatch.delenv("WEAVE_TRAINING_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        run_id = prepare(client)
        directory = tmp_path / "training/runs" / run_id
        connection = json.loads((directory / "reporting.json").read_text())
        endpoint = f"/api/training/reports/{run_id}"
        headers = {"x-weave-report-token": connection["token"]}
        with Run(tmp_path / "train") as training:
            records = [
                json.loads(line)
                for line in (training.directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            assert (
                client.post(
                    endpoint, json={"events": records}, headers=headers
                ).status_code
                == 200
            )
            with Run(tmp_path / "eval", channel="evaluation") as validation:
                validation.log({"eval/loss": 0.5}, step=1)
            records = [
                json.loads(line)
                for line in (validation.directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            assert (
                client.post(
                    endpoint, json={"events": records}, headers=headers
                ).status_code
                == 200
            )
            detail = client.get(f"/api/training/runs/{run_id}").json()
            assert detail["status"]["state"] == "running"
            assert any(
                event["channel"] == "evaluation" and event["kind"] == "metrics"
                for event in detail["events"]
            )


def test_default_reporter_preserves_filesystem_system_channel(tmp_path):
    with Run(tmp_path / "system") as reporter:
        reporter.emit("hardware", {"gpus": []})
    records = events(tmp_path)
    assert records and all(event["channel"] == "system" for event in records)
