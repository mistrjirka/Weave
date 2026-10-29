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
