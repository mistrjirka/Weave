import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.training.api import router
from backend.training.service import TrainingConfig
from backend.training.weave_training import Run
from test_training import tiny_project


@pytest.fixture
def training_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEAVE_TRAINING_ENABLED", "1")
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "test-training-secret")
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        yield client, tmp_path


def test_training_requires_auth_and_same_origin(training_client):
    client, root = training_client
    assert client.get("/api/training/templates").status_code == 401
    assert (
        client.post("/api/training/session", json={"token": "wrong"}).status_code == 401
    )
    assert (
        client.post(
            "/api/training/session",
            json={"token": "test-training-secret"},
            headers={"Origin": "https://unrelated.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/training/session", json={"token": "test-training-secret"}
        ).status_code
        == 200
    )
    assert client.get("/api/training/templates").status_code == 200
    assert (
        client.get(
            "/api/training/templates", headers={"Origin": "https://unrelated.example"}
        ).status_code
        == 403
    )


def test_stage_plan_preview_monitor_and_snapshot(training_client):
    client, root = training_client
    client.post("/api/training/session", json={"token": "test-training-secret"})
    dataset = root / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]})
    )
    templates = client.get("/api/training/templates").json()
    config = TrainingConfig(dataset=str(dataset), execution="monitor").model_dump()
    stage = {
        "id": "one",
        "name": "Pretrain",
        "config": config,
        "script": templates["scripts"]["causal"],
    }
    plan = {"version": 1, "project_id": "training-test", "stages": [stage]}
    assert client.put("/api/training/plans/training-test", json=plan).status_code == 200
    saved_plan = client.get("/api/training/plans/training-test").json()
    assert saved_plan["project_id"] == plan["project_id"]
    assert saved_plan["stages"][0]["script"] == stage["script"]
    assert saved_plan["stages"][0]["config"] == stage["config"]
    assert Path(saved_plan["stages"][0]["script_path"]).is_file()
    stage = saved_plan["stages"][0]
    preview = client.post(
        "/api/training/preview", json={"project": tiny_project(), "stage": stage}
    )
    assert preview.status_code == 200, preview.text
    assert [
        token["supervised"] for token in preview.json()["records"][0]["tokens"]
    ] == [False, False, True, True]
    response = client.post(
        "/api/training/runs", json={"project": tiny_project(), "stage": stage}
    )
    assert response.status_code == 200, response.text
    run_id = response.json()["id"]
    directory = root / "training/runs" / run_id
    with Run(directory) as report:
        report.log({"train/loss": 1.0}, step=1, tokens=3)
        report.sample(prompt="hello", output="world")
    detail = client.get("/api/training/runs/" + run_id).json()
    assert detail["status"]["state"] == "completed"
    assert any(event["kind"] == "sample" for event in detail["events"])
    snapshot = client.post(
        "/api/training/runs/" + run_id + "/commands", json={"kind": "snapshot"}
    ).json()
    archive = client.get(
        "/api/training/runs/" + run_id + "/artifact",
        params={"path": snapshot["artifact"]},
    )
    assert archive.status_code == 200
    restored = client.post(
        "/api/training/import",
        files={"file": ("snapshot.zip", archive.content, "application/zip")},
    )
    assert restored.status_code == 200, restored.text
    assert (
        client.get(
            "/api/training/runs/" + run_id + "/artifact",
            params={"path": "../../operator.token"},
        ).status_code
        == 422
    )
    assert client.get("/api/training/runs/" + run_id + "/package").status_code == 200


def test_local_supervisor_runs_real_custom_script(training_client):
    client, root = training_client
    client.post("/api/training/session", json={"token": "test-training-secret"})
    source = 'from weave_training import Run\nwith Run() as run:\n    run.log({"custom/metric":3},step=1)\n'
    stage = {
        "id": "custom",
        "name": "Custom",
        "config": TrainingConfig(recipe="custom").model_dump(),
        "script": source,
    }
    response = client.post(
        "/api/training/runs", json={"project": tiny_project(), "stage": stage}
    )
    assert response.status_code == 200, response.text
    run_id = response.json()["id"]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        detail = client.get("/api/training/runs/" + run_id).json()
        if detail["status"]["state"] in ("completed", "failed"):
            break
        time.sleep(0.3)
    assert detail["status"]["state"] == "completed", detail
    assert any(e["kind"] == "metrics" for e in detail["events"])
    assert detail["hardware"]
