"""The global indicator returns only active jobs across all projects."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.training.api import router
from backend.training.service import TrainingService
from backend.training.weave_training import atomic_json


def test_active_runs_are_cross_project_lightweight_and_exclude_terminal_states(
    tmp_path,
):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    states = [
        "running",
        "preparing",
        "starting",
        "evaluating",
        "stopped",
        "failed",
        "completed",
        "killed",
        "prepared",
    ]
    for index, state in enumerate(states):
        directory = service.root / "runs" / f"run-global-{index}"
        directory.mkdir()
        (directory / "weave_support").mkdir()
        atomic_json(
            directory / "run.json",
            {
                "id": directory.name,
                "project_id": f"project-{index}",
                "stage_id": "pretraining",
                "name": f"Stage {index}",
                "created": f"2026-10-31T20:0{index}:00+00:00",
                "execution": "local",
            },
        )
        atomic_json(directory / "status.json", {"state": state})
        atomic_json(
            directory / "weave_support/project.json", {"name": f"Project {index}"}
        )
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/training/active-runs")
        assert response.status_code == 200
        rows = response.json()
        assert [row["state"] for row in rows] == [
            "evaluating",
            "starting",
            "preparing",
            "running",
        ]
        assert len({row["project_id"] for row in rows}) == 4
        assert rows[0]["project_name"] == "Project 3"
        assert all("events" not in row and "hardware" not in row for row in rows)
        atomic_json(
            service.root / "runs/run-global-0/status.json", {"state": "stopped"}
        )
        assert len(client.get("/api/training/active-runs").json()) == 3
