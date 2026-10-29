"""Browser/SSH training files share content without silently losing edits."""

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.training.api import router
from backend.training.service import (
    Stage,
    Plan,
    TrainingConfig,
    TrainingService,
    Launch,
)
from backend.training.workspaces import WorkspaceStore, WorkspaceConflict
from test_training import tiny_project


def stage():
    return Stage(
        id="stage-one",
        name="Python stage",
        config=TrainingConfig(recipe="custom", execution="monitor"),
        script="print('original')\n",
    )


def test_real_files_external_edits_and_conflict_protection(tmp_path):
    store = WorkspaceStore(tmp_path)
    saved = store.save_stage("project-test", stage())
    script = Path(saved.script_path)
    config = Path(saved.config_path)
    assert script.read_text() == saved.script
    assert config.is_absolute() and config.is_file()
    assert saved.script_revision
    script.write_text("print('edited through SSH')\n")
    loaded = store.load_plan("project-test")
    assert loaded.stages[0].script == "print('edited through SSH')\n"
    stale = saved.model_copy(update={"script": "print('stale browser')\n"})
    with pytest.raises(WorkspaceConflict, match="changed on the server"):
        store.save_stage("project-test", stale)
    assert script.read_text() == "print('edited through SSH')\n"
    current = loaded.stages[0].model_copy(
        update={"script": "print('saved from browser')\n"}
    )
    store.save_stage("project-test", current)
    assert script.read_text() == current.script
    values = json.loads(config.read_text())
    values["learning_rate"] = 0.001
    config.write_text(json.dumps(values))
    assert store.load_plan("project-test").stages[0].config.learning_rate == 0.001
    with pytest.raises(WorkspaceConflict, match="config.json"):
        store.save_stage(
            "project-test",
            current.model_copy(
                update={
                    "config": TrainingConfig(
                        recipe="custom", execution="monitor", learning_rate=0.02
                    )
                }
            ),
        )


def test_legacy_inline_stages_migrate_without_overwriting_ssh_changes(tmp_path):
    store = WorkspaceStore(tmp_path)
    plan = Plan(project_id="legacy", stages=[stage()])
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans/legacy.json").write_text(plan.model_dump_json())
    first = store.load_plan("legacy").stages[0]
    Path(first.script_path).write_text("print('external')\n")
    assert store.load_plan("legacy").stages[0].script == "print('external')\n"


def test_default_training_has_no_token_and_reports_python(tmp_path, monkeypatch):
    monkeypatch.delenv("WEAVE_TRAINING_TOKEN", raising=False)
    monkeypatch.setenv("WEAVE_TRAINING_ENABLED", "1")
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        capabilities = client.get("/api/training/capabilities").json()
        assert capabilities["authenticated"] and not capabilities["auth_required"]
        assert client.get("/api/training/templates").status_code == 200
        assert (
            client.get(
                "/api/training/templates",
                headers={"Origin": "https://elsewhere.example"},
            ).status_code
            == 403
        )
        info = client.get(
            "/api/training/environment", params={"python": sys.executable}
        )
        assert info.status_code == 200, info.text
        assert info.json()["python_version"]
        assert info.json()["torch_version"]
        body = {"project": tiny_project(), "stage": stage().model_dump()}
        response = client.put(
            "/api/training/workspaces/training-test/stages/stage-one", json=body
        )
        assert response.status_code == 200, response.text
        saved = response.json()
        assert Path(saved["script_path"]).is_file()
        assert (Path(saved["script_path"]).parent / "model.py").is_file()
        Path(saved["script_path"]).write_text("print('SSH')\n")
        body["stage"] = saved
        body["stage"]["script"] = "print('stale')\n"
        assert (
            client.put(
                "/api/training/workspaces/training-test/stages/stage-one", json=body
            ).status_code
            == 409
        )
        fresh = client.get(
            "/api/training/workspaces/training-test/stages/stage-one"
        ).json()
        assert fresh["script"] == "print('SSH')\n"


def test_workspace_helpers_freeze_into_run_and_later_edits_do_not_change_it(tmp_path):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    saved = service.workspaces.save_stage("training-test", stage())
    service.workspaces.prepare_model(tiny_project(), saved, service.tokenizer_root)
    helper = Path(saved.script_path).parent / "my_helpers.py"
    helper.write_text("VALUE=7\n")
    result = service.launch(Launch(project=tiny_project(), stage=saved))
    run = service.run_dir(result["id"])
    assert (run / "my_helpers.py").read_text() == "VALUE=7\n"
    helper.write_text("VALUE=10\n")
    Path(saved.script_path).write_text("print('later edit')\n")
    assert (run / "my_helpers.py").read_text() == "VALUE=7\n"
    assert (run / "train.py").read_text() == "print('original')\n"
