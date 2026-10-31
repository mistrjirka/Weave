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
        assert (Path(saved["model_path"])).is_file()
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
    saved = service.workspaces.read("training-test", saved)
    helper = Path(saved.script_path).parent / "my_helpers.py"
    helper.write_text("VALUE=7\n")
    result = service.launch(Launch(project=tiny_project(), stage=saved))
    run = service.run_dir(result["id"])
    assert (run / "my_helpers.py").read_text() == "VALUE=7\n"
    helper.write_text("VALUE=10\n")
    Path(saved.script_path).write_text("print('later edit')\n")
    assert (run / "my_helpers.py").read_text() == "VALUE=7\n"
    assert (run / "train.py").read_text() == "print('original')\n"


def test_project_root_is_configurable_and_moves_legacy_files(tmp_path, monkeypatch):
    from backend.paths import projects_directory

    chosen = tmp_path / "chosen-projects"
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(chosen))
    legacy = tmp_path / "data/projects/project-test/stages/stage-one"
    legacy.mkdir(parents=True)
    original = "# My custom training logic must survive.\nfrom weave_training import Run\nprint('mine')\n"
    (legacy / "train.py").write_text(original)
    (legacy / "config.json").write_text(stage().config.model_dump_json())
    (legacy / "runtime.py").write_text("VALUE=7\n")
    (legacy / ".weave-generated.json").write_text(json.dumps({"files": ["runtime.py"]}))
    store = WorkspaceStore(tmp_path / "data/training")
    saved = store.save_stage(
        "project-test", stage().model_copy(update={"script": original})
    )
    assert projects_directory() == chosen
    assert Path(saved.script_path) == chosen / "project-test/stages/stage-one/train.py"
    assert "# My custom training logic must survive." in saved.script
    assert "from weave_support.weave_training import Run" in saved.script
    assert saved.previous_script == original
    assert not legacy.exists()
    directory = Path(saved.script_path).parent
    assert not (directory / "runtime.py").exists()
    assert (directory / "weave_support/runtime.py").read_text() == "VALUE=7\n"
    assert (
        directory / "weave_support/migrations/train.original.py"
    ).read_text() == original


def test_only_exact_shipped_long_script_is_shortened(tmp_path, monkeypatch):
    import hashlib
    from backend.training import layout

    old = "from weave_training import Run\n# Original long template\n"
    monkeypatch.setattr(
        layout, "LEGACY_RECIPE_HASHES", {hashlib.sha256(old.encode()).hexdigest()}
    )
    store = WorkspaceStore(tmp_path / "training")
    recipe = stage().model_copy(update={"config": TrainingConfig(), "script": old})
    saved = store.save_stage("shortened", recipe)
    assert saved.script == (Path(layout.__file__).parent / "template.py").read_text()
    assert len(saved.script.splitlines()) < 30
    assert saved.previous_script == old
    edited = store.save_stage(
        "edited", recipe.model_copy(update={"script": old + "print('my edit')\n"})
    )
    assert "print('my edit')" in edited.script
    assert "Original long template" in edited.script
    assert edited.previous_script == old + "print('my edit')\n"


def test_support_package_and_user_files_are_separated(tmp_path):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    saved = service.workspaces.save_stage("training-test", stage())
    service.workspaces.prepare_model(tiny_project(), saved, service.tokenizer_root)
    saved = service.workspaces.read("training-test", saved)
    directory = Path(saved.script_path).parent
    assert {p.name for p in directory.iterdir() if not p.name.startswith(".")} == {
        "train.py",
        "config.json",
        "weave_support",
    }
    assert (directory / "weave_support/model.py").is_file()
    assert (directory / "weave_support/requirements.txt").is_file()
    assert "from ." in (directory / "weave_support/model.py").read_text()


def test_install_config_and_environment_override(tmp_path, monkeypatch):
    from backend import paths

    config = tmp_path / ".weave-config.json"
    monkeypatch.setattr(paths, "INSTALL_SETTINGS", config)
    monkeypatch.delenv("WEAVE_PROJECTS_DIR", raising=False)
    assert (
        paths.projects_directory()
        == Path("~/shared/weave-projects").expanduser().absolute()
    )
    config.write_text(json.dumps({"projects_directory": str(tmp_path / "installed")}))
    assert paths.projects_directory() == tmp_path / "installed"
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(tmp_path / "override"))
    assert paths.projects_directory() == tmp_path / "override"


def test_generated_custom_block_uses_relative_package_imports(tmp_path):
    import os
    import subprocess

    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    project = tiny_project()
    project["graph"]["nodes"][2]["data"] = {
        "kind": "custom",
        "label": "Custom transform",
        "params": {
            "input_shape": ["B", "T", "$width"],
            "output_shape": ["B", "T", "$width"],
            "parameter_count": 0,
            "source": "class CustomBlock(nn.Module):\n    def forward(self, x):\n        return x * 2\n",
        },
    }
    saved = service.workspaces.save_stage("training-test", stage())
    service.workspaces.prepare_model(project, saved, service.tokenizer_root)
    saved = service.workspaces.read("training-test", saved)
    directory = Path(saved.script_path).parent
    assert "from .custom_" in (directory / "weave_support/model.py").read_text()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from weave_support.model import Model; m=Model(); y=m(torch.tensor([[1,2,3]])); assert y.shape==(1,3,8); y.sum().backward(); assert any(p.grad is not None for p in m.parameters())",
        ],
        cwd=directory,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_migration_collision_preserves_both_locations(tmp_path, monkeypatch):
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(tmp_path / "chosen"))
    legacy = tmp_path / "data/projects/project-test"
    chosen = tmp_path / "chosen/project-test"
    legacy.mkdir(parents=True)
    chosen.mkdir(parents=True)
    (legacy / "note.txt").write_text("old")
    (chosen / "note.txt").write_text("new")
    with pytest.raises(WorkspaceConflict, match="Files were not overwritten"):
        WorkspaceStore(tmp_path / "data/training").project_directory("project-test")
    assert (legacy / "note.txt").read_text() == "old"
    assert (chosen / "note.txt").read_text() == "new"


def test_changing_configured_root_preserves_latest_ssh_script(tmp_path, monkeypatch):
    first = tmp_path / "first-projects"
    second = tmp_path / "second-projects"
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(first))
    store = WorkspaceStore(tmp_path / "training")
    saved = store.save_stage("project-test", stage())
    Path(saved.script_path).write_text("print('newer SSH edit')\n")
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(second))
    moved = store.load_plan("project-test").stages[0]
    assert moved.script == "print('newer SSH edit')\n"
    assert Path(moved.script_path).is_relative_to(second)
    assert not (first / "project-test").exists()
