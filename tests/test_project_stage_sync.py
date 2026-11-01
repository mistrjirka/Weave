"""Saving a graph refreshes stage models without rewriting recipes or run snapshots."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import backend.app as server
from backend.training.service import TrainingService, Stage, TrainingConfig, Launch
from backend.training.layout import PACKAGE
from test_training import tiny_project


def tree_hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_project_save_syncs_all_stages_and_preserves_scripts_and_frozen_runs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "DATA", tmp_path)
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    project = tiny_project()
    info = {
        "id": "a" * 20,
        "size": 8,
        "capacity": 8,
        "base_size": 8,
        "special_tokens": {},
    }
    project["graph"]["nodes"][0]["data"]["tokenizer"] = info
    for token_id in ["a" * 20, "b" * 20]:
        directory = tmp_path / "tokenizers" / token_id
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text(json.dumps({"version": token_id}))
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": token_id + " template"})
        )
    client = TestClient(server.app)
    endpoint = "/api/projects/" + project["id"]
    assert client.put(endpoint, json=project).status_code == 200
    stages = [
        service.workspaces.save_stage(
            project["id"],
            Stage(
                id=f"stage-{i}",
                name=f"Stage {i}",
                script="print('original')\n",
                config=TrainingConfig(recipe="custom", execution="monitor"),
            ),
        )
        for i in range(2)
    ]
    response = client.put(endpoint, json=project)
    assert response.json()["stage_sync"] == {"updated": 2, "error": None}
    run = service.launch(Launch(project=project, stage=stages[0]))
    frozen = service.run_dir(run["id"])
    before_run = tree_hashes(frozen)
    preserved = {}
    for stage in stages:
        directory = service.workspaces.directory(project["id"], stage.id)
        (directory / "train.py").write_text(
            "# Edited in SSH\nprint('my custom recipe')\n"
        )
        (directory / "my_loader.py").write_text("USER_HELPER = True\n")
        preserved[stage.id] = {
            name: (directory / name).read_bytes()
            for name in ("train.py", "config.json", "my_loader.py")
        }
    updated = copy.deepcopy(project)
    updated["graph"]["nodes"][0]["data"]["tokenizer"]["id"] = "b" * 20
    updated["config"]["width"] = 16
    updated["graph"]["nodes"][1]["data"]["params"]["width"] = 16
    response = client.put(endpoint, json=updated)
    assert response.status_code == 200, response.text
    assert response.json()["stage_sync"] == {"updated": 2, "error": None}
    for stage in stages:
        directory = service.workspaces.directory(project["id"], stage.id)
        package = directory / PACKAGE
        assert json.loads((package / "project.json").read_text()) == updated
        assert (
            json.loads((package / "tokenizers.json").read_text())["ids"]["id"]
            == "b" * 20
        )
        assert "embedding_dim=16" in (package / "model.py").read_text()
        assert not (package / "tokenizers" / ("a" * 20)).exists()
        assert (
            json.loads(
                (
                    package / "tokenizers" / ("b" * 20) / "tokenizer_config.json"
                ).read_text()
            )["chat_template"]
            == "b" * 20 + " template"
        )
        assert {
            name: (directory / name).read_bytes() for name in preserved[stage.id]
        } == preserved[stage.id]
        assert not (directory / ".weave-model-error.txt").exists()
    assert tree_hashes(frozen) == before_run


def test_failed_sync_is_visible_and_blocks_stale_standalone_import_until_fixed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "DATA", tmp_path)
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    project = tiny_project()
    stage = service.workspaces.save_stage(
        project["id"],
        Stage(
            id="stage-one",
            name="Stage",
            script="print('keep')\n",
            config=TrainingConfig(recipe="custom", execution="monitor"),
        ),
    )
    client = TestClient(server.app)
    endpoint = "/api/projects/" + project["id"]
    assert client.put(endpoint, json=project).json()["stage_sync"]["error"] is None
    directory = service.workspaces.directory(project["id"], stage.id)
    invalid = copy.deepcopy(project)
    invalid["graph"]["edges"] = []
    result = client.put(endpoint, json=invalid)
    assert result.status_code == 200  # Invalid drafts can still be saved.
    assert result.json()["stage_sync"]["error"]
    assert (
        "could not be synchronized"
        in service.workspaces.get_stage(project["id"], stage.id).workspace_warning
    )
    failure = subprocess.run(
        [sys.executable, "-c", "import weave_support"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    assert failure.returncode != 0
    assert "could not be synchronized" in failure.stderr
    assert (directory / "train.py").read_text() == "print('keep')\n"
    assert client.put(endpoint, json=project).json()["stage_sync"]["error"] is None
    assert not (directory / ".weave-model-error.txt").exists()
    assert (
        service.workspaces.get_stage(project["id"], stage.id).workspace_warning is None
    )
    success = subprocess.run(
        [sys.executable, "-c", "import weave_support"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    assert success.returncode == 0, success.stderr


def test_launch_preparation_repairs_older_stage_and_rejects_missing_tokenizer(tmp_path):
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    project = tiny_project()
    stage = service.workspaces.save_stage(
        project["id"],
        Stage(
            id="stage-one",
            name="Stage",
            script="print('keep')\n",
            config=TrainingConfig(recipe="custom", execution="monitor"),
        ),
    )
    service.workspaces.prepare_model(project, stage, service.tokenizer_root)
    updated = copy.deepcopy(project)
    updated["config"]["width"] = 16
    updated["graph"]["nodes"][1]["data"]["params"]["width"] = 16
    service.workspaces.ensure_model(updated, stage, service.tokenizer_root)
    directory = service.workspaces.directory(project["id"], stage.id)
    assert json.loads((directory / PACKAGE / "project.json").read_text()) == updated
    assert (directory / "train.py").read_text() == "print('keep')\n"
    updated["graph"]["nodes"][0]["data"]["tokenizer"] = {
        "id": "c" * 20,
        "size": 8,
        "capacity": 8,
        "base_size": 8,
        "special_tokens": {},
    }
    import pytest

    with pytest.raises(ValueError, match="missing from this server"):
        service.workspaces.ensure_model(updated, stage, service.tokenizer_root)
    assert (directory / ".weave-model-error.txt").exists()
