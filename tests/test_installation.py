"""Installation keeps user environments/configuration and launches without exports."""

import json
import os
from pathlib import Path
import sys

import pytest

from backend import installation as setup


@pytest.fixture
def installation(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup, "SETTINGS", tmp_path / ".weave-config.json")
    for variable in setup.ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def test_configuration_preserves_existing_settings_and_interpreter_symlink(
    installation,
):
    python = installation / "my-python"
    python.symlink_to(sys.executable)
    setup.SETTINGS.write_text(
        json.dumps({"projects_directory": "/keep/me", "custom": 42})
    )
    setup.main(
        [
            "configure",
            "--enable-training",
            "--training-python",
            str(python),
            "--port",
            "9191",
        ]
    )
    settings = setup.read_settings()
    assert settings["projects_directory"] == "/keep/me"
    assert settings["custom"] == 42
    assert settings["training_python"] == str(python)
    setup.main(["configure", "--disable-training"])
    assert setup.read_settings()["training_enabled"] is False
    assert setup.read_settings()["port"] == 9191


def test_start_applies_saved_configuration_with_environment_precedence(
    installation, monkeypatch
):
    (installation / "dist").mkdir()
    (installation / "dist/index.html").write_text("built")
    setup.save_settings(
        {
            "host": "127.0.0.1",
            "port": 9191,
            "training_enabled": True,
            "training_python": sys.executable,
            "data_directory": str(installation / "data"),
        }
    )
    monkeypatch.setenv("WEAVE_PORT", "9292")
    calls = []
    import uvicorn

    monkeypatch.setattr(
        uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.chdir(installation)
    setup.main(["run"])
    assert calls == [(("backend.app:app",), {"host": "127.0.0.1", "port": 9292})]
    assert os.environ["WEAVE_TRAINING_ENABLED"] == "1"
    assert os.environ["WEAVE_DATA"] == str(installation / "data")


def test_fresh_install_uses_uv_lock_and_separate_cpu_environment(
    installation, monkeypatch
):
    calls = []
    monkeypatch.setattr(setup, "check_tools", lambda: None)

    def run(command, **kwargs):
        command = list(map(str, command))
        calls.append(command)
        if command[:2] == ["uv", "venv"]:
            interpreter = Path(command[-1]) / "bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)

    monkeypatch.setattr(setup, "run", run)
    setup.main(["install", "--projects-dir", str(installation / "projects")])
    assert ["uv", "sync", "--locked", "--no-dev", "--python", "3.12"] in calls
    assert ["npm", "ci"] in calls and ["npm", "run", "build"] in calls
    assert any("https://download.pytorch.org/whl/cpu" in command for command in calls)
    settings = setup.read_settings()
    assert settings["training_enabled"] is True
    assert settings["training_python"] == str(
        installation / ".venv-training/bin/python"
    )


def test_reinstall_preserves_explicit_disabled_state(installation, monkeypatch):
    setup.save_settings({"training_enabled": False, "training_python": sys.executable})
    calls = []
    monkeypatch.setattr(setup, "check_tools", lambda: None)
    monkeypatch.setattr(setup, "run", lambda command: calls.append(command))
    setup.main(["install", "--projects-dir", str(installation / "projects")])
    assert not any(
        "torch" in command or "training-requirements.txt" in command
        for command in calls
    )
    assert setup.read_settings()["training_enabled"] is False


def test_existing_gpu_environment_does_not_install_torch(installation, monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "check_tools", lambda: None)
    monkeypatch.setattr(
        setup, "run", lambda command: calls.append(list(map(str, command)))
    )
    setup.main(
        [
            "install",
            "--training-python",
            sys.executable,
            "--projects-dir",
            str(installation / "projects"),
        ]
    )
    assert not any("torch" in command for command in calls)
    assert any("training-requirements.txt" in command for command in calls)
    assert setup.read_settings()["training_python"] == sys.executable


def test_invalid_port_and_conflicting_training_options_reject_before_install(
    installation, monkeypatch
):
    with pytest.raises(SystemExit):
        setup.main(["configure", "--port", "99999"])
    monkeypatch.setattr(setup, "check_tools", lambda: None)
    calls = []
    monkeypatch.setattr(setup, "run", lambda command: calls.append(command))
    with pytest.raises(SystemExit):
        setup.main(
            ["install", "--training", "none", "--training-python", sys.executable]
        )
    assert calls == []
