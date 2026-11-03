"""The stage UI previews real settings without executing editable Python."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.training.api import router
from backend.training.service import Stage, TrainingConfig
from backend.training.stage_settings import preview_stage_settings, trainer_defaults


def stage(source, **config):
    return Stage(
        id="preview", name="Preview", script=source, config=TrainingConfig(**config)
    )


def script(arguments=""):
    return f"from weave_support.trainer import TrainingConfig\nconfig = TrainingConfig({arguments})\n"


def test_preview_uses_script_values_and_auto_strategy_without_changing_stage():
    item = stage(
        script(
            'devices=[0, 1], precision="fp16", async_evaluation=True, evaluation_device=2, compile_mode="reduce-overhead", fused_optimizer=True'
        )
    )
    before = item.model_dump()
    result = preview_stage_settings(item)
    assert result["error"] is None and result["unresolved"] == []
    assert {
        key: result["config"][key]
        for key in (
            "devices",
            "strategy",
            "precision",
            "evaluation_mode",
            "evaluation_device",
            "compile_mode",
            "fused_optimizer",
        )
    } == {
        "devices": [0, 1],
        "strategy": "ddp",
        "precision": "fp16",
        "evaluation_mode": "async",
        "evaluation_device": 2,
        "compile_mode": "reduce-overhead",
        "fused_optimizer": True,
    }
    assert item.model_dump() == before


def test_launch_choices_take_precedence_and_unedited_groups_keep_their_values():
    result = preview_stage_settings(
        stage(
            script(
                'devices=[1], precision="fp16", compile_mode="default", async_evaluation=True, evaluation_device=1'
            ),
            compute_mode="override",
            devices=[0, 1],
            strategy="ddp",
            precision="bf16",
            evaluation_mode="inline",
        )
    )
    assert result["config"]["devices"] == [0, 1]
    assert result["config"]["precision"] == "bf16"
    assert result["config"]["compile_mode"] == "default"
    assert result["config"]["evaluation_mode"] == "inline"


def test_static_constants_and_import_aliases_are_supported():
    result = preview_stage_settings(
        stage(
            'from weave_support.trainer import TrainingConfig as Config\nDEVICES = [1]\nPRECISION = "fp16"\nconfig = Config(devices=DEVICES, precision=PRECISION)'
        )
    )
    assert result["config"]["devices"] == [1]
    assert result["config"]["precision"] == "fp16"
    assert result["config"]["strategy"] == "single"
    assert result["config"]["evaluation_mode"] == "inline"
    assert not result["unresolved"]


def test_dynamic_python_is_not_executed_or_presented_as_known(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"import pathlib\npathlib.Path({str(marker)!r}).touch()\n" + script(
        'devices=get_devices(), precision="fp16"'
    )
    result = preview_stage_settings(stage(source))
    assert not marker.exists()
    assert set(result["unresolved"]) == {"devices", "strategy"}
    assert result["config"]["precision"] == "fp16"
    explicit = preview_stage_settings(
        stage(source, compute_mode="override", devices=[1])
    )
    assert explicit["unresolved"] == []


def test_async_validation_without_device_uses_the_actual_training_device():
    result = preview_stage_settings(stage(script("devices=[1], async_evaluation=True")))
    assert result["config"]["evaluation_device"] == 1
    result = preview_stage_settings(stage(script("devices=[], async_evaluation=True")))
    assert result["config"]["evaluation_device"] == "cpu"
    result = preview_stage_settings(
        stage(
            script('devices=[1], strategy="train_eval", evaluation_device=0'),
            evaluation_mode="inline",
        )
    )
    assert result["config"]["strategy"] == "single"


@pytest.mark.parametrize(
    "arguments",
    ["devices=None", 'precision="int8"', "devices=[-1]", 'compile_mode="magic"'],
)
def test_invalid_literals_are_reported_without_crashing(arguments):
    assert preview_stage_settings(stage(script(arguments)))["error"]


def test_syntax_errors_and_unrecognised_constructors_have_explicit_feedback():
    assert "Python line" in preview_stage_settings(stage(script() + "broken("))["error"]
    result = preview_stage_settings(
        stage("from weave_support.trainer import Trainer\nconfig = build_config()\n")
    )
    assert "devices" in result["unresolved"]
    assert trainer_defaults()["devices"] == []
    assert trainer_defaults()["precision"] == "fp32"


def test_preview_endpoint_is_authenticated_and_does_not_write_a_stage(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "preview-secret")
    app = FastAPI()
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        body = stage(script('devices=[1], precision="fp16"')).model_dump()
        assert (
            client.post("/api/training/settings/preview", json=body).status_code == 401
        )
        response = client.post(
            "/api/training/settings/preview",
            json=body,
            headers={"x-weave-training-token": "preview-secret"},
        )
        assert response.status_code == 200
        assert response.json()["config"]["devices"] == [1]
        assert not list(tmp_path.rglob("train.py"))


def test_multiple_configs_and_later_mutations_are_not_mistaken_for_final_values():
    multiple = script("devices=[0]") + "other = TrainingConfig(devices=[1])\n"
    assert "devices" in preview_stage_settings(stage(multiple))["unresolved"]
    modified = (
        script('devices=[0], compile_mode="off"')
        + 'config.devices.append(1)\nif enabled():\n    config.compile_mode = "default"\n'
    )
    unknown = preview_stage_settings(stage(modified))["unresolved"]
    assert set(unknown) >= {"devices", "strategy", "compile_mode"}
