"""Opt-in real two-GPU acceptance, including cross-device gradients."""

import json
import os
from pathlib import Path
import subprocess
import time

import pytest
import torch

from backend.training.service import Stage, Launch, events
from test_training import make_run, run_script, tiny_project, ROOT

PYTHON = os.environ.get("WEAVE_CUDA_PYTHON")
pytestmark = pytest.mark.skipif(
    not PYTHON, reason="Set WEAVE_CUDA_PYTHON to a verified two-GPU PyTorch environment"
)


def state(directory):
    checkpoints = events(directory, kind="checkpoint")
    return torch.load(
        directory / checkpoints[-1]["payload"]["path"],
        map_location="cpu",
        weights_only=True,
    )


def test_split_model_and_ddp_match_single_device(tmp_path):
    (tmp_path / "single").mkdir()
    (tmp_path / "split").mkdir()
    (tmp_path / "ddp").mkdir()
    _, single = make_run(
        tmp_path / "single", steps=2, config={"devices": [0], "accumulation": 2}
    )
    run_script(single, PYTHON)
    _, split = make_run(
        tmp_path / "split",
        steps=2,
        config={
            "devices": [0, 1],
            "strategy": "split",
            "placement": {"embedding": 0, "shared_transformer": 1, "vocabulary": 0},
            "accumulation": 2,
        },
    )
    run_script(split, PYTHON)
    _, ddp = make_run(
        tmp_path / "ddp", steps=2, config={"devices": [0, 1], "strategy": "ddp"}
    )
    result = subprocess.run(
        [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "train.py",
        ],
        cwd=ddp,
        env=dict(os.environ, WEAVE_RUN_DIR=str(ddp), CUDA_DEVICE_ORDER="PCI_BUS_ID"),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = state(single)
    for other in (state(split), state(ddp)):
        assert other["tokens"] == expected["tokens"]
        for key, value in expected["model"].items():
            torch.testing.assert_close(other["model"][key], value, rtol=2e-4, atol=2e-5)
    assert len(state(ddp)["rank_rng"]) == 2


def test_shared_weights_cannot_be_split(tmp_path):
    service, directory = make_run(tmp_path, steps=1)
    project = tiny_project()
    project["graph"]["nodes"][3]["data"]["params"]["tie"] = "emb"
    source = json.loads((directory / "stage.json").read_text())
    source["config"].update(
        devices=[0, 1],
        strategy="split",
        placement={"embedding": 0, "shared_transformer": 1, "vocabulary": 1},
    )
    info = service.launch(Launch(project=project, stage=Stage(**source)))
    invalid = service.run_dir(info["id"])
    result = subprocess.run(
        [PYTHON, "train.py"],
        cwd=invalid,
        env=dict(
            os.environ, WEAVE_RUN_DIR=str(invalid), CUDA_DEVICE_ORDER="PCI_BUS_ID"
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Keep shared/tied modules on one GPU" in result.stderr


def test_training_with_separate_evaluation_gpu(tmp_path):
    service, directory = make_run(tmp_path, steps=2)
    stage = json.loads((directory / "stage.json").read_text())
    stage["config"].update(
        execution="local",
        python=PYTHON,
        strategy="train_eval",
        devices=[0],
        evaluation_device=1,
        eval_dataset=str(tmp_path / "train.jsonl"),
    )
    info = service.launch(Launch(project=tiny_project(), stage=Stage(**stage)))
    directory = service.run_dir(info["id"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = service.status(directory)
        if status["state"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert status["state"] == "completed", (directory / "output.log").read_text()
    evaluations = [
        event
        for event in events(directory)
        if event.get("channel") == "evaluation" and event["kind"] == "metrics"
    ]
    assert evaluations, (
        (directory / "evaluation.log").read_text()
        if (directory / "evaluation.log").exists()
        else status
    )
    assert "eval/loss" in evaluations[-1]["payload"]


def test_moe_auxiliary_gradients_cross_split_devices():
    source = """
import torch
from torch import nn
from backend.runtime import SparseMoE, moe_auxiliary_loss
modules = nn.ModuleList([SparseMoE(4,[nn.Linear(4,4),nn.Linear(4,4)],1,.01).to(f"cuda:{i}") for i in range(2)])
for index,module in enumerate(modules):
    module(torch.randn(1,8,4,device=f"cuda:{index}"))
loss=moe_auxiliary_loss(modules)
loss.backward()
assert all(module.router.weight.grad is not None and torch.isfinite(module.router.weight.grad).all() for module in modules)
"""
    result = subprocess.run(
        [PYTHON, "-c", source],
        cwd=ROOT,
        env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
