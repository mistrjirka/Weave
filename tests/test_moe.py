import copy
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from backend.compiler import compile_project
from backend.core import validate_project
from backend.runtime import SparseMoE, clear_moe_losses, moe_auxiliary_loss
from backend.templates import templates


def example(name):
    return copy.deepcopy(next(p for p in templates() if p["id"] == name))


def moe_node(project):
    return next(
        n
        for n in next(
            n for n in project["graph"]["nodes"] if n["data"]["kind"] == "loop"
        )["data"]["body"]["nodes"]
        if n["data"]["kind"] == "moe"
    )


def test_examples_validate_and_bad_moe_shapes_report():
    assert validate_project(example("embedding-table"))["valid"]
    project = example("moe-transformer")
    assert validate_project(project)["valid"]
    moe = moe_node(project)
    moe["data"]["params"]["top_k"] = 5
    report = validate_project(project)
    assert not report["valid"] and any(
        "top k" in e["message"] for e in report["errors"]
    )
    moe["data"]["params"]["top_k"] = 2
    expert = next(
        n for n in moe["data"]["body"]["nodes"] if n["data"].get("role") == "moe-expert"
    )
    ffn = next(n for n in expert["data"]["body"]["nodes"] if n["data"]["kind"] == "ffn")
    ffn["data"] = {"kind": "linear", "label": "Wrong width", "params": {"out": 7}}
    report = validate_project(project)
    assert not report["valid"] and any(
        "expert" in e["message"].lower() for e in report["errors"]
    )


def test_sparse_routing_matches_reference_and_top1_router_has_gradient():
    torch.manual_seed(3)
    experts = [nn.Linear(3, 3, bias=False) for _ in range(3)]
    moe = SparseMoE(3, experts, top_k=1, balance_weight=0.01)
    x = torch.randn(2, 4, 3, requires_grad=True)
    actual = moe(x)
    probabilities = moe.router(x).softmax(-1)
    weights, indices = probabilities.topk(1, dim=-1)
    all_outputs = torch.stack([expert(x) for expert in experts], dim=-2)
    expected = (
        all_outputs.gather(-2, indices.unsqueeze(-1).expand(-1, -1, -1, 3))
        * weights.unsqueeze(-1)
    ).sum(-2)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    assert moe_auxiliary_loss(moe).item() > 0
    clear_moe_losses(moe)
    assert moe_auxiliary_loss(moe).item() == 0
    moe.eval()
    moe(x)
    assert moe_auxiliary_loss(moe).item() == 0


def test_exports_run_and_parameter_count_matches(tmp_path):
    for name in ("embedding-table", "moe-transformer"):
        project = example(name)
        # Keep the example fast while exercising the same paths.
        project["config"].update(width=16, ffn_width=32, vocab=32, heads=4)
        if name == "moe-transformer":
            attention = next(
                n
                for n in next(
                    n for n in project["graph"]["nodes"] if n["data"]["kind"] == "loop"
                )["data"]["body"]["nodes"]
                if n["data"]["kind"] == "attention"
            )
            attention["data"]["params"].update(heads=4, kv_heads=2)
        report = validate_project(project)
        assert report["valid"], report["errors"]
        directory = tmp_path / name
        directory.mkdir()
        for filename, content in compile_project(project).items():
            path = directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for script in (
            ["example.py", "train_example.py"]
            if name == "moe-transformer"
            else ["example.py"]
        ):
            subprocess.run(
                [sys.executable, script],
                cwd=directory,
                check=True,
                capture_output=True,
                timeout=60,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
        check = f'from model import Model\nm=Model()\nassert sum(p.numel() for p in m.parameters()) == {report["parameters"]}\n'
        subprocess.run(
            [sys.executable, "-c", check],
            cwd=directory,
            check=True,
            capture_output=True,
            timeout=60,
        )


def test_top2_routing_is_sparse_and_differentiable():
    torch.manual_seed(11)

    class CountingExpert(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(float(scale)))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x * self.scale

    experts = [CountingExpert(i + 1) for i in range(4)]
    moe = SparseMoE(2, experts, top_k=2, balance_weight=0.01)
    with torch.no_grad():
        moe.router.weight.copy_(
            torch.tensor([[5.0, 0.0], [4.0, 0.0], [-4.0, 0.0], [-5.0, 0.0]])
        )
    x = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
    actual = moe(x)
    probabilities = moe.router(x).softmax(-1)
    weights = probabilities[..., :2] / probabilities[..., :2].sum(-1, keepdim=True)
    expected = x * (weights[..., 0:1] + 2 * weights[..., 1:2])
    torch.testing.assert_close(actual, expected)
    (actual.sum() + moe_auxiliary_loss(moe)).backward()
    assert [e.calls for e in experts] == [1, 1, 0, 0]
    assert experts[0].scale.grad is not None and experts[2].scale.grad is None
    assert moe.router.weight.grad.abs().sum() > 0


def test_invalid_scaffold_and_expert_count_report_errors():
    project = example("moe-transformer")
    moe = moe_node(project)
    moe["data"]["body"]["edges"].pop()
    report = validate_project(project)
    assert not report["valid"] and any(
        "scaffold" in e["message"] for e in report["errors"]
    )
    project = example("moe-transformer")
    moe_node(project)["data"]["params"]["experts"] = 3
    report = validate_project(project)
    assert not report["valid"] and any(
        "expert" in e["message"].lower() for e in report["errors"]
    )


def test_expert_weight_groups_share_modules_in_export(tmp_path):
    project = example("moe-transformer")
    project["config"].update(width=16, ffn_width=32, vocab=32, heads=4)
    body = next(n for n in project["graph"]["nodes"] if n["data"]["kind"] == "loop")[
        "data"
    ]["body"]
    next(n for n in body["nodes"] if n["data"]["kind"] == "attention")["data"][
        "params"
    ].update(heads=4, kv_heads=2)
    experts = [
        n
        for n in moe_node(project)["data"]["body"]["nodes"]
        if n["data"].get("role") == "moe-expert"
    ]
    experts[0]["data"]["share"] = experts[1]["data"]["share"] = "shared-expert"
    report = validate_project(project)
    assert report["valid"], report["errors"]
    for filename, content in compile_project(project).items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    check = f"""from model import Model
m = Model()
a, b = m.transformer_layers[0].mixture_of_experts.routing.experts[:2]
assert a is b
assert sum(p.numel() for p in m.parameters()) == {report['parameters']}
"""
    subprocess.run(
        [sys.executable, "-c", check],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
