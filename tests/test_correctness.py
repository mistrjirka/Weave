"""Independent numerical, training, and causality checks for generated PyTorch."""

import subprocess
import sys
import textwrap

import pytest
import torch

from backend.compiler import compile_project
from backend.templates import edge, node, templates
from test_codegen import instantiate, reduced


@pytest.mark.parametrize("model_id", ["transformer", "looped", "needle1"])
def test_exported_package_trains_without_backend_and_loss_decreases(model_id, tmp_path):
    project = reduced(next(p for p in templates() if p["id"] == model_id))
    for name, source in compile_project(project).items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    input_count = sum(n["data"]["kind"] == "input" for n in project["graph"]["nodes"])
    script = textwrap.dedent(f"""
        import sys
        import torch
        from torch.nn import functional as F
        from model import Model
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = Model().train()
        inputs = [torch.randint(0, 32, (2, 5)) for _ in range({input_count})]
        targets = torch.randint(0, 32, (2, 5))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        losses = []
        for step in range(25):
            optimizer.zero_grad(set_to_none=True)
            output = model(*inputs)
            loss = F.cross_entropy(output.reshape(-1, 32), targets.flatten())
            losses.append(float(loss.detach()))
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        assert losses[-1] < losses[0] * 0.5, losses
        assert not any(k == 'backend' or k.startswith('backend.') for k in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_shared_loop_residual_has_correct_values_and_derivatives(monkeypatch):
    body = {
        "nodes": [
            node("x", "input", "X", 0, 0, {"shape": ["B", "T", 2], "dtype": "float32"}),
            node("linear", "linear", "Projection", 0, 0, {"out": 2, "bias": True}),
            node("activation", "silu", "Activation", 0, 0),
            node("residual", "add", "Residual", 0, 0),
            node("out", "output", "Out", 0, 0),
        ],
        "edges": [
            edge("x", "linear"),
            edge("linear", "activation"),
            edge("x", "residual"),
            edge("activation", "residual", "b"),
            edge("residual", "out"),
        ],
    }
    project = {
        "version": 1,
        "name": "Gradient check",
        "config": {"width": 2},
        "graph": {
            "nodes": [
                node(
                    "x",
                    "input",
                    "X",
                    0,
                    0,
                    {"shape": ["B", "T", 2], "dtype": "float32"},
                ),
                node(
                    "repeat",
                    "loop",
                    "Repeat",
                    0,
                    0,
                    {"iterations": 3, "shared": True},
                    body,
                ),
                node("out", "output", "Out", 0, 0),
            ],
            "edges": [edge("x", "repeat"), edge("repeat", "out")],
        },
    }
    model, _ = instantiate(project, monkeypatch)
    model = model.double()
    torch.manual_seed(4)
    weight = torch.randn(2, 2, dtype=torch.double, requires_grad=True) * 0.2
    bias = torch.randn(2, dtype=torch.double, requires_grad=True) * 0.1
    x = torch.randn(1, 2, 2, dtype=torch.double, requires_grad=True)
    names = list(dict(model.named_parameters()))
    assert len(names) == 2  # The three applications really share one layer.

    def generated(x, weight, bias):
        return torch.func.functional_call(model, dict(zip(names, (weight, bias))), (x,))

    def handwritten(x, weight, bias):
        for _ in range(3):
            x = x + torch.nn.functional.silu(
                torch.nn.functional.linear(x, weight, bias)
            )
        return x

    torch.testing.assert_close(generated(x, weight, bias), handwritten(x, weight, bias))
    actual = torch.autograd.grad(
        generated(x, weight, bias).square().sum(), (x, weight, bias)
    )
    expected = torch.autograd.grad(
        handwritten(x, weight, bias).square().sum(), (x, weight, bias)
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert torch.autograd.gradcheck(generated, (x, weight, bias), fast_mode=True)


@pytest.mark.parametrize("model_id", ["transformer", "looped", "sparse"])
def test_causal_export_does_not_leak_future_tokens(model_id, monkeypatch):
    project = reduced(next(p for p in templates() if p["id"] == model_id))
    model, _ = instantiate(project, monkeypatch)
    model.eval()
    original = torch.arange(8).reshape(1, 8)
    changed = original.clone()
    changed[:, 4:] += 10
    with torch.no_grad():
        a, b = model(original), model(changed)
    torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=1e-6)
    assert not torch.allclose(a[:, 4:], b[:, 4:])
