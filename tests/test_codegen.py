"""Export contract: editable Python names, executable packages, and training gradients."""

import ast
import copy
import inspect
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from backend import runtime, export_helpers, sparse_runtime
from backend.compiler import compile_project
from backend.core import validate_project
from backend.templates import templates, node, edge


def instantiate(project, monkeypatch):
    files = compile_project(project)
    for name, module in (
        ("runtime", runtime),
        ("export_helpers", export_helpers),
        ("sparse_runtime", sparse_runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    namespace = {}
    exec(files["model.py"], namespace)
    return namespace["Model"](), files


def reduced(project):
    result = copy.deepcopy(project)
    result["config"].update(width=16, heads=4, ffn_width=32, vocab=32)

    def shrink(graph):
        for n in graph["nodes"]:
            settings = n["data"]["params"]
            if n["data"]["kind"] in ("attention", "cross_attention"):
                settings.update(heads=4, kv_heads=2, head_dim=4, value_dim=4)
            if n["data"]["kind"] in ("engram_hash", "engram_lookup"):
                settings.update(hash_heads=1, buckets=32, sub_dim=4)
            if n["data"]["kind"] == "scale":
                settings["factor"] = 4.0
            if n["data"].get("body"):
                shrink(n["data"]["body"])

    shrink(result["graph"])
    return result


@pytest.mark.parametrize(
    "model_id",
    ["transformer", "looped", "needle1", "needle2", "needle3", "thimble", "sparse"],
)
def test_readable_exports_support_training_and_checkpoint_reload(model_id, monkeypatch):
    project = reduced(next(p for p in templates() if p["id"] == model_id))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, files = instantiate(project, monkeypatch)
        model.train()
        count = sum(n["data"]["kind"] == "input" for n in project["graph"]["nodes"])
        inputs = [torch.randint(0, 32, (2, 5)) for _ in range(count)]
        logits = model(*inputs)
        targets = torch.randint(0, 32, logits.shape[:-1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        original = next(model.parameters()).detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten())
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()
        assert not torch.equal(original, next(model.parameters()))
        restored, _ = instantiate(project, monkeypatch)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(*inputs), model(*inputs))
        assert (
            sum(p.numel() for p in model.parameters())
            == validate_project(project)["parameters"]
        )
        source = files["model.py"]
        assert "make_module" not in source and "ModuleDict" not in source
        assert "*inputs" not in source and "Loop(" not in source
        assert not any(
            isinstance(n, ast.ClassDef)
            and n.name in {"Loop", "Functional"}
            or isinstance(n, ast.FunctionDef)
            and n.name == "make_module"
            for n in ast.parse(files["runtime.py"]).body
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_python_names_handle_duplicates_reserved_words_and_non_ascii(monkeypatch):
    labels = [
        "forward",
        "forward",
        "state_dict",
        "class",
        "123 linear",
        "🧠",
        "x\n# not executable",
    ]
    nodes = [
        node("x", "input", "torch", 0, 0, {"shape": ["B", "T", 8], "dtype": "float32"})
    ]
    edges = []
    last = "x"
    for index, label in enumerate(labels):
        current = f"linear{index}"
        nodes.append(node(current, "linear", label, 0, 0, {"out": 8}))
        edges.append(edge(last, current))
        last = current
    nodes.append(node("out", "output", "Output", 0, 0))
    edges.append(edge(last, "out"))
    project = {
        "version": 1,
        "name": "Model",
        "config": {"width": 8},
        "graph": {"nodes": nodes, "edges": edges},
    }
    model, files = instantiate(project, monkeypatch)
    assert callable(model.forward) and callable(model.state_dict)
    assert model(torch.randn(2, 3, 8)).shape == (2, 3, 8)
    assert len(list(model.children())) == len(labels)
    assert files == compile_project(project)


def test_tensor_inputs_cannot_shadow_imported_helpers(monkeypatch):
    project = {
        "version": 1,
        "name": "Rotary test",
        "config": {"width": 4},
        "graph": {
            "nodes": [
                node(
                    "x",
                    "input",
                    "rotary_positions",
                    0,
                    0,
                    {"shape": ["B", 2, "T", 4], "dtype": "float32"},
                ),
                node("rope", "rope", "for", 0, 0),
                node("out", "output", "Output", 0, 0),
            ],
            "edges": [edge("x", "rope"), edge("rope", "out")],
        },
    }
    model, _ = instantiate(project, monkeypatch)
    x = torch.randn(2, 2, 3, 4)
    torch.testing.assert_close(model(x), export_helpers.rotary_positions(x))
    assert list(inspect.signature(model.forward).parameters) == ["rotary_positions_2"]


def test_reused_bodies_keep_distinct_effective_attention_settings(monkeypatch):
    body = {
        "nodes": [
            node(
                "x",
                "input",
                "Hidden states",
                0,
                0,
                {"index": 0, "shape": ["B", "T", 8], "dtype": "float32"},
            ),
            node("attention", "attention", "Attention", 0, 0),
            node("out", "output", "Output", 0, 0),
        ],
        "edges": [edge("x", "attention"), edge("attention", "out")],
    }
    project = {
        "version": 1,
        "name": "Different heads",
        "config": {"width": 8, "heads": 2},
        "graph": {
            "nodes": [
                node(
                    "x",
                    "input",
                    "Hidden states",
                    0,
                    0,
                    {"shape": ["B", "T", 8], "dtype": "float32"},
                ),
                node(
                    "first",
                    "block",
                    "First block",
                    0,
                    0,
                    {"heads": 2},
                    copy.deepcopy(body),
                ),
                node(
                    "second",
                    "block",
                    "Second block",
                    0,
                    0,
                    {"heads": 4},
                    copy.deepcopy(body),
                ),
                node("out", "output", "Output", 0, 0),
            ],
            "edges": [
                edge("x", "first"),
                edge("first", "second"),
                edge("second", "out"),
            ],
        },
    }
    model, _ = instantiate(project, monkeypatch)
    assert model.first_block.attention.h == 2
    assert model.second_block.attention.h == 4
    assert model(torch.randn(1, 3, 8)).shape == (1, 3, 8)


def test_public_signatures_and_examples_preserve_optional_input_positions(monkeypatch):
    import types

    project = {
        "version": 1,
        "name": "Multiple inputs",
        "config": {"width": 8},
        "graph": {
            "nodes": [
                node(
                    "x",
                    "input",
                    "Features",
                    0,
                    0,
                    {"index": 0, "shape": ["B", "T", 8], "dtype": "float32"},
                ),
                node(
                    "mask",
                    "input",
                    "Optional mask",
                    0,
                    0,
                    {
                        "index": 1,
                        "shape": ["B", "T"],
                        "dtype": "bool",
                        "optional": True,
                    },
                ),
                node(
                    "memory",
                    "input",
                    "Memory",
                    0,
                    0,
                    {"index": 2, "shape": ["B", "T", 8], "dtype": "float32"},
                ),
                node("sum", "add", "Combined features", 0, 0),
                node("out", "output", "Output", 0, 0),
            ],
            "edges": [edge("x", "sum"), edge("memory", "sum", "b"), edge("sum", "out")],
        },
    }
    model, files = instantiate(project, monkeypatch)
    x, memory = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
    torch.testing.assert_close(model(x, None, memory), x + memory)
    module = types.ModuleType("model")
    module.Model = type(model)
    monkeypatch.setitem(sys.modules, "model", module)
    namespace = {}
    exec(files["example.py"], namespace)
    assert namespace["output"].shape == (1, 8, 8)
