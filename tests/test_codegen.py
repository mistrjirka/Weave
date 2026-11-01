"""Export contract: editable Python names, executable packages, and training gradients."""

import ast
import copy
import inspect
import io
import json
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from backend import runtime, export_helpers, sparse_runtime
from backend.compiler import compile_project
from backend.core import validate_project
from backend.templates import templates, node, edge, transformer


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


def test_tied_head_shares_one_parameter_and_receives_both_gradients(monkeypatch):
    project = reduced(next(p for p in templates() if p["id"] == "transformer"))
    model, files = instantiate(project, monkeypatch)
    names = json.loads(files["graph_modules.json"])
    embedding = getattr(model, names["emb"])
    head = getattr(model, names["head"])
    assert head.weight is embedding.weight
    assert sum(p.numel() for p in model.parameters()) == validate_project(project)["parameters"]
    ids = torch.randint(0, 32, (2, 4))
    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, 32), torch.randint(0, 32, (8,)))
    loss.backward()
    assert embedding.weight.grad is head.weight.grad
    assert embedding.weight.grad is not None
    assert torch.isfinite(embedding.weight.grad).all()
    before = embedding.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(before, embedding.weight)
    assert head.weight is embedding.weight


@pytest.mark.parametrize("head_first", [False, True])
def test_tied_initialization_is_once_per_matrix_and_leaves_other_parameters_alone(monkeypatch, head_first):
    model = nn.Module()
    embedding = nn.Embedding(1024, 64)
    head = nn.Linear(64, 1024, bias=True)
    head.weight = embedding.weight
    model.add_module("first", head if head_first else embedding)
    model.add_module("second", embedding if head_first else head)
    model.alias = embedding
    model.other_embedding = nn.Embedding(1024, 64)
    model.other_linear = nn.Linear(64, 64)
    model.norm = nn.RMSNorm(64)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    weight = embedding.weight
    calls = []
    normal = nn.init.normal_

    def record(tensor, *args, **kwargs):
        calls.append(id(tensor))
        return normal(tensor, *args, **kwargs)

    monkeypatch.setattr(nn.init, "normal_", record)
    runtime.initialize_tied_embeddings(model)
    assert calls == [id(weight)]
    assert embedding.weight is head.weight is weight
    assert weight.std().item() == pytest.approx(0.02, rel=0.02)
    for name, value in model.named_parameters():
        if value is not weight:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_tied_head_counts_independent_bias(monkeypatch):
    project = reduced(next(p for p in templates() if p["id"] == "transformer"))
    head = next(n for n in project["graph"]["nodes"] if n["id"] == "head")
    head["data"]["params"]["bias"] = True
    report = validate_project(project)
    assert report["valid"]
    model, files = instantiate(project, monkeypatch)
    names = json.loads(files["graph_modules.json"])
    embedding = getattr(model, names["emb"])
    output = getattr(model, names["head"])
    assert output.weight is embedding.weight
    assert output.bias is not None
    assert output.bias.numel() == project["config"]["vocab"]
    assert sum(p.numel() for p in model.parameters()) == report["parameters"]


def test_weight_group_shares_matrix_and_same_kind_head_bias(monkeypatch):
    project = reduced(next(p for p in templates() if p["id"] == "transformer"))
    graph = project["graph"]
    head = next(n for n in graph["nodes"] if n["id"] == "head")
    head["data"]["params"]["bias"] = True
    second = copy.deepcopy(head)
    second["id"] = "head2"
    graph["nodes"].append(second)
    graph["nodes"].append(node("out2", "output", "Second logits", 0, 0, {"index": 1}))
    graph["edges"].extend([edge("norm", "head2"), edge("head2", "out2")])
    report = validate_project(project)
    assert report["valid"], report["errors"]
    model, files = instantiate(project, monkeypatch)
    names = json.loads(files["graph_modules.json"])
    embedding = getattr(model, names["emb"])
    first = getattr(model, names["head"])
    other = getattr(model, names["head2"])
    assert first is other
    assert first is not embedding
    assert first.weight is embedding.weight
    assert first.bias is other.bias
    assert sum(p.numel() for p in model.parameters()) == report["parameters"]
    outputs = model(torch.randint(0, 32, (2, 4)))
    sum(x.square().mean() for x in outputs).backward()
    assert embedding.weight.grad is first.weight.grad
    assert first.bias.grad is not None


def test_legacy_multiple_tied_heads_keep_separate_biases(monkeypatch):
    project = reduced(next(p for p in templates() if p["id"] == "transformer"))
    graph = project["graph"]
    embedding = next(n for n in graph["nodes"] if n["id"] == "emb")
    embedding["data"].pop("share")
    head = next(n for n in graph["nodes"] if n["id"] == "head")
    head["data"].pop("share")
    head["data"]["params"].update(tie="emb", bias=True)
    second = copy.deepcopy(head)
    second["id"] = "head2"
    graph["nodes"].extend([second, node("out2", "output", "Second logits", 0, 0, {"index": 1})])
    graph["edges"].extend([edge("norm", "head2"), edge("head2", "out2")])
    report = validate_project(project)
    assert report["valid"], report["errors"]
    model, files = instantiate(project, monkeypatch)
    names = json.loads(files["graph_modules.json"])
    first = getattr(model, names["head"])
    other = getattr(model, names["head2"])
    assert first.weight is other.weight is getattr(model, names["emb"]).weight
    assert first.bias is not other.bias
    assert sum(p.numel() for p in model.parameters()) == report["parameters"]


def test_nested_head_first_group_shares_matrix_through_independent_loop(monkeypatch):
    group = "vocabulary"
    head_body = {
        "nodes": [
            node("x", "input", "Features", 0, 0, {"index": 0, "shape": ["B", "T", 4], "dtype": "float32"}),
            node("head", "head", "Projection", 1, 0, {"out": 4, "bias": True}, share=group),
            node("y", "output", "Logits", 2, 0),
        ],
        "edges": [edge("x", "head"), edge("head", "y")],
    }
    embedding_body = {
        "nodes": [
            node("x", "input", "IDs", 0, 0, {"index": 0, "shape": ["B", "T"], "dtype": "int64"}),
            node("emb", "embedding", "Table", 1, 0, {"vocab": 4, "width": 4}, share=group),
            node("y", "output", "Features", 2, 0),
        ],
        "edges": [edge("x", "emb"), edge("emb", "y")],
    }
    project = {
        "version": 1, "id": "nested_matrix", "name": "Nested matrix",
        "config": {"width": 4, "vocab": 4},
        "graph": {
            "nodes": [
                node("features", "input", "Features", 0, 0, {"index": 0, "shape": ["B", "T", 4], "dtype": "float32"}),
                node("loop", "loop", "Head loop", 1, 0, {"iterations": 3, "shared": False}, head_body),
                node("logits", "output", "Logits", 2, 0, {"index": 0}),
                node("ids", "input", "IDs", 0, 1, {"index": 1, "shape": ["B", "T"], "dtype": "int64"}),
                node("block", "block", "Embedding block", 1, 1, {}, embedding_body),
                node("embedded", "output", "Embedded", 2, 1, {"index": 1}),
            ],
            "edges": [edge("features", "loop"), edge("loop", "logits"), edge("ids", "block"), edge("block", "embedded")],
        },
    }
    report = validate_project(project)
    assert report["valid"], report["errors"]
    assert report["parameters"] == 20
    model, files = instantiate(project, monkeypatch)
    names = json.loads(files["graph_modules.json"])
    loops = getattr(model, names["loop"])
    heads = [next(m for m in layer.modules() if isinstance(m, nn.Linear)) for layer in loops]
    embedding = next(m for m in getattr(model, names["block"]).modules() if isinstance(m, nn.Embedding))
    assert heads[0] is heads[1] is heads[2]
    assert heads[0].weight is embedding.weight
    assert files["model.py"].count("initialize_tied_embeddings(self, std=0.02)") == 1
    assert embedding.weight.abs().max() < 0.1
    assert heads[0].bias is not None
    assert sum(p.numel() for p in model.parameters()) == report["parameters"]
    logits, embedded = model(torch.randn(2, 3, 4), torch.randint(0, 4, (2, 3)))
    (logits.square().mean() + embedded.square().mean()).backward()
    assert embedding.weight.grad is heads[0].weight.grad


def test_mixed_weight_group_rejects_matrix_shape_mismatch():
    project = reduced(next(p for p in templates() if p["id"] == "transformer"))
    head = next(n for n in project["graph"]["nodes"] if n["id"] == "head")
    head["data"]["params"]["out"] = 33
    report = validate_project(project)
    assert not report["valid"]
    assert any("matrix dimensions" in error["message"] for error in report["errors"])


def test_legacy_ties_in_independent_loop_keep_each_iteration_separate(monkeypatch):
    body = {
        "nodes": [
            node("features", "input", "Features", 0, 0, {"index": 0, "shape": ["B", "T", 4], "dtype": "float32"}),
            node("ids", "input", "IDs", 0, 1, {"index": 1, "shape": ["B", "T"], "dtype": "int64"}),
            node("emb", "embedding", "Table", 1, 1, {"vocab": 4, "width": 4}),
            node("head", "head", "Projection", 1, 0, {"out": 4, "bias": True, "tie": "emb"}),
            node("y", "output", "Output", 2, 0),
        ],
        "edges": [edge("ids", "emb"), edge("features", "head"), edge("head", "y")],
    }
    project = {
        "version": 1, "id": "legacy_loop", "name": "Legacy loop",
        "config": {"width": 4, "vocab": 4},
        "graph": {
            "nodes": [
                node("features", "input", "Features", 0, 0, {"index": 0, "shape": ["B", "T", 4], "dtype": "float32"}),
                node("ids", "input", "IDs", 0, 1, {"index": 1, "shape": ["B", "T"], "dtype": "int64"}),
                node("loop", "loop", "Loop", 1, 0, {"iterations": 3, "shared": False}, body),
                node("out", "output", "Output", 2, 0),
            ],
            "edges": [edge("features", "loop"), edge("ids", "loop", "b"), edge("loop", "out")],
        },
    }
    report = validate_project(project)
    assert report["valid"], report["errors"]
    assert report["parameters"] == 60
    model, files = instantiate(project, monkeypatch)
    names = json.loads(files["graph_modules.json"])
    layers = list(getattr(model, names["loop"]))
    embeddings = [next(m for m in layer.modules() if isinstance(m, nn.Embedding)) for layer in layers]
    heads = [next(m for m in layer.modules() if isinstance(m, nn.Linear)) for layer in layers]
    assert all(head.weight is embedding.weight for head, embedding in zip(heads, embeddings))
    assert len({id(embedding.weight) for embedding in embeddings}) == 3
    assert files["model.py"].count("initialize_tied_embeddings(self, std=0.02)") == 1
    assert all(embedding.weight.abs().max() < 0.1 for embedding in embeddings)
    assert sum(p.numel() for p in model.parameters()) == report["parameters"]


def test_shared_three_block_model_count_and_causality(monkeypatch):
    config = {"vocab": 16000, "width": 768, "heads": 12, "ffn_width": 2176}

    def layer(name):
        body = transformer(heads=12, kv=12)
        attention = next(n for n in body["nodes"] if n["id"] == "attn")
        attention["data"]["params"]["qk_norm"] = False
        attention["data"]["label"] = "Multi-head attention (MHA)"
        return node(name, "block", name, 0, 0, {}, body)

    loop_body = {
        "nodes": [
            node("x", "input", "Input", 0, 0, {"index": 0, "shape": ["B", "T", "$width"], "dtype": "float32"}),
            layer("a"), layer("b"), layer("c"),
            node("y", "output", "Output", 0, 0),
        ],
        "edges": [edge("x", "a"), edge("a", "b"), edge("b", "c"), edge("c", "y")],
    }
    project = {
        "version": 1, "id": "shared_3", "name": "Shared three block model",
        "config": config,
        "graph": {
            "nodes": [
                node("ids", "input", "Token IDs", 0, 0, {"index": 0, "shape": ["B", "T"], "dtype": "int64"}),
                node("emb", "embedding", "Token embedding", 0, 0, {"width": "$width", "vocab": "$vocab"}),
                layer("prelude"),
                node("shared", "loop", "Shared blocks", 0, 0, {"iterations": 4, "shared": True}, loop_body),
                layer("coda"),
                node("norm", "norm", "Final norm", 0, 0),
                node("head", "head", "Tied head", 0, 0, {"out": "$vocab", "tie": "emb"}),
                node("out", "output", "Logits", 0, 0),
            ],
            "edges": [edge("ids", "emb"), edge("emb", "prelude"), edge("prelude", "shared"),
                      edge("shared", "coda"), edge("coda", "norm"), edge("norm", "head"), edge("head", "out")],
        },
    }
    report = validate_project(project)
    assert report["valid"], report["errors"]
    assert report["parameters"] == 49_160_448
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, files = instantiate(project, monkeypatch)
        names = json.loads(files["graph_modules.json"])
        assert getattr(model, names["head"]).weight is getattr(model, names["emb"]).weight
        assert sum(p.numel() for p in model.parameters()) == 49_160_448

        # A changed future token cannot alter logits at an earlier position.
        model.eval()
        first = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 3, 5]])
        with torch.no_grad():
            a, b = model(first), model(changed)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=1e-5, atol=1e-5)
        # Check initialization as well as wiring: PyTorch's N(0,1) embedding
        # default previously produced initial next-token losses in the hundreds.
        embedding = getattr(model, names["emb"])
        assert embedding.weight.std().item() == pytest.approx(0.02, rel=0.02)
        loss = F.cross_entropy(a[:, :-1].reshape(-1, 16000), first[:, 1:].reshape(-1))
        assert 8 < loss.item() < 12
        assert a.std().item() < 1.5
        training_logits = model(first)
        F.cross_entropy(training_logits[:, :-1].reshape(-1, 16000), first[:, 1:].reshape(-1)).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

        # A fresh constructor may initialize weights, but loading a checkpoint
        # must restore the learned values without changing the weight alias.
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        restored, _ = instantiate(project, monkeypatch)
        checkpoint.seek(0)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        assert getattr(restored, names["head"]).weight is getattr(restored, names["emb"]).weight
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(first), a, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


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
