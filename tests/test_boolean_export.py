"""Boolean settings survive validation and compact Python export."""

import copy
import inspect
import json
import sys

import pytest
import torch
from torch import nn

from backend import export_helpers, runtime, sparse_runtime
from backend.compiler import compile_project
from backend.codegen import attention_constructor
from backend.core import validate_project
from backend.sparse_runtime import AttentionPattern
from backend.templates import edge, node


def project(kind, settings=None, *, shape=None):
    shape = shape or [2, 4, 8]
    return {
        "version": 1, "id": "boolean_export", "name": "Boolean export",
        "config": {"width": 8, "heads": 2, "vocab": 16},
        "graph": {
            "nodes": [
                node("x", "input", "Input", 0, 0, {"index": 0, "shape": shape, "dtype": "float32"}),
                node("op", kind, "Operation", 200, 0, settings or {}),
                node("out", "output", "Output", 400, 0),
            ],
            "edges": [edge("x", "op"), edge("op", "out")],
        },
    }


def exported(project_data, monkeypatch):
    report = validate_project(project_data)
    assert report["valid"], report["errors"]
    files = compile_project(project_data)
    for name, module in (("runtime", runtime), ("export_helpers", export_helpers), ("sparse_runtime", sparse_runtime)):
        monkeypatch.setitem(sys.modules, name, module)
    namespace = {}
    exec(files["model.py"], namespace)
    model = namespace["Model"]().eval()
    module_name = json.loads(files["graph_modules.json"])["op"]
    assert sum(p.numel() for p in model.parameters()) == report["parameters"]
    return getattr(model, module_name), model, files["model.py"]


@pytest.mark.parametrize("enabled", [True, False, None])
def test_qk_norm_on_off_and_omitted(enabled, monkeypatch):
    settings = {"heads": 2, "kv_heads": 1, "rope": False}
    if enabled is not None:
        settings["qk_norm"] = enabled
    layer, model, source = exported(project("attention", settings), monkeypatch)
    expected = enabled is True
    assert isinstance(layer.q_norm, nn.RMSNorm if expected else nn.Identity)
    assert isinstance(layer.k_norm, nn.RMSNorm if expected else nn.Identity)
    assert ("qk_norm=True" in source) == expected
    assert ("qk_norm=False" in source) is False  # compact default is explicitly off
    assert model(torch.randn(2, 4, 8)).shape == (2, 4, 8)


def test_compact_attention_defaults_match_runtime_signature():
    signature = inspect.signature(runtime.Attention)
    for cross in (False, True):
        source = attention_constructor({}, {"heads": 2}, [2, 4, 8], cross=cross)
        expected = {
            "causal": not cross, "rope": True, "qk_norm": False,
            "qkv_bias": False, "output_bias": False, "gated": False,
            "conv_respects_window": False,
        }
        for key, value in expected.items():
            default = signature.parameters[key].default
            if value == default:
                assert f"{key}=" not in source
            else:
                assert f"{key}={value!r}" in source


@pytest.mark.parametrize("causal", [True, False, None])
def test_cross_attention_causal_on_off_and_omitted(causal, monkeypatch):
    settings = {"heads": 2, "kv_heads": 1, "rope": False}
    if causal is not None:
        settings["causal"] = causal
    data = project("cross_attention", settings)
    data["graph"]["nodes"].insert(1, node("memory", "input", "Memory", 0, 100, {"index": 1, "shape": [2, 4, 8], "dtype": "float32"}))
    data["graph"]["edges"].append(edge("memory", "op", "b"))
    layer, model, source = exported(data, monkeypatch)
    expected = causal is True
    assert layer.causal is expected
    assert ("causal=False" in source) is not expected
    assert "causal=True" not in source  # True is the Attention constructor default.
    assert model(torch.randn(2, 4, 8), torch.randn(2, 4, 8)).shape == (2, 4, 8)


@pytest.mark.parametrize("enabled", [True, False, None])
def test_attention_boolean_constructor_and_runtime_parity(enabled, monkeypatch):
    settings = {"heads": 2, "kv_heads": 1, "conv_taps": 3, "window": 2}
    keys = ("causal", "rope", "qk_norm", "qkv_bias", "output_bias", "gated", "conv_respects_window")
    defaults = dict(causal=True, rope=True, qk_norm=False, qkv_bias=False, output_bias=False, gated=False, conv_respects_window=False)
    for key in keys:
        params = copy.deepcopy(settings)
        if enabled is not None:
            params[key] = enabled
        layer, model, source = exported(project("attention", params), monkeypatch)
        legacy = runtime.make_module("attention", params, [2, 4, 8], {"width": 8, "heads": 2, "vocab": 16})
        value = defaults[key] if enabled is None else enabled
        assert layer.causal == legacy.causal == (value if key == "causal" else True)
        assert layer.rope == legacy.rope == (value if key == "rope" else True)
        assert isinstance(layer.q_norm, type(legacy.q_norm))
        assert (layer.q.bias is not None) == (legacy.q.bias is not None) == (value if key == "qkv_bias" else False)
        assert (layer.o.bias is not None) == (legacy.o.bias is not None) == (value if key == "output_bias" else False)
        assert (layer.gate is not None) == (legacy.gate is not None) == (value if key == "gated" else False)
        assert layer.conv_respects_window == legacy.conv_respects_window == (value if key == "conv_respects_window" else False)
        if key == "qk_norm":
            assert isinstance(layer.q_norm, nn.RMSNorm if value else nn.Identity)
        if enabled is None or value == defaults[key]:
            assert f"{key}=" not in source
        else:
            assert f"{key}={value!r}" in source
        assert model(torch.randn(2, 4, 8)).shape == (2, 4, 8)


@pytest.mark.parametrize("key", ["causal", "rope", "qk_norm", "gated", "conv_respects_window"])
def test_attention_boolean_switch_changes_generated_forward(key, monkeypatch):
    base = {"heads": 2, "kv_heads": 1, "window": 2, "conv_taps": 3,
            "causal": False, "rope": False, "qk_norm": False, "gated": False,
            "conv_respects_window": False}
    models = []
    for enabled in (False, True):
        settings = dict(base, **{key: enabled})
        torch.manual_seed(11)
        layer, model, _ = exported(project("attention", settings), monkeypatch)
        if key == "conv_respects_window":
            with torch.no_grad():
                for taps in layer.taps:
                    taps[2].fill_(0.5)
        models.append(model)
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 17
    with torch.no_grad():
        outputs = [model(x) for model in models]
    assert not torch.allclose(*outputs), key


@pytest.mark.parametrize("kind,settings,shape,attribute", [
    ("linear", {"out": 8}, [2, 4, 8], "bias"),
    ("conv1d", {"out": 8, "kernel_size": 3, "padding": 1}, [2, 4, 8], "conv.bias"),
    ("conv2d", {"out_channels": 4, "kernel_size": 3, "padding": 1}, [2, 3, 4, 4], "bias"),
])
@pytest.mark.parametrize("enabled", [True, False, None])
def test_projection_and_convolution_bias(kind, settings, shape, attribute, enabled, monkeypatch):
    settings = dict(settings)
    if enabled is not None:
        settings["bias"] = enabled
    layer, model, _ = exported(project(kind, settings, shape=shape), monkeypatch)
    bias = layer
    for part in attribute.split("."):
        bias = getattr(bias, part)
    expected = (kind != "linear") if enabled is None else enabled
    assert (bias is not None) == expected
    assert model(torch.randn(*shape)).numel() > 0


@pytest.mark.parametrize("affine,bias", [(True, True), (True, False), (False, True), (False, False)])
def test_layer_norm_affine_and_bias(affine, bias, monkeypatch):
    layer, model, _ = exported(project("layer_norm", {"affine": affine, "bias": bias}), monkeypatch)
    assert (layer.weight is not None) == affine
    assert (layer.bias is not None) == (affine and bias)
    assert model(torch.randn(2, 4, 8)).shape == (2, 4, 8)


@pytest.mark.parametrize("causal", [True, False])
def test_custom_pattern_causal_switch(causal):
    x = torch.randn(1, 1, 4, 2)
    pattern = AttentionPattern("custom", x, p={"matrix": [[True] * 4 for _ in range(4)], "causal": causal})
    mask = pattern.dense()
    assert bool(mask[0, 3]) is not causal
    assert bool(mask[3, 0])


def test_custom_pattern_causal_switch_survives_compiled_graph(monkeypatch):
    outputs = []
    for causal in (False, True):
        data = project("attention", {"heads": 2, "kv_heads": 1, "causal": False, "rope": False})
        data["graph"]["nodes"].insert(1, node("pattern", "attention_pattern", "Custom", 100, 100,
            {"pattern": "custom", "matrix": [[True] * 4 for _ in range(4)], "causal": causal}))
        data["graph"]["edges"].extend([edge("x", "pattern"), edge("pattern", "op", "b")])
        torch.manual_seed(17)
        _, model, _ = exported(data, monkeypatch)
        with torch.no_grad():
            outputs.append(model(torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 13))
    assert not torch.allclose(*outputs)


@pytest.mark.parametrize("split", [True, False, None])
def test_monarch_split_export(split, monkeypatch):
    params = {} if split is None else {"split": split}
    layer, model, source = exported(project("monarch", params), monkeypatch)
    direct = runtime.make_module("monarch", params, [2, 4, 8], {"width": 8})
    torch.testing.assert_close(layer.perm0, direct.perm0)
    assert ("split=True" in source) == (split is True)
    assert model(torch.randn(2, 4, 8)).shape == (2, 4, 8)
