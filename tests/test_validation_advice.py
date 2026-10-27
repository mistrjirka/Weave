"""Actionable shape failures and conservative, nonblocking design advice."""

import copy

import pytest

from backend.compiler import compile_project
from backend.core import validate_project
from backend.templates import edge, node, templates


def chain(kinds, widths=None):
    nodes = [
        node(
            "x",
            "input",
            "Features",
            0,
            0,
            {"shape": ["B", "T", 128], "dtype": "float32"},
        )
    ]
    edges = []
    last = "x"
    for index, kind in enumerate(kinds):
        current = f"n{index}"
        settings = (
            {"out": (widths or {}).get(index, 128)}
            if kind in ("linear", "head")
            else {}
        )
        nodes.append(node(current, kind, kind, 0, 0, settings))
        edges.append(edge(last, current))
        last = current
    nodes.append(node("out", "output", "Output", 0, 0))
    edges.append(edge(last, "out"))
    return {
        "version": 1,
        "id": "advice-test",
        "name": "Advice test",
        "description": "",
        "config": {"width": 128, "vocab": 128, "heads": 8},
        "graph": {"nodes": nodes, "edges": edges},
    }


def codes(project):
    report = validate_project(project)
    assert report["valid"], report["errors"]
    return {w.get("code") for w in report["warnings"]}


@pytest.mark.parametrize("dimension", [0, -1, True, False, 1.5, "", None, "$missing"])
def test_bad_dimensions_point_to_origin_and_distinguish_downstream_errors(dimension):
    project = chain(["linear"])
    project["graph"]["nodes"][0]["data"]["params"]["shape"][1] = dimension
    report = validate_project(project)
    assert not report["valid"]
    origin = next(e for e in report["errors"] if e["node"] == "x")
    assert (
        "dimensions" in origin["message"] and "parameter bindings" in origin["message"]
    )
    downstream = next(e for e in report["errors"] if e["node"] == "n0")
    assert downstream["port"] == "a"
    assert "upstream error" in downstream["message"]
    assert report["shapes"]["x"] is None


@pytest.mark.parametrize("shape", [["B", "T", 128], [1, 5, 128], ["B", "T+16", 128]])
def test_supported_symbolic_and_fixed_dimensions_remain_valid(shape):
    project = chain(["linear"])
    project["graph"]["nodes"][0]["data"]["params"]["shape"] = shape
    assert validate_project(project)["valid"]


def test_missing_connection_and_wrong_source_are_distinct():
    project = chain(["linear"])
    project["graph"]["edges"].pop(0)
    report = validate_project(project)
    issue = next(e for e in report["errors"] if e["node"] == "n0")
    assert issue["port"] == "a" and "unconnected" in issue["message"]
    project["graph"]["edges"].append(edge("missing", "n0"))
    report = validate_project(project)
    assert not report["valid"]
    assert any("missing node" in e["message"] for e in report["errors"])


def test_residual_mismatch_has_exact_node_port_and_both_shapes():
    project = chain(["linear", "add"], {0: 64})
    project["graph"]["edges"].append(edge("x", "n1", "b"))
    report = validate_project(project)
    issue = next(e for e in report["errors"] if e["node"] == "n1")
    assert issue["port"] == "b"
    assert "64" in issue["message"] and "128" in issue["message"]


def test_matmul_rejects_integer_second_operand_before_export():
    project = chain(["matmul"])
    project["graph"]["nodes"].insert(
        1,
        node(
            "y",
            "input",
            "Integer matrix",
            0,
            0,
            {"index": 1, "shape": ["B", 128, "T"], "dtype": "int64"},
        ),
    )
    project["graph"]["edges"].append(edge("y", "n0", "b"))
    issue = next(e for e in validate_project(project)["errors"] if e["node"] == "n0")
    assert issue["port"] == "b" and "floating point" in issue["message"]
    with pytest.raises(ValueError):
        compile_project(project)


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        (["T", "S"], "bool", "query/key"),
        (["B", 3, "T", "T"], "bool", "batch/head"),
        (["T", "T"], "int64", "boolean"),
    ],
)
def test_attention_mask_mismatches_identify_mask_port(shape, dtype, message):
    project = chain(["attention"])
    project["graph"]["nodes"].insert(
        1,
        node(
            "mask", "input", "Mask", 0, 0, {"index": 1, "shape": shape, "dtype": dtype}
        ),
    )
    project["graph"]["edges"].append(edge("mask", "n0", "b"))
    report = validate_project(project)
    issue = next(e for e in report["errors"] if e["node"] == "n0")
    assert issue["port"] == "b" and message in issue["message"]


def test_extreme_internal_bottleneck_warns_but_exports():
    project = chain(["linear", "silu", "linear"], {0: 1})
    assert codes(project) == {"narrow-bottleneck"}
    assert "model.py" in compile_project(project)
    warning = validate_project(project)["warnings"][0]
    assert warning["node"] == "n0"
    assert "128 features to 1" in warning["message"]
    assert warning["suggestion"] and warning["source"].startswith("https://")


@pytest.mark.parametrize("width", [8, 16, 64, 128])
def test_ordinary_hidden_widths_do_not_trigger_extreme_bottleneck(width):
    assert "narrow-bottleneck" not in codes(
        chain(["linear", "silu", "linear"], {0: width})
    )


def test_scalar_output_and_residual_merge_are_not_mislabelled_as_global_bottlenecks():
    assert "narrow-bottleneck" not in codes(chain(["linear"], {0: 1}))
    project = chain(["linear", "concat", "linear"], {0: 1})
    project["graph"]["edges"].append(edge("x", "n1", "b"))
    assert "narrow-bottleneck" not in codes(project)


def test_ffn_extreme_internal_width_uses_resolved_nested_settings():
    project = chain(["ffn"])
    project["graph"]["nodes"][1]["data"]["params"]["hidden"] = "$tiny"
    project["config"]["tiny"] = 2
    assert "narrow-bottleneck" in codes(project)
    body = project["graph"]
    project["graph"] = {
        "nodes": [
            copy.deepcopy(body["nodes"][0]),
            node("body", "block", "Nested", 0, 0, {}, body),
            node("out", "output", "Out", 0, 0),
        ],
        "edges": [edge("x", "body"), edge("body", "out")],
    }
    assert validate_project(project)["warnings"][0]["node"] == "body/n0"


def test_affine_chain_has_one_explanatory_note_and_activations_stop_it():
    project = chain(["linear"] * 4)
    hints = [
        w
        for w in validate_project(project)["warnings"]
        if w.get("code") == "consecutive-linear"
    ]
    assert len(hints) == 1
    assert "factorized" in hints[0]["suggestion"]
    assert "consecutive-linear" not in codes(chain(["linear", "silu", "linear"]))


def test_softmax_training_advice_is_conditional_and_inference_still_exports():
    project = chain(["head", "softmax"])
    assert "output-softmax" in codes(project)
    assert "model.py" in compile_project(project)
    assert "output-softmax" not in codes(chain(["softmax", "linear"]))


def test_unused_learned_branch_is_advisory_and_live_auxiliary_outputs_count():
    project = chain(["linear"])
    project["graph"]["nodes"].append(
        node("unused", "linear", "Draft", 0, 0, {"out": 128})
    )
    project["graph"]["edges"].append(edge("x", "unused"))
    assert "unused-branch" in codes(project)
    project["graph"]["nodes"].append(
        node("aux", "output", "Auxiliary", 0, 0, {"index": 1})
    )
    project["graph"]["edges"].append(edge("unused", "aux"))
    assert "unused-branch" not in codes(project)


def test_diffusion_gets_sampling_advice_without_rejecting_bidirectional_attention():
    for model_id in ["combined", "diffusion"]:
        project = next(p for p in templates() if p["id"] == model_id)
        assert "discrete-sampling" in codes(project)
        assert not any(
            "causal" in str(w) for w in validate_project(project)["warnings"]
        )


def test_malformed_nested_graph_does_not_crash_advice():
    body = chain(["linear"])["graph"]
    body["edges"].append(edge("missing", "n0"))
    project = chain(["block"])
    project["graph"]["nodes"][1]["data"]["body"] = body
    assert not validate_project(project)["valid"]


@pytest.mark.parametrize(
    ("factor", "warns"),
    [(0, True), (0.0, True), (0.001, False), (-1, False), (1, False)],
)
def test_zero_scale_warns_only_for_exact_zero_and_never_blocks_export(factor, warns):
    project = chain(["scale"])
    project["graph"]["nodes"][1]["data"]["params"]["factor"] = factor
    assert ("zero-scale" in codes(project)) == warns
    assert "model.py" in compile_project(project)


@pytest.mark.parametrize(
    ("matrix", "warns"),
    [
        ([[True, False], [False, True]], False),
        ([[True, False], [False, False]], True),
        ([[False, False], [False, False]], True),
    ],
)
def test_empty_custom_attention_rows_explain_mask_semantics_and_composition(
    matrix, warns
):
    project = chain(["attention_pattern"])
    project["graph"]["nodes"][0]["data"]["params"]["shape"] = ["B", 2, 128]
    project["graph"]["nodes"][1]["data"]["params"] = {
        "pattern": "custom",
        "matrix": matrix,
    }
    assert ("empty-attention-rows" in codes(project)) == warns
    if warns:
        hint = next(
            w
            for w in validate_project(project)["warnings"]
            if w["code"] == "empty-attention-rows"
        )
        assert "zero-based" in hint["message"]
        assert "union" in hint["suggestion"]
    assert "model.py" in compile_project(project)
