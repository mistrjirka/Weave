"""Explicit lane ports preserve values, gradients and source-port contracts."""

import copy

import pytest
import torch

from backend.core import validate_project
from backend.connections import connection_options
from backend.templates import node, edge, templates
from test_codegen import instantiate


def document(nodes, edges, **variables):
    return {
        "version": 1,
        "id": "lanes-test",
        "name": "Lanes",
        "config": {"width": 8, "vocab": 32, "heads": 2, "ffn_width": 16},
        "graph": {"nodes": nodes, "edges": edges, "variables": variables},
    }


def wire(source, target, lane, port="a"):
    return {
        **edge(source, target, port),
        "id": f"{source}-{lane}-{target}-{port}",
        "sourceHandle": f"lane{lane}",
    }


def features(ident="x", shape=None, dtype="float32"):
    return node(
        ident,
        "input",
        ident,
        0,
        0,
        {"index": 0, "shape": shape or ["B", "T", 8], "dtype": dtype},
    )


@pytest.mark.parametrize("mode", ["stack", "mean", "sum"])
def test_fork_join_matches_torch_forward_and_gradients(mode, monkeypatch):
    p = document(
        [
            features(),
            node("fork", "lane_fork", "Fork", 0, 0, {"lanes": "$lanes"}),
            node("join", "lane_join", "Join", 0, 0, {"lanes": "$lanes", "mode": mode}),
            node("out", "output", "Out", 0, 0),
        ],
        [
            edge("x", "fork"),
            *[wire("fork", "join", i, f"lane{i}") for i in range(3)],
            edge("join", "out"),
        ],
        lanes=3,
    )
    report = validate_project(p)
    assert report["valid"], report["errors"]
    assert list(report["port_shapes"]["fork"]) == ["lane0", "lane1", "lane2"]
    assert report["port_shapes"]["fork"]["lane2"]["shape"] == ["B", "T", 8]
    model, files = instantiate(p, monkeypatch)
    assert not list(model.parameters())
    x = torch.randn(2, 5, 8, requires_grad=True)
    actual = model(x)
    expected = torch.stack([x, x, x], dim=-2)
    if mode == "mean":
        expected = expected.mean(-2)
    if mode == "sum":
        expected = expected.sum(-2)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.square().sum().backward()
    torch.testing.assert_close(grad, x.grad)
    assert "torch.stack" in files["model.py"]


def test_split_selects_each_correct_lane_and_accepts_independent_layers(monkeypatch):
    p = document(
        [
            features(shape=["B", "T", 3, 8]),
            node("split", "lane_split", "Split", 0, 0, {"lanes": 3}),
            node("projection", "linear", "Projection", 0, 0, {"out": 4, "bias": False}),
            node("out", "output", "Out", 0, 0),
        ],
        [edge("x", "split"), wire("split", "projection", 2), edge("projection", "out")],
    )
    report = validate_project(p)
    assert report["valid"], report["errors"]
    assert report["shapes"]["projection"]["shape"] == ["B", "T", 4]
    model, files = instantiate(p, monkeypatch)
    x = torch.randn(2, 5, 3, 8, requires_grad=True)
    actual = model(x)
    torch.testing.assert_close(actual, model.projection(x[:, :, 2]))
    actual.sum().backward()
    assert torch.count_nonzero(x.grad[:, :, :2]) == 0
    assert torch.count_nonzero(x.grad[:, :, 2]) > 0
    assert "torch.unbind" in files["model.py"]
    options = connection_options(
        p,
        [],
        "split",
        "lane2",
        [{"id": "linear", "data": {"kind": "linear", "params": {"out": 4}}}],
    )
    assert options["source"]["shape"] == ["B", "T", 8]
    assert options["matches"] == [{"id": "linear", "ports": ["a"]}]


def test_join_reports_specific_mismatching_and_missing_ports():
    y = features("y", shape=["B", "T", 4])
    y["data"]["params"]["index"] = 1
    p = document(
        [
            features(),
            y,
            node("join", "lane_join", "Join", 0, 0, {"lanes": 3}),
            node("out", "output", "Out", 0, 0),
        ],
        [edge("x", "join", "lane0"), edge("y", "join", "lane1"), edge("join", "out")],
    )
    report = validate_project(p)
    assert not report["valid"]
    assert any(
        e["node"] == "join"
        and e.get("port") == "lane1"
        and "matching shape" in e["message"]
        for e in report["errors"]
    )
    y["data"]["params"]["shape"] = ["B", "T", 8]
    report = validate_project(p)
    assert any(
        e["node"] == "join"
        and e.get("port") == "lane2"
        and "unconnected" in e["message"]
        for e in report["errors"]
    )


@pytest.mark.parametrize("count", [0, -1, 33, True, 2.5])
def test_invalid_counts_are_node_errors(count):
    p = document(
        [
            features(),
            node("fork", "lane_fork", "Fork", 0, 0, {"lanes": count}),
            node("out", "output", "Out", 0, 0),
        ],
        [edge("x", "fork"), wire("fork", "out", 0)],
    )
    report = validate_project(p)
    assert any(
        e["node"] == "fork" and "Lane count" in e["message"] for e in report["errors"]
    )


def test_unknown_split_handle_and_wrong_lane_axis_are_rejected():
    p = document(
        [
            features(shape=["B", "T", 3, 8]),
            node("split", "lane_split", "Split", 0, 0, {"lanes": 3}),
            node("out", "output", "Out", 0, 0),
        ],
        [edge("x", "split"), edge("split", "out")],
    )
    assert any(
        "unknown source port" in e["message"] for e in validate_project(p)["errors"]
    )
    p["graph"]["edges"][-1]["sourceHandle"] = "lane0"
    p["graph"]["nodes"][1]["data"]["params"]["lanes"] = 4
    assert any(
        "second-to-last dimension" in e["message"]
        for e in validate_project(p)["errors"]
    )


def test_lane_example_exports_and_learned_branches_receive_gradients(monkeypatch):
    p = copy.deepcopy(next(p for p in templates() if p["id"] == "residual-lanes"))
    p["config"]["width"] = 8
    model, _ = instantiate(p, monkeypatch)
    x = torch.randn(2, 5, 8)
    actual = model(x)
    expected = torch.stack(
        [getattr(model, f"independent_branch_{i+1}")(x) for i in range(4)], -2
    ).mean(-2)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert len(list(model.parameters())) == 4
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_lane_output_picker_can_create_a_graph_output():
    p = document(
        [
            features(),
            node("fork", "lane_fork", "Fork", 0, 0, {"lanes": 2}),
            node("out", "output", "Out", 0, 0),
        ],
        [edge("x", "fork"), wire("fork", "out", 0)],
    )
    options = connection_options(
        p,
        [],
        "fork",
        "lane1",
        [{"id": "output", "data": {"kind": "output", "params": {}}}],
    )
    assert options["matches"] == [{"id": "output", "ports": ["a"]}]
