"""Input token domains, scoped graph variables, and executable exports."""

import copy
import io
import json
import sys
import types
import zipfile

import pytest
import torch

from backend.compiler import compile_project, export_zip
from backend.core import validate_project
from backend.templates import node, edge, templates


def tokenizer(token_id, capacity):
    return {
        "id": token_id * 20,
        "size": capacity,
        "capacity": capacity,
        "base_size": capacity,
        "class": "TestTokenizer",
        "mask_id": 2,
        "special_tokens": {},
    }


def two_inputs():
    graph = {"nodes": [], "edges": []}
    for index, (name, capacity) in enumerate((("source", 7), ("target", 11))):
        token_input = node(
            name,
            "input",
            name,
            0,
            index * 400,
            {
                "index": index,
                "shape": ["B", "S" if index == 0 else "T"],
                "dtype": "int64",
            },
        )
        token_input["data"]["tokenizer"] = tokenizer(str(index + 1), capacity)
        graph["nodes"] += [
            token_input,
            node(
                name + "_emb",
                "embedding",
                name + " embedding",
                300,
                index * 400,
                {"width": "$width", "vocab": "$vocab"},
            ),
            node(
                name + "_head",
                "head",
                name + " logits",
                600,
                index * 400,
                {"out": "$vocab", "tie": name + "_emb"},
            ),
            node(
                name + "_out",
                "output",
                name + " output",
                900,
                index * 400,
                {"index": index},
            ),
        ]
        graph["edges"] += [
            edge(name, name + "_emb"),
            edge(name + "_emb", name + "_head"),
            edge(name + "_head", name + "_out"),
        ]
    return {
        "version": 1,
        "id": "test",
        "name": "Two tokenizers",
        "config": {"width": 4, "vocab": 64},
        "graph": graph,
    }


def instantiate(project, monkeypatch):
    from backend import runtime, export_helpers, sparse_runtime

    for name, module in [
        ("runtime", runtime),
        ("export_helpers", export_helpers),
        ("sparse_runtime", sparse_runtime),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    files = compile_project(project)
    namespace = {}
    exec(files["model.py"], namespace)
    return namespace["Model"](), files


def test_input_vocabularies_are_independent_and_export_supports_backward(monkeypatch):
    project = two_inputs()
    original = copy.deepcopy(project)
    report = validate_project(project)
    assert report["valid"], report["errors"]
    assert report["shapes"]["source_head"]["shape"][-1] == 7
    assert report["shapes"]["target_head"]["shape"][-1] == 11
    assert report["parameters"] == 4 * (7 + 11)
    model, files = instantiate(project, monkeypatch)
    a, b = model(torch.tensor([[6, 1]]), torch.tensor([[10, 3, 1]]))
    assert a.shape == (1, 2, 7) and b.shape == (1, 3, 11)
    (a.square().mean() + b.square().mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert json.loads(files["project.json"]) == original
    assert project == original


def test_cross_attention_output_keeps_the_decoder_vocabulary(monkeypatch):
    project = two_inputs()
    graph = project["graph"]
    graph["nodes"] = [
        n for n in graph["nodes"] if n["id"] not in ("source_head", "source_out")
    ]
    graph["edges"] = [
        e
        for e in graph["edges"]
        if e["source"] != "source_head"
        and e["target"] not in ("source_head", "target_head")
    ]
    graph["nodes"].append(
        node(
            "cross",
            "cross_attention",
            "Cross attention",
            500,
            0,
            {"heads": 2, "kv_heads": 2},
        )
    )
    graph["edges"] += [
        edge("target_emb", "cross"),
        edge("source_emb", "cross", "b"),
        edge("cross", "target_head"),
    ]
    report = validate_project(project)
    assert report["valid"], report["errors"]
    assert report["tokenizers"]["cross"]["capacity"] == 11
    model, _ = instantiate(project, monkeypatch)
    assert model(torch.tensor([[6, 1]]), torch.tensor([[10, 3, 1]])).shape == (1, 3, 11)


def test_nested_denoising_uses_its_originating_input_tokenizer(monkeypatch):
    project = copy.deepcopy(templates()[0])
    project["config"].update(width=8, heads=2, ffn_width=16, vocab=64)
    project["graph"]["nodes"][0]["data"]["tokenizer"] = tokenizer("1", 5)
    report = validate_project(project)
    assert report["valid"], report["errors"]
    head = next(
        value for key, value in report["shapes"].items() if key.endswith("/head")
    )
    assert head["shape"][-1] == 5
    model, files = instantiate(project, monkeypatch)
    output = model(torch.tensor([[4, 1]]))
    assert output.shape == (1, 18)
    assert output.max() < 5
    assert "mask_id=2" in files["model.py"]


def test_capacity_errors_point_to_the_affected_branch_and_shared_weights():
    project = two_inputs()
    project["graph"]["nodes"][1]["data"]["params"]["vocab"] = 3
    report = validate_project(project)
    assert any(
        e["node"] == "source_emb" and "7 rows" in e["message"] for e in report["errors"]
    )
    assert report["shapes"]["target_head"]["shape"][-1] == 11
    project = two_inputs()
    for n in project["graph"]["nodes"]:
        if n["data"]["kind"] == "embedding":
            n["data"]["share"] = "embedding"
    report = validate_project(project)
    assert any("Shared module" in error["message"] for error in report["errors"])


def test_graph_variables_have_lexical_scope_and_unknown_names_are_actionable(
    monkeypatch,
):
    body = {
        "variables": {"bottleneck": 3},
        "nodes": [
            node(
                "x",
                "input",
                "Features",
                0,
                0,
                {"shape": ["B", "T", "$width"], "dtype": "float32"},
            ),
            node("down", "linear", "Down", 300, 0, {"out": "$bottleneck"}),
            node("up", "linear", "Up", 600, 0, {"out": "$width"}),
            node("out", "output", "Output", 900, 0),
        ],
        "edges": [edge("x", "down"), edge("down", "up"), edge("up", "out")],
    }
    project = {
        "version": 1,
        "name": "Variables",
        "config": {"width": 4, "bottleneck": 12},
        "graph": {
            "variables": {"width": 6},
            "nodes": [
                node(
                    "x",
                    "input",
                    "Input",
                    0,
                    0,
                    {"shape": ["B", "T", "$width"], "dtype": "float32"},
                ),
                node("body", "block", "Scoped block", 300, 0, {}, body),
                node("out", "output", "Output", 600, 0),
            ],
            "edges": [edge("x", "body"), edge("body", "out")],
        },
    }
    report = validate_project(project)
    assert report["valid"], report["errors"]
    assert report["shapes"]["body/down"]["shape"][-1] == 3
    assert report["shapes"]["body/up"]["shape"][-1] == 6
    model, _ = instantiate(project, monkeypatch)
    assert model(torch.randn(1, 2, 6)).shape == (1, 2, 6)
    del body["variables"]["bottleneck"]
    del project["config"]["bottleneck"]
    report = validate_project(project)
    assert any(
        error["node"] == "body/down"
        and "$bottleneck" in error["message"]
        and "Graph variables" in error["message"]
        for error in report["errors"]
    )


def test_numeric_overrides_work_without_project_defaults(monkeypatch):
    project = two_inputs()
    project["config"] = {}
    for n in project["graph"]["nodes"]:
        if n["data"]["kind"] == "embedding":
            n["data"]["params"]["width"] = 4
    assert validate_project(project)["valid"]
    model, _ = instantiate(project, monkeypatch)
    assert model(torch.tensor([[6]]), torch.tensor([[10]]))[1].shape == (1, 1, 11)


def test_export_includes_each_input_bundle_and_an_unambiguous_loader(
    tmp_path, monkeypatch
):
    project = two_inputs()
    directories = {}
    for index in (1, 2):
        token_id = str(index) * 20
        directory = tmp_path / token_id
        directory.mkdir()
        (directory / "tokenizer.json").write_text("{}")
        directories[token_id] = directory
    archive = zipfile.ZipFile(
        io.BytesIO(export_zip(project, tokenizer_dirs=directories))
    )
    for token_id in directories:
        assert f"tokenizers/{token_id}/tokenizer.json" in archive.namelist()
    manifest = json.loads(archive.read("tokenizers.json"))
    assert set(manifest) == {"source", "target"}
    export = tmp_path / "export"
    archive.extractall(export)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda source, **kwargs: source
            )
        ),
    )
    namespace = {"__file__": str(export / "load_tokenizer.py")}
    exec(archive.read("load_tokenizer.py"), namespace)
    with pytest.raises(ValueError, match="Choose an input node"):
        namespace["load_tokenizer"]()
    assert namespace["load_tokenizer"]("target") == str(
        export / "tokenizers" / ("2" * 20)
    )


def test_export_api_collects_bundles_from_input_nodes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import app as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    project = two_inputs()
    for token_id in ("1" * 20, "2" * 20):
        directory = tmp_path / "tokenizers" / token_id
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text("{}")
    result = TestClient(module.app).post("/api/export", json=project)
    assert result.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(result.content))
    assert "tokenizers/" + "1" * 20 + "/tokenizer.json" in archive.namelist()
    assert "tokenizers/" + "2" * 20 + "/tokenizer.json" in archive.namelist()


def test_rewrite_apis_respect_graph_scoped_variables():
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    project = copy.deepcopy(next(p for p in templates() if p["id"] == "transformer"))
    project["config"].update(width=8, heads=2, ffn_width=16, vocab=16)
    project["graph"]["variables"] = {"repeat_count": 2}
    loop = next(n for n in project["graph"]["nodes"] if n["data"]["kind"] == "loop")
    loop["data"]["params"]["iterations"] = "$repeat_count"
    response = client.post(
        "/api/graph/unroll",
        json={"project": project, "path": [], "node_id": loop["id"]},
    )
    assert response.status_code == 200, response.text
    applications = [
        n for n in response.json()["graph"]["nodes"] if n["data"].get("iteration")
    ]
    assert len(applications) == 2
    loop["data"]["body"]["variables"] = {"query_heads": 4}
    attention = next(
        n for n in loop["data"]["body"]["nodes"] if n["data"]["kind"] == "attention"
    )
    attention["data"]["params"]["heads"] = "$query_heads"
    response = client.post(
        "/api/attention/expand",
        json={"project": project, "path": [loop["id"]], "node_id": attention["id"]},
    )
    assert response.status_code == 200, response.text
    split = [n for n in response.json()["nodes"] if n["data"]["kind"] == "split_heads"]
    assert any(n["data"]["params"]["heads"] == 4 for n in split)


def test_weight_groups_reject_different_node_kinds():
    graph = {'nodes': [
        node('x', 'input', 'Features', 0, 0, {'dtype': 'float32', 'shape': ['B', 'T', 8]}),
        node('linear', 'linear', 'Linear', 300, 0, {'out': 8}, share='weights'),
        node('norm', 'norm', 'Norm', 600, 0, {}, share='weights'),
        node('out', 'output', 'Output', 900, 0)],
        'edges': [edge('x', 'linear'), edge('linear', 'norm'), edge('norm', 'out')]}
    report = validate_project({'version': 1, 'config': {'width': 8, 'vocab': 16}, 'graph': graph})
    assert not report['valid']
    assert any(error['node'] == 'norm' and 'choose a group for norm' in error['message'] for error in report['errors'])
