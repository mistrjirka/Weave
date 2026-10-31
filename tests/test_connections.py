"""Connected-node suggestions use the same shape rules as graph validation."""

from copy import deepcopy

import pytest

from backend.connections import compatible_ports, connection_options
from backend.templates import node, edge, transformer

CFG = {"width": 128, "heads": 8, "ffn_width": 256, "vocab": 1024, "mask_id": 0}


def data(kind, **settings):
    return {"kind": kind, "label": kind, "params": settings}


def contract(shape, dtype="float32", **metadata):
    return {"shape": shape, "dtype": dtype, **metadata}


def ports(kind, source, **settings):
    return compatible_ports(data(kind, **settings), source, CFG, [])


def project(source):
    return {
        "version": 1,
        "config": CFG,
        "graph": {
            "nodes": [
                node("source", "input", "Source", 0, 0, {"index": 0, **source}),
                node("out", "output", "Output", 200, 0),
            ],
            "edges": [edge("source", "out")],
        },
    }


def test_token_ids_float_features_and_head_ranks_are_distinguished():
    ids = contract(["B", "T"], "int64")
    features = contract(["B", "T", 128])
    assert ports("embedding", ids, width=128, vocab=1024) == ["a"]
    assert ports("embedding", features, width=128, vocab=1024) == []
    assert ports("attention", ids, heads=8) == []
    assert ports("attention", features, heads=8) == ["a"]
    assert ports("merge_heads", features) == []
    assert ports("merge_heads", contract(["B", 8, "T", 16])) == ["a"]
    assert ports("split_heads", contract(["B", "T", 130]), heads=8) == []


def test_attention_masks_match_their_optional_ports_and_rectangular_lengths():
    square = contract(["T", "T"], "bool")
    rectangular = contract(["T", "S"], "attention_pattern")
    assert ports("attention", square, heads=8) == ["b"]
    assert ports("cross_attention", square, heads=8) == ["c"]
    assert ports("sdpa", square) == ["d"]
    assert ports("attention", rectangular, heads=8) == []
    assert ports("cross_attention", rectangular, heads=8) == ["c"]
    assert ports("sdpa", rectangular) == ["d"]


def test_partial_multi_input_blocks_are_not_hidden_for_missing_future_inputs():
    features = contract(["B", "T", 128])
    assert ports("add", features) == ["a", "b"]
    assert ports("matmul", features) == ["a", "b"]
    assert ports("cross_attention", features, heads=8) == ["a", "b"]
    assert ports("select", contract(["B", "T"], "int64"), mask_id=0) == ["b"]
    assert ports("memory_fuse", features) == ["a", "b"]
    assert ports("lane_merge", contract(["B", "T", 24]), lanes=4) == ["b", "c"]


def test_custom_declared_ports_and_symbolic_variables_are_used_without_executing_code(
    tmp_path,
):
    marker = tmp_path / "must-not-exist"
    custom = data(
        "custom",
        input_shapes=[["B", "T"], ["B", "T", "$width"]],
        input_dtypes=["int64", "float32"],
        output_shape=["B", "T", "$width"],
        source=f"from pathlib import Path\nPath({str(marker)!r}).touch()\nclass CustomBlock: pass\n",
    )
    assert compatible_ports(custom, contract(["B", "T"], "int64"), CFG, []) == ["a"]
    assert compatible_ports(custom, contract(["B", "T", 128]), CFG, []) == ["b"]
    assert compatible_ports(custom, contract(["B", "T", 64]), CFG, []) == []
    assert not marker.exists()


def test_nested_container_contracts_and_loop_state_preservation():
    block = {**data("block"), "body": transformer(kv=2)}
    assert compatible_ports(block, contract(["B", "T", 128]), CFG, []) == ["a"]
    assert compatible_ports(block, contract(["B", "T"], "int64"), CFG, []) == []
    body = {
        "nodes": [
            node(
                "x",
                "input",
                "Input",
                0,
                0,
                {"index": 0, "shape": ["B", "T", "$width"], "dtype": "float32"},
            ),
            node("projection", "linear", "Projection", 100, 0, {"out": 64}),
            node("out", "output", "Output", 200, 0),
        ],
        "edges": [edge("x", "projection"), edge("projection", "out")],
    }
    assert compatible_ports(
        {**data("block"), "body": body}, contract(["B", "T", 128]), CFG, []
    ) == ["a"]
    assert (
        compatible_ports(
            {**data("loop", iterations=3), "body": body},
            contract(["B", "T", 128]),
            CFG,
            [],
        )
        == []
    )


def test_metadata_is_preserved_for_hash_lookup_compatibility():
    source = contract(["B", "T", 4], "int64", orders=[2, 3], hash_heads=2, buckets=512)
    assert ports(
        "engram_lookup", source, orders=[2, 3], hash_heads=2, buckets=512, width=128
    ) == ["a"]
    assert (
        ports(
            "engram_lookup",
            source,
            orders=[2, 3],
            hash_heads=2,
            buckets=1024,
            width=128,
        )
        == []
    )


def test_moe_operations_are_available_only_in_their_own_container_context():
    router = data("moe_router", experts=4)
    source = contract(["B", "T", 128])
    assert compatible_ports(router, source, CFG, [], moe_body=False) == []
    assert compatible_ports(router, source, CFG, [], moe_body=True) == ["a"]
    combine = data("moe_combine", experts=4)
    assert compatible_ports(combine, source, CFG, [], moe_body=True) == [
        "e0",
        "e1",
        "e2",
        "e3",
    ]


def test_batch_uses_current_scope_ignores_unrelated_errors_and_never_mutates_project():
    document = project(contract(["B", "T"], "int64"))
    document["graph"]["nodes"].append(
        node("unfinished", "linear", "Unfinished", 0, 300)
    )
    original = deepcopy(document)
    candidates = [
        {"id": "embedding", "data": data("embedding", width="$width", vocab="$vocab")},
        {"id": "attention", "data": data("attention", heads=8)},
        {"id": "malformed", "data": data("custom", input_shapes=[128])},
    ]
    result = connection_options(document, [], "source", "out", candidates)
    assert result["source"] == contract(["B", "T"], "int64")
    assert result["matches"] == [{"id": "embedding", "ports": ["a"]}]
    assert document == original
    document["graph"]["nodes"].append(node("broken", "linear", "Broken", 0, 500))
    result = connection_options(document, [], "broken", "out", candidates)
    assert result["source"] is None and result["matches"] == []
    assert "upstream" in result["message"]
    with pytest.raises(ValueError, match="source node"):
        connection_options(document, [], "gone", "out", candidates)


def test_nested_graph_source_uses_lexical_variables_and_outer_actual_contract():
    inner = {
        "nodes": [
            node(
                "x",
                "input",
                "Input",
                0,
                0,
                {"index": 0, "shape": ["B", "T", "$width"], "dtype": "float32"},
            ),
            node("out", "output", "Output", 200, 0),
        ],
        "edges": [edge("x", "out")],
        "variables": {"width": 64},
    }
    document = project(contract(["B", "T", 128]))
    document["graph"]["nodes"].append(
        node("container", "block", "Container", 100, 0, {}, inner)
    )
    document["graph"]["edges"].append(edge("source", "container"))
    custom = data(
        "custom", input_shape=["B", "T", "$width"], source="class CustomBlock: pass"
    )
    result = connection_options(
        document,
        ["container"],
        "x",
        "out",
        [
            {"id": "custom", "data": custom},
            {"id": "linear", "data": data("linear", out=64)},
        ],
    )
    assert result["source"]["shape"][-1] == 128
    assert result["matches"] == [{"id": "linear", "ports": ["a"]}]
