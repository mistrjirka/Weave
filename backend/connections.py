"""Find input ports that accept a source tensor using the graph validator.

Single-input blocks are checked directly. For incomplete multi-input blocks we
supply example contracts for their remaining ports, then require full validation.
These witnesses are never inserted into the user's graph. Custom/container input
contracts are used where available; a small registry supplies linked witnesses
for operations whose other inputs necessarily differ (lanes and expert routing).
No Python from custom blocks is executed.
"""

from copy import deepcopy
import json

from .lanes import source_contract
from .core import (
    graph_context,
    optional_ports,
    output_port_names,
    params,
    port_names,
    resolve,
    validate_project,
)


def tensor(shape, dtype="float32"):
    return {"shape": list(shape), "dtype": dtype}


def witness_pool(source, config, settings, existing):
    """A bounded pool of example tensors; acceptance is decided by validation."""
    shape = source["shape"]
    batch = shape[0] if len(shape) >= 3 else "B"
    sequence = shape[-2] if len(shape) >= 3 else shape[-1] if shape else "T"
    if source["dtype"] in ("bool", "attention_pattern") and len(shape) >= 2:
        sequence = shape[-2]
    if source["dtype"] == "int64" and len(shape) == 2:
        batch, sequence = shape
    heads = settings.get("heads", config.get("heads", 8))
    heads = heads if type(heads) is int and heads > 0 else 8
    lanes = settings.get("lanes", 4)
    lanes = lanes if type(lanes) is int and lanes > 0 else 4
    widths = [
        config.get("width", 128),
        config.get("vocab", 256),
        settings.get("width"),
        settings.get("out"),
        lanes * (lanes + 2),
    ]
    if shape and type(shape[-1]) is int:
        widths += [shape[-1], shape[-1] * 2]
        if shape[-1] % 2 == 0:
            widths.append(shape[-1] // 2)
    result = [source, *existing, tensor([batch, sequence], "int64")]
    if len(shape) >= 2:
        result.append(tensor([*shape[:-2], shape[-1], shape[-2]]))
    for width in dict.fromkeys(
        value for value in widths if type(value) is int and value > 0
    ):
        result += [
            tensor([batch, sequence, width]),
            tensor([batch, sequence, lanes, width]),
        ]
        if width % heads == 0:
            result.append(tensor([batch, heads, sequence, width // heads]))
    for mask in (
        [sequence, sequence],
        [batch, sequence, sequence],
        [batch, 1, sequence, sequence],
    ):
        result += [tensor(mask, "bool"), tensor(mask, "attention_pattern")]
    unique = {}
    for contract in result:
        if contract:
            unique.setdefault(json.dumps(contract, sort_keys=True), contract)
    return list(unique.values())[:96]


def declared_inputs(data, config):
    if data.get("body"):
        scope = {
            **config,
            **resolve(data.get("params", {}), config),
            **data["body"].get("variables", {}),
        }
        inputs = sorted(
            (node for node in data["body"]["nodes"] if node["data"]["kind"] == "input"),
            key=lambda node: node["data"]["params"].get("index", 0),
        )
        return {
            chr(97 + index): tensor(
                resolve(node["data"]["params"].get("shape", ["B", "T"]), scope),
                node["data"]["params"].get("dtype", "int64"),
            )
            for index, node in enumerate(inputs)
        }
    if data["kind"] == "custom":
        settings = resolve(data.get("params", {}), config)
        shapes = settings.get("input_shapes") or [
            settings.get("input_shape", ["B", "T", config.get("width", 128)])
        ]
        dtypes = settings.get("input_dtypes") or ["float32"] * len(shapes)
        return {
            chr(97 + index): tensor(shape, dtypes[index])
            for index, shape in enumerate(shapes)
            if index < len(dtypes)
        }
    return {}


def lane_witness(source, port, settings, config):
    shape = source["shape"]
    if len(shape) < 2:
        return {}
    lanes = settings.get("lanes", 4)
    width = (
        shape[-1]
        if port in ("a", "b") and len(shape) >= 3
        else config.get("width", 128)
    )
    batch, sequence = shape[:2]
    return {
        "a": tensor([batch, sequence, lanes, width]),
        "b": tensor([batch, sequence, width]),
        "c": tensor([batch, sequence, lanes * (lanes + 2)]),
    }


def expert_witness(source, port, settings, config):
    shape = source["shape"]
    if len(shape) != 3:
        return {}
    count = settings.get("experts", 0)
    if type(count) is not int or not 1 <= count <= 256:
        return {}
    width = shape[-1] if port != "a" else config.get("width", 128)
    return {
        "a": tensor([*shape[:2], count]),
        **{f"e{index}": tensor([*shape[:2], width]) for index in range(count)},
    }


LINKED_WITNESSES = {"lane_merge": lane_witness, "moe_combine": expert_witness}


def probe_valid(data, contracts, config, moe_body):
    node = {"id": "candidate", "data": data}
    inputs = list(contracts)
    graph = {
        "nodes": [
            {
                "id": f"input-{index}",
                "data": {"kind": "input", "params": {"index": index}},
            }
            for index in range(len(inputs))
        ]
        + [
            node,
            {
                "id": "result",
                "data": {
                    "kind": "output",
                    "params": {"index": 1 if data["kind"] == "output" else 0},
                },
            },
        ],
        "edges": [
            {"source": f"input-{index}", "target": "candidate", "targetHandle": port}
            for index, port in enumerate(inputs)
        ]
        + [{"source": "candidate", "target": "result", "targetHandle": "a"}],
    }
    if data["kind"] == "output":
        # Output terminates the graph and has no source socket of its own.
        graph["nodes"].pop()
        graph["edges"].pop()
    else:
        outputs = output_port_names(node, config)
        if outputs:
            graph["edges"][-1]["sourceHandle"] = outputs[0]
    report = validate_project(
        {"version": 1, "config": config, "graph": graph},
        input_contracts=[contracts[port] for port in inputs],
        moe_body=moe_body,
        include_advice=False,
    )
    return report["valid"]


def compatible_ports(data, source, config, existing, moe_body=False):
    node = {"id": "candidate", "data": data}
    ports = port_names(node, config)
    optional = set(optional_ports(node))
    settings = params(node, config)
    declared = declared_inputs(data, config)
    pool = witness_pool(source, config, settings, existing)
    matches = []
    for port in ports:
        required = [other for other in ports if other != port and other not in optional]
        linked = LINKED_WITNESSES.get(data["kind"])
        witnesses = [linked(source, port, settings, config)] if linked else []
        if declared:
            witnesses.append(declared)
        if not required:
            witnesses.append({})
        else:
            for example in pool:
                assignment = {other: example for other in required}
                # A rectangular mask relates Q to one length and K/V to another.
                if (
                    port in optional
                    and source["dtype"] in ("bool", "attention_pattern")
                    and len(source["shape"]) >= 2
                ):
                    assignment = {other: deepcopy(example) for other in required}
                    for index, other in enumerate(required):
                        shape = assignment[other]["shape"]
                        if len(shape) >= 2:
                            shape[-2] = source["shape"][-2 if index == 0 else -1]
                witnesses.append(assignment)
        for witness in witnesses:
            contracts = {
                other: witness[other] for other in required if other in witness
            }
            contracts[port] = source
            if len(contracts) != len(required) + 1:
                continue
            if probe_valid(data, contracts, config, moe_body):
                matches.append(port)
                break
    return matches


def connection_options(project, path, source_id, handle, candidates):
    graph, config = graph_context(project, path)
    if not any(node["id"] == source_id for node in graph["nodes"]):
        raise ValueError("The source node is no longer available.")
    report = validate_project(project, include_advice=False)
    prefix = "/".join([*path, ""])
    key = prefix + source_id if path else source_id
    source = source_contract(report, key, handle)
    if source is None:
        return {
            "source": None,
            "matches": [],
            "message": "The source tensor shape is unresolved. Connect its inputs or fix the upstream graph first.",
        }
    domain = report.get("tokenizers", {}).get(key)
    if domain:
        config = {
            **config,
            "vocab": domain["capacity"],
            **(
                {"mask_id": domain["mask_id"]}
                if domain.get("mask_id") is not None
                else {}
            ),
        }
    existing = [
        contract
        for node_key, contract in report["shapes"].items()
        if contract and node_key.rpartition("/")[0] == key.rpartition("/")[0]
    ][:32]
    context_moe = False
    if path:
        parent, _ = graph_context(project, path[:-1])
        context_moe = (
            next(node for node in parent["nodes"] if node["id"] == path[-1])["data"][
                "kind"
            ]
            == "moe"
        )
    matches = []
    for candidate in candidates:
        try:
            ports = compatible_ports(
                candidate["data"], source, config, existing, context_moe
            )
        except (ValueError, TypeError, KeyError, IndexError, SyntaxError):
            # One malformed saved preset must not hide the other valid choices.
            ports = []
        if ports:
            matches.append({"id": candidate["id"], "ports": ports})
    return {"source": source, "matches": matches, "message": None}
