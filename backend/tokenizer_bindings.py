"""Resolve input-owned token vocabularies without changing the saved graph.

Tokenizer provenance follows the primary tensor stream. Container inputs inherit
it through their ports; token selection returns the ID stream at port b. Keeping
this separate from shape contracts avoids treating encoder memory as decoder IDs.
"""

from __future__ import annotations

import copy


def input_tokenizers(project):
    """All explicitly configured inputs, keyed by their scoped node path."""
    result = {}

    def visit(graph, prefix=""):
        for node in graph.get("nodes", []):
            data = node["data"]
            if data.get("kind") == "input" and data.get("tokenizer"):
                result[prefix + node["id"]] = data["tokenizer"]
            if data.get("body"):
                visit(data["body"], prefix + node["id"] + "/")

    visit(project.get("graph", {}))
    return result


def bind_input_tokenizers(project, ordered):
    """Return a compilation copy, token domains by path, and input errors."""
    bound = copy.deepcopy(project)
    domains, errors = {}, []
    legacy = bound.get("tokenizer")

    def substitute(value, variables):
        if isinstance(value, str):
            return variables.get(value, value)
        if isinstance(value, list):
            return [substitute(item, variables) for item in value]
        if isinstance(value, dict):
            return {key: substitute(item, variables) for key, item in value.items()}
        return value

    def scope(graph, config, inputs=None, prefix=""):
        config = {**config, **graph.get("variables", {})}
        local, outputs = {}, []
        try:
            sequence = ordered(graph)
        except (ValueError, KeyError):
            return []  # Structural errors belong to the shape validator.
        for node in sequence:
            data = node["data"]
            kind = data["kind"]
            raw = data.get("params", {})
            key = prefix + node["id"]
            ports = {
                edge.get("targetHandle") or "a": local.get(edge["source"])
                for edge in graph.get("edges", [])
                if edge["target"] == node["id"]
            }
            domain = ports.get("b" if kind == "select" else "a")
            if kind == "input":
                index = raw.get("index", 0)
                inherited = inputs.get(index) if inputs is not None else None
                own = data.get("tokenizer")
                shape = raw.get("shape", ["B", "T"])
                eligible = (
                    raw.get("dtype", "int64") == "int64"
                    and isinstance(shape, list)
                    and len(shape) == 2
                )
                if own and not eligible:
                    errors.append(
                        {
                            "node": key,
                            "message": "A tokenizer belongs to an int64 token input with shape [B,T]. Remove it or change this input contract.",
                        }
                    )
                domain = (
                    inherited
                    or own
                    or (legacy if inputs is None and eligible else None)
                )
                if own and inherited and own.get("id") != inherited.get("id"):
                    errors.append(
                        {
                            "node": key,
                            "message": "This block input already receives token IDs from a different tokenizer. Configure the originating input instead.",
                        }
                    )
            if domain:
                capacity = domain.get("capacity")
                if type(capacity) is not int or capacity < 1:
                    errors.append(
                        {
                            "node": key,
                            "message": "Tokenizer capacity must be a positive integer. Reload the tokenizer on its input.",
                        }
                    )
                    domain = None
            if domain:
                variables = {"$vocab": domain["capacity"]}
                if domain.get("mask_id") is not None:
                    variables["$mask_id"] = domain["mask_id"]
                data["params"] = raw = substitute(raw, variables)
                defaults = {
                    "embedding": {"vocab": domain["capacity"]},
                    "head": {"out": domain["capacity"]},
                    "mask": {"mask_id": domain.get("mask_id", config.get("mask_id"))},
                    "select": {"mask_id": domain.get("mask_id", config.get("mask_id"))},
                }
                for name, value in defaults.get(kind, {}).items():
                    if value is not None:
                        raw.setdefault(name, value)
            if data.get("body"):
                resolved = {
                    name: (
                        config.get(value[1:])
                        if isinstance(value, str) and value.startswith("$")
                        else value
                    )
                    for name, value in raw.items()
                }
                child_inputs = {
                    child["data"]
                    .get("params", {})
                    .get("index", 0): ports.get(
                        chr(97 + child["data"].get("params", {}).get("index", 0))
                    )
                    for child in data["body"].get("nodes", [])
                    if child["data"].get("kind") == "input"
                }
                child_outputs = scope(
                    data["body"], {**config, **resolved}, child_inputs, key + "/"
                )
                domain = child_outputs[0] if child_outputs else domain
            if domain:
                domains[key] = domain
            local[node["id"]] = domain
            if kind == "output":
                outputs.append((raw.get("index", 0), domain))
        return [value for _, value in sorted(outputs, key=lambda item: item[0])]

    scope(bound.get("graph", {}), bound.get("config", {}))
    return bound, domains, errors
