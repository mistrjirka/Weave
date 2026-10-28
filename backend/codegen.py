"""Readable Python source generation for a validated graph.

Operation emitters describe constructors or tensor expressions. Graph traversal,
parameter identity, scheduling and Python naming are handled independently.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import json
import keyword
import re
import unicodedata
from collections.abc import Callable

from .core import definition_signature, ordered, params, port_names, moe_structure

MODULE_NAMES = {
    "forward",
    "training",
    "train",
    "eval",
    "parameters",
    "named_parameters",
    "buffers",
    "named_buffers",
    "children",
    "named_children",
    "modules",
    "named_modules",
    "state_dict",
    "load_state_dict",
    "to",
    "cpu",
    "cuda",
    "float",
    "double",
    "half",
    "bfloat16",
    "apply",
    "requires_grad_",
    "zero_grad",
    "get_parameter",
    "get_submodule",
    "register_parameter",
    "register_buffer",
    "add_module",
    "register_module",
    "compile",
    "type",
    "share_memory",
    "extra_repr",
    "set_extra_state",
    "get_extra_state",
}
RESERVED = (
    set(keyword.kwlist)
    | set(dir(builtins))
    | MODULE_NAMES
    | {
        "self",
        "torch",
        "nn",
        "F",
        "step",
        "steps",
        "shared_modules",
    }
)


def identifier(label, fallback="tensor"):
    text = unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:64]
    if not text:
        text = fallback
    if text[0].isdigit():
        text = "layer_" + text
    if text in RESERVED:
        text += "_tensor"
    return text


class Names:
    def __init__(self, reserved=()):
        self.used = set(reserved)

    def take(self, base):
        name, index = base, 2
        while name in self.used:
            name = f"{base}_{index}"
            index += 1
        self.used.add(name)
        return name


@dataclass(frozen=True)
class Operation:
    constructor: Callable | None = None
    expression: Callable | None = None
    runtime: tuple[str, ...] = ()
    helpers: tuple[str, ...] = ()
    sparse: tuple[str, ...] = ()


# Builders receive resolved settings, inferred input dimensions and local names.
OPERATIONS = {
    "embedding": Operation(
        constructor=lambda p, c, s: f"nn.Embedding(num_embeddings={p.get('vocab', c.get('vocab'))!r}, embedding_dim={p.get('width', c.get('width', 128))!r})"
    ),
    "linear": Operation(
        constructor=lambda p, c, s: f"nn.Linear(in_features={s[-1]!r}, out_features={p.get('out', c.get('width', 128))!r}, bias={p.get('bias', False)!r})"
    ),
    "head": Operation(
        constructor=lambda p, c, s: f"nn.Linear(in_features={s[-1]!r}, out_features={p.get('out', c.get('vocab'))!r}, bias={p.get('bias', False)!r})"
    ),
    "norm": Operation(
        constructor=lambda p, c, s: f"nn.RMSNorm({s[-1]!r}, eps={p.get('eps', 1e-6)!r})"
    ),
    "ffn": Operation(
        constructor=lambda p, c, s: f"FFN(width={s[-1]!r}, hidden={p.get('hidden', s[-1]*4)!r})",
        runtime=("FFN",),
    ),
    "ngram": Operation(
        constructor=lambda p, c, s: f"NGram(orders={p.get('orders', [2,3])!r}, buckets={p.get('buckets', 8192)!r})",
        runtime=("NGram",),
    ),
    "lookup": Operation(
        constructor=lambda p, c, s: f"Lookup(orders={s[-1]!r}, buckets={p.get('buckets', 8192)!r}, width={p.get('width', c.get('width', 128))!r})",
        runtime=("Lookup",),
    ),
    "gate": Operation(constructor=lambda p, c, s: "Gate()", runtime=("Gate",)),
    "hadamard": Operation(
        constructor=lambda p, c, s: f"Hadamard({s[-1]!r})", runtime=("Hadamard",)
    ),
    "monarch": Operation(
        constructor=lambda p, c, s: f"Monarch(width={s[-1]!r}, split={p.get('split', False)!r})",
        runtime=("Monarch",),
    ),
    "causal_conv": Operation(
        constructor=lambda p, c, s: f"CausalConv(width={s[-1]!r}, taps={p.get('taps', 3)!r})",
        runtime=("CausalConv",),
    ),
    "routing": Operation(
        constructor=lambda p, c, s: f"Routing(width={s[-1]!r}, lanes={p.get('lanes', 4)!r}, layer={p.get('layer', 0)!r})",
        runtime=("Routing",),
    ),
    "engram_hash": Operation(
        constructor=lambda p, c, s: f"EngramHash(orders={p.get('orders', [2,3])!r}, heads={p.get('hash_heads', 2)!r}, slots={p.get('buckets', 8192)!r})",
        runtime=("EngramHash",),
    ),
    "engram_lookup": Operation(
        constructor=lambda p, c, s: f"EngramLookup(width={p.get('width', c.get('width', 128))!r}, orders={p.get('orders', [2,3])!r}, heads={p.get('hash_heads', 2)!r}, slots={p.get('buckets', 8192)!r}, sub_dim={p.get('sub_dim', 128)!r}, dilation={p.get('dilation', 3)!r})",
        runtime=("EngramLookup",),
    ),
    "add": Operation(expression=lambda a, p, c: f"{a[0]} + {a[1]}"),
    "subtract": Operation(expression=lambda a, p, c: f"{a[0]} - {a[1]}"),
    "multiply": Operation(expression=lambda a, p, c: f"{a[0]} * {a[1]}"),
    "concat": Operation(
        expression=lambda a, p, c: f"torch.cat(({a[0]}, {a[1]}), dim=-1)"
    ),
    "silu": Operation(expression=lambda a, p, c: f"F.silu({a[0]})"),
    "sigmoid": Operation(expression=lambda a, p, c: f"torch.sigmoid({a[0]})"),
    "scale": Operation(expression=lambda a, p, c: f"{a[0]} * {p.get('factor', 1.0)!r}"),
    "softmax": Operation(
        expression=lambda a, p, c: f"{a[0]}.softmax(dim={p.get('dim', -1)!r})"
    ),
    "matmul": Operation(expression=lambda a, p, c: f"{a[0]} @ {a[1]}"),
    "transpose": Operation(
        expression=lambda a, p, c: f"{a[0]}.transpose({p.get('dim0', -2)!r}, {p.get('dim1', -1)!r})"
    ),
    "split_heads": Operation(
        expression=lambda a, p, c: f"{a[0]}.reshape({a[0]}.shape[0], {a[0]}.shape[1], {p.get('heads', c.get('heads', 8))!r}, -1).transpose(1, 2)"
    ),
    "merge_heads": Operation(
        expression=lambda a, p, c: f"{a[0]}.transpose(1, 2).contiguous().flatten(-2)"
    ),
    "repeat_heads": Operation(
        expression=lambda a, p, c: f"{a[0]}.repeat_interleave({p.get('repeats', 1)!r}, dim=1)"
    ),
    "rope": Operation(
        expression=lambda a, p, c: f"rotary_positions({a[0]}, theta={p.get('theta', 10000.0)!r})",
        helpers=("rotary_positions",),
    ),
    "sdpa": Operation(
        expression=lambda a, p, c: f"attend({', '.join(a[:3])}, mask={a[3]}, causal={p.get('causal', True)!r}, window={p.get('window', 0)!r}, backend={p.get('backend', 'dense')!r})",
        sparse=("attend",),
    ),
    "attention_pattern": Operation(
        expression=lambda a, p, c: f"AttentionPattern({p.get('pattern', 'local')!r}, {a[0]}, {a[1]}, p={p!r})",
        sparse=("AttentionPattern",),
    ),
    "pattern_union": Operation(
        expression=lambda a, p, c: f"combine_patterns({a[0]}, {a[1]})",
        helpers=("combine_patterns",),
    ),
    "pattern_intersection": Operation(
        expression=lambda a, p, c: f"combine_patterns({a[0]}, {a[1]}, intersection=True)",
        helpers=("combine_patterns",),
    ),
    "lane_expand": Operation(
        expression=lambda a, p, c: f"{a[0]}.unsqueeze(-2).expand(*{a[0]}.shape[:-1], {p.get('lanes', 4)!r}, {a[0]}.shape[-1])"
    ),
    "lane_reduce": Operation(expression=lambda a, p, c: f"{a[0]}.mean(dim=-2)"),
    "lane_mix": Operation(
        expression=lambda a, p, c: f"torch.einsum('btn,btnc->btc', {a[1]}[..., :{p.get('lanes', 4)!r}], {a[0]})"
    ),
    "lane_merge": Operation(
        expression=lambda a, p, c: f"merge_residual_lanes({', '.join(a)}, lanes={p.get('lanes', 4)!r})",
        helpers=("merge_residual_lanes",),
    ),
    "memory_fuse": Operation(
        expression=lambda a, p, c: f"fuse_memory({a[0]}, {a[1]})",
        helpers=("fuse_memory",),
    ),
    "mask": Operation(
        expression=lambda a, p, c: f"F.pad({a[0]}, (0, {p.get('length', 0)!r}), value={p.get('mask_id', c.get('mask_id'))!r})"
    ),
    "select": Operation(
        expression=lambda a, p, c: f"select_tokens({a[0]}, {a[1]}, mask_id={p.get('mask_id', c.get('mask_id'))!r}, step=step, steps=steps, temperature={p.get('temperature', 0)!r}, strategy={p.get('strategy', 'confidence')!r})",
        helpers=("select_tokens",),
    ),
}


def attention_constructor(p, config, shape, *, cross=False):
    width = shape[-1]
    heads = p.get("heads", config.get("heads", 8))
    settings = {
        "width": width,
        "heads": heads,
        "kv_heads": p.get("kv_heads", heads),
        "head_dim": p.get("head_dim"),
        "value_dim": p.get("value_dim"),
        "causal": p.get("causal", not cross),
        "window": p.get("window", 0),
        "rope": p.get("rope", True),
        "theta": p.get("theta", 10000.0),
        "qk_norm": p.get("qk_norm", False),
        "gated": p.get("gated", False),
        "conv_taps": p.get("conv_taps", 0),
        "backend": p.get("backend", "dense"),
    }
    defaults = {
        "head_dim": None,
        "value_dim": None,
        "causal": True,
        "window": 0,
        "rope": True,
        "theta": 10000.0,
        "qk_norm": False,
        "gated": False,
        "conv_taps": 0,
        "backend": "dense",
    }
    arguments = [
        f"{key}={value!r}"
        for key, value in settings.items()
        if key not in defaults or value != defaults[key]
    ]
    return "Attention(" + ", ".join(arguments) + ")"


OPERATIONS["attention"] = Operation(
    constructor=attention_constructor, runtime=("Attention",)
)
OPERATIONS["cross_attention"] = Operation(
    constructor=lambda p, c, s: attention_constructor(p, c, s, cross=True),
    runtime=("Attention",),
)


def needs_schedule(graph):
    return any(
        n["data"]["kind"] == "select"
        or (n["data"]["kind"] == "block" and needs_schedule(n["data"]["body"]))
        for n in graph["nodes"]
    )


def has_moe(graph):
    return any(
        n["data"]["kind"] == "moe"
        or (n["data"].get("body") and has_moe(n["data"]["body"]))
        for n in graph["nodes"]
    )


def has_shared_modules(graph):
    return any(
        n["data"].get("share")
        or (n["data"].get("body") and has_shared_modules(n["data"]["body"]))
        for n in graph["nodes"]
    )


class SourceGenerator:
    def __init__(self, project, report):
        self.project, self.report = project, report
        imported_names = {
            name
            for op in OPERATIONS.values()
            for group in (op.runtime, op.helpers, op.sparse)
            for name in group
        }
        self.imported_names = imported_names
        self.class_names = Names(
            imported_names
            | {
                "Model",
                "CustomBlock",
                "SparseMoE",
                "clear_moe_losses",
                "torch",
                "nn",
                "F",
            }
        )
        self.classes = []
        self.cache = {}
        self.files = {}
        self.imports = {
            "runtime": set(),
            "export_helpers": set(),
            "sparse_runtime": set(),
        }
        self.custom_imports = []
        self.shared = has_shared_modules(project["graph"])

    def class_name(self, label):
        base = "".join(
            word.capitalize() for word in identifier(label, "network").split("_")
        )
        return self.class_names.take(base)

    def graph_class(self, graph, label, prefix="", config=None):
        config = {
            **(self.project["config"] if config is None else config),
            **graph.get("variables", {}),
        }
        contracts = {
            n["id"]: self.report["shapes"].get(prefix + n["id"]) for n in graph["nodes"]
        }
        semantic = definition_signature(
            {"data": {"kind": "block", "params": {}, "body": graph}}, config
        )
        key = json.dumps([semantic, contracts, config], sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        class_name = self.class_name(label)
        self.cache[key] = class_name
        sequence = ordered(graph)
        pool = Names(RESERVED | self.imported_names)
        names = {
            n["id"]: pool.take(identifier(n["data"].get("label") or n["data"]["kind"]))
            for n in sequence
        }
        init = ["super().__init__()"]
        if self.shared:
            init.append(
                "shared_modules = {} if shared_modules is None else shared_modules"
            )
        inputs = {}
        for node in sequence:
            if node["data"]["kind"] == "input":
                p = params(node, config)
                inputs.setdefault(p.get("index", 0), node)
        last_required = max(
            (i for i, n in inputs.items() if not params(n, config).get("optional")),
            default=-1,
        )
        parameters, input_names = [], {}
        for index in range(max(inputs, default=-1) + 1):
            input_node = inputs.get(index)
            name = (
                names[input_node["id"]]
                if input_node
                else pool.take(f"unused_input_{index}")
            )
            input_names[index] = name
            parameters.append(name + ("=None" if index > last_required else ""))
        if needs_schedule(graph):
            parameters.extend(["*", "step=0", "steps=1"])
        forward = []
        if not prefix and has_moe(graph):
            self.imports["runtime"].add("clear_moe_losses")
            forward.append("clear_moe_losses(self)")
        returns = []
        attributes = {}
        for node in sequence:
            data = node["data"]
            kind = data["kind"]
            p = params(node, config)
            name = names[node["id"]]
            edges = {
                e.get("targetHandle") or "a": e
                for e in graph["edges"]
                if e["target"] == node["id"]
            }
            args = [
                names[edges[port]["source"]] if port in edges else "None"
                for port in port_names(node)
            ]
            if kind == "input":
                input_name = input_names[p.get("index", 0)]
                if name != input_name:
                    forward.append(f"{name} = {input_name}")
                continue
            if kind == "output":
                returns.append((p.get("index", 0), args[0]))
                continue
            if kind == "moe":
                source = edges["a"]["source"]
                width = contracts[source]["shape"][-1]
                nested = self.moe_class(node, prefix, config, width)
                constructor = nested + ("(shared_modules)" if self.shared else "()")
            elif kind in ("block", "loop"):
                body = data["body"]
                nested = self.graph_class(
                    body,
                    data.get("role") or data.get("label", "Block"),
                    prefix + node["id"] + "/",
                    {**config, **p},
                )
                constructor = nested + ("(shared_modules)" if self.shared else "()")
                body_schedule = needs_schedule(body)
                if kind == "loop":
                    if p.get("shared", True):
                        self.add_module(init, name, constructor, data.get("share"))
                        counter = pool.take("step_index")
                        count = pool.take(name + "_iterations")
                        init.append(f"self.{count} = {p.get('iterations', 1)!r}")
                        loop = f"for {counter} in range(self.{count}):"
                        call = f"self.{name}"
                        total = f"self.{count}"
                    else:
                        constructor = f"nn.ModuleList([{constructor} for _ in range({p.get('iterations', 1)!r})])"
                        self.add_module(init, name, constructor, data.get("share"))
                        counter, layer = pool.take("step_index"), pool.take("layer")
                        loop = (
                            f"for {counter}, {layer} in enumerate(self.{name}):"
                            if body_schedule
                            else f"for {layer} in self.{name}:"
                        )
                        call, total = layer, f"len(self.{name})"
                    forward.extend([f"{name} = {args[0]}", loop])
                    call_args = [name, *args[1:]]
                    if body_schedule:
                        call_args.extend([f"step={counter}", f"steps={total}"])
                    forward.append(f"    {name} = {call}({', '.join(call_args)})")
                    attributes[node["id"]] = name
                    continue
                if body_schedule:
                    schedule = data.get("iteration")
                    args.extend(
                        [f"step={schedule['index']}", f"steps={schedule['total']}"]
                        if schedule
                        else ["step=step", "steps=steps"]
                    )
            elif kind == "custom":
                alias = self.class_name(name + " custom block")
                module_name = "custom_" + identifier(alias)
                self.files[module_name + ".py"] = (
                    "import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n"
                    + p["source"]
                    + "\n"
                )
                self.custom_imports.append(
                    f"from {module_name} import CustomBlock as {alias}"
                )
                constructor = (
                    f"{alias}(**{p.get('kwargs', {})!r})"
                    if p.get("kwargs")
                    else f"{alias}()"
                )
            else:
                operation = OPERATIONS[kind]
                for module, symbols in (
                    ("runtime", operation.runtime),
                    ("export_helpers", operation.helpers),
                    ("sparse_runtime", operation.sparse),
                ):
                    self.imports[module].update(symbols)
                if operation.expression:
                    forward.append(f"{name} = {operation.expression(args, p, config)}")
                    continue
                source = edges.get("a", {}).get("source")
                shape = contracts.get(source, {}).get("shape") if source else None
                constructor = operation.constructor(p, config, shape)
                if kind == "attention":
                    args = [args[0], f"mask={args[1]}"]
                elif kind == "cross_attention":
                    args = [args[0], f"memory={args[1]}", f"mask={args[2]}"]
            self.add_module(init, name, constructor, data.get("share"))
            attributes[node["id"]] = name
            forward.append(f"{name} = self.{name}({', '.join(args)})")
        for node in sequence:
            tie = params(node, config).get("tie")
            if tie:
                init.extend(
                    [
                        "# Share the embedding and output projection weights.",
                        f"self.{attributes[node['id']]}.weight = self.{attributes[tie]}.weight",
                    ]
                )
        values = [value for _, value in sorted(returns)]
        if not values:
            raise ValueError(
                "The graph needs an output node before it can be exported."
            )
        forward.append(
            "return "
            + (values[0] if len(values) == 1 else "(" + ", ".join(values) + ")")
        )
        signature = ", ".join(["self", *parameters])
        init_signature = "self, shared_modules=None" if self.shared else "self"
        source = (
            f"class {class_name}(nn.Module):\n    def __init__({init_signature}):\n"
            + "\n".join("        " + line for line in init)
        )
        source += f"\n\n    def forward({signature}):\n" + "\n".join(
            "        " + line for line in forward
        )
        self.classes.append(source)
        return class_name

    def moe_class(self, node, prefix, config, width):
        settings = params(node, config)
        _, experts = moe_structure(node, settings)
        body = node["data"]["body"]
        scope = {**config, **settings, **body.get("variables", {})}
        name = self.class_name(node["data"].get("label", "Mixture of experts"))
        init = ["super().__init__()"]
        if self.shared:
            init.append(
                "shared_modules = {} if shared_modules is None else shared_modules"
            )
        attributes = []
        for index, expert in enumerate(experts):
            expert_name = f"expert_{index + 1}"
            nested = self.graph_class(
                expert["data"]["body"],
                expert["data"].get("label", expert_name),
                prefix + node["id"] + "/" + expert["id"] + "/",
                {**scope, **params(expert, scope)},
            )
            constructor = nested + ("(shared_modules)" if self.shared else "()")
            share = expert["data"].get("share")
            if share:
                init.extend(
                    [
                        f"if {share!r} not in shared_modules:",
                        f"    shared_modules[{share!r}] = {constructor}",
                        f"{expert_name} = shared_modules[{share!r}]",
                    ]
                )
            else:
                init.append(f"{expert_name} = {constructor}")
            attributes.append(expert_name)
        init.append(
            f"self.routing = SparseMoE(width={width!r}, experts=[{', '.join(attributes)}], top_k={settings.get('top_k', 2)!r}, balance_weight={settings.get('balance_weight', 0.01)!r}, router={settings.get('router', 'softmax')!r})"
        )
        self.imports["runtime"].add("SparseMoE")
        signature = "self, shared_modules=None" if self.shared else "self"
        source = (
            f"class {name}(nn.Module):\n    def __init__({signature}):\n"
            + "\n".join("        " + line for line in init)
        )
        source += "\n\n    def forward(self, x):\n        return self.routing(x)"
        self.classes.append(source)
        return name

    def add_module(self, lines, name, constructor, share):
        if share:
            lines.extend(
                [
                    f"# Parameter group: {str(share).replace(chr(10), ' ').replace(chr(13), ' ')}",
                    f"if {share!r} not in shared_modules:",
                    f"    shared_modules[{share!r}] = {constructor}",
                    f"self.{name} = shared_modules[{share!r}]",
                ]
            )
        else:
            lines.append(f"self.{name} = {constructor}")

    def generate(self):
        root = self.graph_class(
            self.project["graph"], self.project.get("name", "Network")
        )
        used_names = {
            node.id
            for node in ast.walk(ast.parse("\n\n".join(self.classes)))
            if isinstance(node, ast.Name)
        }
        imports = ["from torch import nn"]
        if "torch" in used_names:
            imports.insert(0, "import torch")
        if "F" in used_names:
            imports.append("from torch.nn import functional as F")
        imports.extend(
            f"from {module} import {', '.join(sorted(symbols))}"
            for module, symbols in self.imports.items()
            if symbols
        )
        imports.extend(self.custom_imports)
        source = (
            "# Generated by Weave. Ordinary PyTorch; randomly initialized weights.\n"
        )
        source += "\n".join(imports) + "\n\n\n" + "\n\n\n".join(self.classes)
        source += f"\n\n\nModel = {root}\n"
        return source, self.files
