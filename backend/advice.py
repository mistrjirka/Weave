"""Nonblocking design hints over resolved graph contracts, without executing code.

Rules describe local evidence and exceptions, never assert model quality. Add a
rule to RULES to extend the analysis; all graph kinds share the same traversal.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

MLP_SOURCE = "https://www.deeplearningbook.org/contents/mlp.html"
LOSS_SOURCE = (
    "https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html"
)
AUTOGRAD_SOURCE = (
    "https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
)
ATTENTION_SOURCE = "https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html"


@dataclass
class Scope:
    nodes: dict
    parents: dict
    children: dict
    live: set
    shapes: dict
    counts: dict
    config: dict
    prefix: str
    resolve: Callable

    def shape(self, node_id):
        contract = self.shapes.get(self.prefix + node_id)
        return contract["shape"] if contract else None

    def settings(self, node):
        return self.resolve(node["data"].get("params", {}), self.config)

    def input_node(self, node_id, port="a"):
        upstream = self.parents[node_id].get(port)
        return self.nodes.get(upstream)


def hint(scope, node, code, title, message, suggestion, source):
    return {
        "node": scope.prefix + node["id"],
        "code": code,
        "title": title,
        "message": message,
        "suggestion": suggestion,
        "source": source,
    }


def narrow_bottleneck(scope, node):
    parent = scope.input_node(node["id"])
    before = scope.shape(parent["id"]) if parent else None
    after = scope.shape(node["id"])
    if not before or not isinstance(before[-1], int):
        return
    width = before[-1]
    kind = node["data"]["kind"]
    narrow = (
        scope.settings(node).get("hidden", width * 4) if kind == "ffn" else after[-1]
    )
    # Product heuristic, deliberately limited to extreme internal compression.
    # A scalar output/classifier is not itself an architectural mistake.
    if type(narrow) is not int or not 1 <= narrow <= 4 or width < 16 * narrow:
        return
    expanded = kind == "ffn"
    pending = list(scope.children[node["id"]])
    seen = set()
    while pending and not expanded:
        candidate = pending.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        child = scope.nodes[candidate]
        # Stop at merges: a residual/bypass may preserve the original signal.
        if (
            child["data"]["kind"] not in {"linear", "silu", "sigmoid", "norm", "scale"}
            or len(scope.parents[candidate]) != 1
        ):
            continue
        shape = scope.shape(candidate)
        expanded = bool(shape and isinstance(shape[-1], int) and shape[-1] >= width)
        pending.extend(scope.children[candidate])
    if expanded:
        yield hint(
            scope,
            node,
            "narrow-bottleneck",
            "Very narrow hidden bottleneck",
            f"This branch compresses {width} features to {narrow} before expanding again. Later expansion cannot recover distinctions already lost here; a bypass can preserve them.",
            "Check whether this compression is intentional. Try a wider hidden representation and compare validation results. Small adapters and autoencoders can intentionally use bottlenecks. This is a heuristic, not a minimum-width rule.",
            MLP_SOURCE,
        )


def consecutive_linear(scope, node):
    parent = scope.input_node(node["id"])
    if not parent or parent["data"]["kind"] != "linear":
        return
    grandparent = scope.input_node(parent["id"])
    if grandparent and grandparent["data"]["kind"] == "linear":
        return  # One hint per uninterrupted chain.
    yield hint(
        scope,
        node,
        "consecutive-linear",
        "Linear layers without an activation",
        "Consecutive linear projections compose into a single affine transformation. Extra depth here does not add a nonlinear function.",
        "If you intended a nonlinear hidden layer, insert an activation such as SiLU. Keep the projections if you want a factorized or constrained linear map.",
        MLP_SOURCE,
    )


def output_softmax(scope, node):
    if not any(
        scope.nodes[c]["data"]["kind"] == "output" for c in scope.children[node["id"]]
    ):
        return
    yield hint(
        scope,
        node,
        "output-softmax",
        "Probabilities or training logits?",
        "This output contains softmax probabilities. PyTorch CrossEntropyLoss expects raw logits and already includes log-softmax.",
        "For cross-entropy training, expose the tensor before softmax and flatten [B,T,V] to [B*T,V] (or move the class axis). Keep softmax if this is an inference probability output or your loss explicitly expects probabilities.",
        LOSS_SOURCE,
    )


def discrete_sampling(scope, node):
    yield hint(
        scope,
        node,
        "discrete-sampling",
        "Sampling interrupts gradients",
        "Token selection returns discrete integer IDs. Gradients cannot flow back through these selections to the predictor logits.",
        "Use this path for generation. Train the predictor on its floating-point logits before token selection; for masked diffusion, compute the loss on the intended masked positions.",
        AUTOGRAD_SOURCE,
    )


def zero_scale(scope, node):
    if scope.settings(node).get("factor", 1) != 0:
        return
    yield hint(
        scope,
        node,
        "zero-scale",
        "This branch is scaled to zero",
        "For finite inputs, this fixed scale outputs zeros and supplies zero gradients to its input through this path. A separate bypass may still carry signal and gradients.",
        "Check whether you intended to disable this branch. Keep zero for an intentional ablation; use a nonzero factor if the branch should learn through this path.",
        AUTOGRAD_SOURCE,
    )


def empty_attention_rows(scope, node):
    settings = scope.settings(node)
    if settings.get("pattern") != "custom":
        return
    rows = [
        index for index, row in enumerate(settings.get("matrix", [])) if not any(row)
    ]
    if not rows:
        return
    preview = ", ".join(map(str, rows[:8])) + ("…" if len(rows) > 8 else "")
    yield hint(
        scope,
        node,
        "empty-attention-rows",
        "Attention queries with no visible keys",
        f"This custom pattern has {len(rows)} empty query row(s), at zero-based indices {preview}. True allows attention; every key in these rows is False.",
        "If this pattern is used directly as a mask, these queries cannot read any values. Allow an intended key, often the query itself for self-attention, or keep empty rows for intentional masking. A later pattern union may supply keys; an intersection cannot.",
        ATTENTION_SOURCE,
    )


RULES = (
    (frozenset({"linear", "ffn"}), narrow_bottleneck),
    (frozenset({"linear", "head"}), consecutive_linear),
    (frozenset({"softmax"}), output_softmax),
    (frozenset({"select"}), discrete_sampling),
    (frozenset({"scale"}), zero_scale),
    (frozenset({"attention_pattern"}), empty_attention_rows),
)


def design_advice(project, shapes, counts, resolve):
    advice = []

    def visit(graph, config, prefix=""):
        nodes = {n["id"]: n for n in graph["nodes"]}
        parents, children = defaultdict(dict), defaultdict(list)
        for edge in graph["edges"]:
            if edge["source"] not in nodes or edge["target"] not in nodes:
                return
            parents[edge["target"]][edge.get("targetHandle") or "a"] = edge["source"]
            children[edge["source"]].append(edge["target"])
        live = set()
        pending = [n["id"] for n in nodes.values() if n["data"]["kind"] == "output"]
        while pending:
            current = pending.pop()
            if current in live:
                continue
            live.add(current)
            pending.extend(parents[current].values())
        scope = Scope(
            nodes, parents, children, live, shapes, counts, config, prefix, resolve
        )
        for node in nodes.values():
            key = prefix + node["id"]
            if not shapes.get(key):
                continue
            if node["id"] not in live:
                if counts.get(key, 0) > 0 or node["data"].get("share"):
                    advice.append(
                        hint(
                            scope,
                            node,
                            "unused-branch",
                            "Learned branch does not reach an output",
                            "This application cannot affect any output in this graph. A loss on those outputs supplies no gradient through this branch.",
                            "Connect it to an output if it should participate in training, or keep it as a draft. Explicitly shared weights may still receive gradients through another application.",
                            AUTOGRAD_SOURCE,
                        )
                    )
                continue
            for kinds, rule in RULES:
                if node["data"]["kind"] in kinds:
                    advice.extend(rule(scope, node))
            if node["data"].get("body"):
                visit(
                    node["data"]["body"], {**config, **scope.settings(node)}, key + "/"
                )

    visit(project["graph"], project.get("config", {}))
    return advice
