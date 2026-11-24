import type { Graph, GraphNode, NodeData, Parameters } from "../types";

/** Only a single inner transformer can be controlled by a wrapper inspector. */
export function wrappedTransformer(data: NodeData): GraphNode | undefined {
  const layers = data.body?.nodes.filter(
    (node) => node.data.kind !== "input" && node.data.kind !== "output",
  );
  return layers?.length === 1 &&
    layers[0].data.kind === "block" &&
    layers[0].data.role === "transformer"
    ? layers[0]
    : undefined;
}
/** New layers link heads to the parent and copy the parent's FFN width locally. */
export function inheritTransformerDefaults(data: NodeData, scope: Parameters) {
  if (typeof scope.heads === "number") data.params.heads = "$heads";
  if (data.body?.nodes.some((node) => node.data.kind === "ffn")) {
    const candidate = data.params.ffn_width;
    const initial =
      typeof scope.ffn_width === "number"
        ? scope.ffn_width
        : typeof candidate === "number"
          ? candidate
          : 256;
    data.body.variables = { ...data.body.variables, ffn_width: initial };
    delete data.params.ffn_width;
  }
}
export function transformerParentScope(data: NodeData, scope: Parameters) {
  return wrappedTransformer(data) ? bodyScope(data, scope) : scope;
}

export function transformerGraph(data: NodeData): Graph | undefined {
  const child = wrappedTransformer(data);
  return child ? transformerGraph(child.data) : data.body;
}
export function bodyScope(data: NodeData, scope: Parameters): Parameters {
  return {
    ...scope,
    ...Object.fromEntries(
      Object.entries(data.params).map(([key, value]) => [
        key,
        typeof value === "string" && value.startsWith("$")
          ? (scope[value.slice(1)] ?? value)
          : value,
      ]),
    ),
    ...data.body?.variables,
  };
}

export function transformerAttention(data: NodeData): GraphNode | undefined {
  const nodes = transformerGraph(data)?.nodes ?? [];
  const attention = nodes.filter((node) => node.data.kind === "attention");
  return attention.length === 1 &&
    !nodes.some((node) => !!node.data.body) &&
    nodes.filter((node) => node.data.kind === "ffn").length <= 1
    ? attention[0]
    : undefined;
}
/** The residual is a separate operation from the attention output gate. */
function attentionResidualNode(data: NodeData): GraphNode | undefined {
  const graph = transformerGraph(data);
  const attention = transformerAttention(data);
  if (!attention || !graph) return undefined;
  const inputs = new Set(
    graph.nodes
      .filter(
        (node) =>
          node.data.kind === "input" && (node.data.params.index ?? 0) === 0,
      )
      .map((node) => node.id),
  );
  const downstream = new Set<string>([attention.id]);
  const pending = [attention.id];
  while (pending.length) {
    const source = pending.pop()!;
    for (const edge of graph.edges.filter((item) => item.source === source)) {
      const next = graph.nodes.find((node) => node.id === edge.target);
      if (
        next &&
        ["norm", "layer_norm", "dropout", "scale"].includes(next.data.kind) &&
        !downstream.has(next.id)
      ) {
        downstream.add(next.id);
        pending.push(next.id);
      }
    }
  }
  const candidates = graph.nodes.filter((node) => {
    if (!["add", "gate"].includes(node.data.kind)) return false;
    const sources = (handle: string) =>
      graph.edges
        .filter(
          (edge) =>
            edge.target === node.id && (edge.targetHandle ?? "a") === handle,
        )
        .map((edge) => edge.source);
    return (
      sources("a").some((id) => inputs.has(id)) &&
      sources("b").some((id) => downstream.has(id))
    );
  });
  return candidates.length === 1 ? candidates[0] : undefined;
}
export function transformerResidual(data: NodeData): "add" | "gate" | null {
  const node = attentionResidualNode(data);
  return node?.data.kind === "add" || node?.data.kind === "gate"
    ? node.data.kind
    : null;
}
export function setTransformerResidual(data: NodeData, kind: "add" | "gate") {
  const node = attentionResidualNode(data);
  if (!node)
    throw new Error("This transformer has no editable attention residual.");
  if (node.data.kind !== kind) delete node.data.share;
  node.data.kind = kind;
  node.data.label =
    kind === "gate" ? "Gated attention residual" : "Attention residual";
}
