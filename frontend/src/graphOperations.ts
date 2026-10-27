import type { Graph, GraphNode, NodeData, TensorContract } from "./types";
import { newId } from "./ui";
import { adapterBody, layoutGraph } from "./graph";

interface LayerContext {
  width: number;
  heads: number;
  transformerBody: () => Graph;
}
const layerBuilders = {
  linear: ({ width }: LayerContext): NodeData => ({
    kind: "linear",
    role: "adapter",
    label: "Independent learned layer",
    params: { out: width },
  }),
  ffn: ({ width }: LayerContext): NodeData => ({
    kind: "ffn",
    label: "Independent feed-forward",
    params: { hidden: width * 2 },
  }),
  norm: (): NodeData => ({
    kind: "norm",
    label: "RMSNorm",
    params: { eps: 1e-6 },
  }),
  adapter: ({ width }: LayerContext): NodeData => ({
    kind: "block",
    role: "adapter",
    label: "Independent residual adapter",
    params: { width, ffn_width: width * 2 },
    body: adapterBody(),
  }),
  transformer: ({
    width,
    heads: configuredHeads,
    transformerBody,
  }: LayerContext): NodeData => {
    const heads = width % configuredHeads === 0 ? configuredHeads : 1;
    const body = transformerBody();
    for (const node of body.nodes)
      if (["attention", "cross_attention"].includes(node.data.kind))
        Object.assign(node.data.params, {
          kv_heads: heads % 2 === 0 ? 2 : 1,
          head_dim: width / heads,
          value_dim: width / heads,
          rope: (width / heads) % 2 === 0,
        });
    return {
      kind: "block",
      role: "transformer",
      label: "Independent transformer layer",
      params: { width, heads, ffn_width: width * 2 },
      body,
    };
  },
} satisfies Record<string, (context: LayerContext) => NodeData>;
export type InsertionKind = keyof typeof layerBuilders;
export function isInsertionKind(value: string): value is InsertionKind {
  return Object.hasOwn(layerBuilders, value);
}
export function buildLayer(
  kind: InsertionKind,
  context: LayerContext,
): NodeData {
  return layerBuilders[kind](context);
}

/** Split a data edge without changing either endpoint's interface. */
export function insertOnEdge(
  graph: Graph,
  edgeId: string,
  data: NodeData,
): GraphNode {
  const edge = graph.edges.find((e) => e.id === edgeId);
  if (!edge) throw new Error("The selected connection no longer exists");
  const node: GraphNode = {
    id: newId(),
    type: "architecture",
    position: { x: 0, y: 0 },
    data,
  };
  graph.nodes.push(node);
  graph.edges = graph.edges.filter((e) => e.id !== edgeId);
  graph.edges.push(
    { ...edge, id: newId(), target: node.id, targetHandle: "a" },
    { ...edge, id: newId(), source: node.id, sourceHandle: "out" },
  );
  layoutGraph(graph);
  return node;
}

/** A weight group is independent of the paths carrying tensors between applications. */
export function addSharedApplication(
  graph: Graph,
  nodeId: string,
  input?: TensorContract | null,
  output?: TensorContract | null,
): GraphNode {
  const original = graph.nodes.find((n) => n.id === nodeId);
  if (!original) throw new Error("The selected node no longer exists");
  original.data.share ||= "shared_" + newId();
  const next = structuredClone(original);
  next.id = newId();
  next.selected = false;
  next.data.label += " · next application";
  graph.nodes.push(next);
  const incoming = graph.edges.filter((e) => e.target === original.id);
  const state = incoming.find((e) => (e.targetHandle || "a") === "a");
  if (
    state &&
    input &&
    output &&
    JSON.stringify(input) === JSON.stringify(output)
  ) {
    const outgoing = graph.edges.filter((e) => e.source === original.id);
    graph.edges = graph.edges.filter((e) => e.source !== original.id);
    graph.edges.push(
      {
        id: newId(),
        source: original.id,
        target: next.id,
        sourceHandle: "out",
        targetHandle: "a",
        type: "smoothstep",
      },
      ...outgoing.map((e) => ({ ...e, source: next.id })),
    );
    graph.edges.push(
      ...incoming
        .filter((e) => (e.targetHandle || "a") !== "a")
        .map((e) => ({ ...e, id: newId(), target: next.id })),
    );
  } else
    graph.edges.push(
      ...incoming.map((e) => ({ ...e, id: newId(), target: next.id })),
    );
  layoutGraph(graph);
  return next;
}
