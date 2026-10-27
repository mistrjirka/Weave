import { graphSchema } from "./api.ts";
import type { Graph, GraphNode } from "./types";

export const graphClipboardType = "application/x-weave-graph";
const format = "weave.graph-selection";

/** Copy document data, without React Flow state, cached routes or view callbacks. */
function documentGraph(graph: Graph): Graph {
  return {
    nodes: graph.nodes.map((node) => {
      const { kind, label, params, body, share, role, description, iteration } =
        node.data;
      return {
        id: node.id,
        type: "architecture",
        position: { ...node.position },
        data: {
          kind,
          label,
          params: structuredClone(params),
          ...(body ? { body: documentGraph(body) } : {}),
          ...(share ? { share } : {}),
          ...(role ? { role } : {}),
          ...(description ? { description } : {}),
          ...(iteration ? { iteration: { ...iteration } } : {}),
        },
      };
    }),
    edges: graph.edges.map(
      ({ id, source, target, sourceHandle, targetHandle }) => ({
        id,
        source,
        target,
        sourceHandle,
        targetHandle,
        type: "smoothstep",
      }),
    ),
  };
}

export function serializeSelection(
  graph: Graph,
  focus: string | null,
): string | null {
  const nodes = graph.nodes.filter((node) => node.selected);
  if (!nodes.length && focus) {
    const node = graph.nodes.find((node) => node.id === focus);
    if (node) nodes.push(node);
  }
  if (!nodes.length) return null;
  const ids = new Set(nodes.map((node) => node.id));
  const selection = documentGraph({
    nodes,
    edges: graph.edges.filter(
      (edge) => ids.has(edge.source) && ids.has(edge.target),
    ),
  });
  return JSON.stringify({ format, version: 1, graph: selection });
}

function validScope(graph: Graph): boolean {
  const ids = new Set(graph.nodes.map((node) => node.id));
  return (
    ids.size === graph.nodes.length &&
    graph.edges.every((edge) => ids.has(edge.source) && ids.has(edge.target)) &&
    graph.nodes.every((node) => !node.data.body || validScope(node.data.body))
  );
}

export function parseSelection(text: string): Graph | null {
  try {
    const value: unknown = JSON.parse(text);
    if (
      typeof value !== "object" ||
      value === null ||
      !("format" in value) ||
      value.format !== format ||
      !("version" in value) ||
      value.version !== 1 ||
      !("graph" in value)
    )
      return null;
    const result = graphSchema.safeParse(value.graph);
    return result.success && result.data.nodes.length && validScope(result.data)
      ? documentGraph(result.data)
      : null;
  } catch {
    return null;
  }
}

/** Keep sharing within a copy, but give it independent weights from its source. */
export function pasteSelection(
  graph: Graph,
  source: Graph,
  anchor: { x: number; y: number },
  newId: () => string,
): GraphNode[] {
  const pasted = documentGraph(source);
  const ids = new Map(pasted.nodes.map((node) => [node.id, newId()]));
  const groups = new Map<string, string>();
  function remapGroups(scope: Graph) {
    for (const node of scope.nodes) {
      const group = node.data.share;
      if (group) {
        if (!groups.has(group)) groups.set(group, newId());
        node.data.share = groups.get(group);
      }
      if (node.data.body) remapGroups(node.data.body);
    }
  }
  remapGroups(pasted);
  const minX = Math.min(...pasted.nodes.map((node) => node.position.x));
  const minY = Math.min(...pasted.nodes.map((node) => node.position.y));
  for (const node of pasted.nodes) {
    node.id = ids.get(node.id)!;
    node.position = {
      x: node.position.x - minX + anchor.x,
      y: node.position.y - minY + anchor.y,
    };
    node.selected = true;
    const tie = node.data.params.tie;
    if (tie) {
      const copiedEmbedding = ids.get(tie);
      if (copiedEmbedding) node.data.params.tie = copiedEmbedding;
      else delete node.data.params.tie;
    }
  }
  for (const edge of pasted.edges) {
    edge.id = newId();
    edge.source = ids.get(edge.source)!;
    edge.target = ids.get(edge.target)!;
  }
  for (const node of graph.nodes) node.selected = false;
  for (const edge of graph.edges) edge.selected = false;
  delete graph.layout;
  graph.nodes.push(...pasted.nodes);
  graph.edges.push(...pasted.edges);
  return pasted.nodes;
}
