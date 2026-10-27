import { graphSchema } from "./api.ts";
import type { Graph, GraphNode, JsonValue, Parameters } from "./types";

export const graphClipboardType = "application/x-weave-graph";
const format = "weave.graph-selection";

/** Copy document data, without React Flow state, cached routes or view callbacks. */
function documentGraph(graph: Graph): Graph {
  return {
    ...(graph.variables ? { variables: { ...graph.variables } } : {}),
    nodes: graph.nodes.map((node) => {
      const {
        kind,
        label,
        params,
        body,
        share,
        role,
        description,
        iteration,
        tokenizer,
      } = node.data;
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
          ...(tokenizer ? { tokenizer: structuredClone(tokenizer) } : {}),
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
  variables?: Parameters,
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
  if (variables) captureExternalBindings(selection, variables);
  return JSON.stringify({ format, version: 1, graph: selection });
}

/** A reference copy carries external dimensions with it, while keeping nested
 * block variables editable and vocabulary bindings owned by the receiving input.
 */
function captureExternalBindings(
  graph: Graph,
  variables: Parameters,
  local = new Set<string>(),
) {
  const inputBindings = new Set(["vocab", "mask_id"]);
  function resolve(value: JsonValue): JsonValue {
    if (typeof value === "string" && value.startsWith("$")) {
      const name = value.slice(1);
      if (local.has(name) || inputBindings.has(name)) return value;
      if (name === "sqrt_width")
        return !local.has("width") && typeof variables.width === "number"
          ? Math.sqrt(variables.width)
          : value;
      return variables[name] ?? value;
    }
    if (Array.isArray(value)) return value.map(resolve);
    if (value && typeof value === "object")
      return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [key, resolve(item)]),
      );
    return value;
  }
  for (const node of graph.nodes) {
    node.data.params = Object.fromEntries(
      Object.entries(node.data.params).map(([key, value]) => [
        key,
        value === undefined ? null : resolve(value),
      ]),
    );
    if (node.data.body)
      captureExternalBindings(
        node.data.body,
        variables,
        new Set([
          ...local,
          ...Object.keys(node.data.params),
          ...Object.keys(node.data.body.variables ?? {}),
        ]),
      );
  }
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
