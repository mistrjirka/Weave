import dagre from "@dagrejs/dagre";
import { newId } from "./ui";
import type {
  Project,
  Graph,
  GraphNode,
  NodeData,
  NodeLocation,
  Parameters,
} from "./types";

export function allNodes(project: Project) {
  const found: NodeLocation[] = [];
  function visit(g: Graph, prefix = "") {
    for (const n of g.nodes) {
      const path = prefix + n.id;
      found.push({ node: n, path });
      if (n.data.body) visit(n.data.body, path + "/");
    }
  }
  visit(project.graph);
  return found;
}

function semantic(data: NodeData): unknown {
  const result: { kind: string; params: Parameters; body?: unknown } = {
    kind: data.kind,
    params: data.params || {},
  };
  if (data.body)
    result.body = {
      nodes: data.body.nodes
        .map((n: GraphNode) => ({
          id: n.id,
          data: semantic(n.data),
          share: n.data.share,
        }))
        .sort((a: { id: string }, b: { id: string }) =>
          a.id.localeCompare(b.id),
        ),
      edges: data.body.edges
        .map((e) => [
          e.source,
          e.sourceHandle || "out",
          e.target,
          e.targetHandle || "a",
        ])
        .sort(),
    };
  return result;
}

export function synchronizeDefinitions(before: Project, next: Project) {
  const previous = new Map(allNodes(before).map((x) => [x.path, x.node]));
  const updates = new Map<string, NodeData>();
  const existing = new Map<string, NodeData>();
  for (const { node } of allNodes(before))
    if (node.data.share) existing.set(node.data.share, node.data);
  for (const { node, path } of allNodes(next)) {
    const key = node.data.share;
    if (!key) continue;
    const old = previous.get(path);
    if (
      old?.data.share === key &&
      JSON.stringify(semantic(old.data)) !== JSON.stringify(semantic(node.data))
    )
      updates.set(key, node.data);
    else if (old?.data.share !== key && !updates.has(key))
      updates.set(key, existing.get(key) || node.data);
  }
  if (!updates.size) return;
  function apply(g: Graph, ancestors = new Set<string>()) {
    for (const n of g.nodes) {
      const group = n.data.share;
      if (group && ancestors.has(group))
        throw new Error(
          "A weight group cannot contain an application of itself. Use a Loop container for recurrence.",
        );
      const definition = group ? updates.get(group) : undefined;
      if (definition) {
        const { kind, params, body, role, description } =
          structuredClone(definition);
        Object.assign(n.data, { kind, params, role, description });
        if (body) n.data.body = body;
        else delete n.data.body;
      }
      if (n.data.body)
        apply(n.data.body, new Set([...ancestors, ...(group ? [group] : [])]));
    }
  }
  apply(next.graph);
}

export function makeIndependent(node: GraphNode) {
  delete node.data.share;
  const remap = new Map<string, string>();
  function visit(g: Graph) {
    for (const n of g.nodes) {
      if (n.data.share) {
        if (!remap.has(n.data.share))
          remap.set(n.data.share, n.data.share + "_" + newId().slice(0, 6));
        n.data.share = remap.get(n.data.share);
      }
      if (n.data.body) visit(n.data.body);
    }
  }
  if (node.data.body) visit(node.data.body);
}

export function layoutGraph(g: Graph) {
  if (!g.nodes.length) return;
  const height = Math.max(
    ...g.nodes.map((n) => n.measured?.height ?? (n.data.body ? 300 : 220)),
  );
  const layout = new dagre.graphlib.Graph({ multigraph: true }).setGraph({
    rankdir: "LR",
    align: "UL",
    ranksep: 120,
    nodesep: 110,
    edgesep: 24,
    marginx: 48,
    marginy: 100,
  });
  layout.setDefaultEdgeLabel(() => ({}));
  for (const n of g.nodes)
    layout.setNode(n.id, {
      width: n.measured?.width ?? (n.data.body ? 300 : 248),
      height,
    });
  for (const e of g.edges)
    if (layout.hasNode(e.source) && layout.hasNode(e.target))
      layout.setEdge(e.source, e.target, {}, e.id);
  dagre.layout(layout);
  for (const n of g.nodes) {
    const p = layout.node(n.id);
    n.position = { x: p.x - p.width / 2, y: p.y - height / 2 };
  }
}

/** Fresh examples are arranged once; imported and saved coordinates stay editable. */
export function arrangeExample(project: Project): string[] {
  const scopes: string[] = [];
  function visit(graph: Graph, path: string[]) {
    for (const n of graph.nodes)
      if (n.data.body) visit(n.data.body, [...path, n.id]);
    layoutGraph(graph);
    scopes.push(path.join("/"));
  }
  visit(project.graph, []);
  return scopes;
}

export function computationOrder(graph: Graph): GraphNode[] {
  const nodes = new Map(graph.nodes.map((n) => [n.id, n])),
    degrees = new Map(graph.nodes.map((n) => [n.id, 0]));
  for (const e of graph.edges)
    if (nodes.has(e.source) && nodes.has(e.target))
      degrees.set(e.target, (degrees.get(e.target) || 0) + 1);
  const ready = graph.nodes.filter((n) => !degrees.get(n.id)),
    result: GraphNode[] = [];
  for (let i = 0; i < ready.length; i++) {
    const node = ready[i];
    result.push(node);
    for (const edge of graph.edges.filter((e) => e.source === node.id)) {
      const degree = (degrees.get(edge.target) || 0) - 1;
      degrees.set(edge.target, degree);
      const next = nodes.get(edge.target);
      if (degree === 0 && next) ready.push(next);
    }
  }
  const visited = new Set(result.map((n) => n.id));
  return [...result, ...graph.nodes.filter((n) => !visited.has(n.id))];
}

export function inferRole(data: NodeData): string {
  if (data.role) return data.role;
  if (!data.body) return data.kind;
  const kinds = data.body.nodes.map((n: GraphNode) => n.data.kind);
  if (kinds.includes("select")) return "denoising";
  if (kinds.includes("attention") || kinds.includes("cross_attention"))
    return "transformer";
  if (kinds.includes("sdpa")) return "attention";
  if (kinds.includes("embedding")) return "predictor";
  if (kinds.includes("ffn")) return "adapter";
  return "composition";
}

export function adapterBody(): Graph {
  const n = (
    id: string,
    kind: string,
    label: string,
    x: number,
    y: number,
    p: Parameters = {},
  ): GraphNode => ({
    id,
    type: "architecture",
    position: { x, y },
    data: { kind, label, params: p },
  });
  const e = (source: string, target: string, targetHandle = "a") => ({
    id: newId(),
    source,
    target,
    sourceHandle: "out",
    targetHandle,
    type: "smoothstep",
  });
  return {
    nodes: [
      n("x", "input", "Adapter input", 0, 100, {
        index: 0,
        shape: ["B", "T", "$width"],
        dtype: "float32",
      }),
      n("norm", "norm", "Adapter RMSNorm", 320, 100),
      n("ffn", "ffn", "Independent learned update", 640, 100, {
        hidden: "$ffn_width",
      }),
      n("res", "add", "Residual adapter", 960, 100),
      n("out", "output", "Adapter output", 1280, 100),
    ],
    edges: [
      e("x", "norm"),
      e("norm", "ffn"),
      e("x", "res"),
      e("ffn", "res", "b"),
      e("res", "out"),
    ],
  };
}
