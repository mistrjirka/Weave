/** Shared presentation of a graph document; never changes its computation. */
import type { Edge } from "@xyflow/react";
import type {
  DisplayNode,
  Graph,
  GraphNode,
  NodeLocation,
  Project,
  ValidationReport,
} from "./types";
import { allNodes, inferRole } from "./graph";

const ports: Record<string, string[]> = {
  input: [],
  output: ["a"],
  add: ["a", "b"],
  multiply: ["a", "b"],
  subtract: ["a", "b"],
  concat: ["a", "b"],
  gate: ["a", "b"],
  attention: ["a", "b"],
  cross_attention: ["a", "b", "c"],
  select: ["a", "b"],
  engram: ["a", "b"],
  lane_mix: ["a", "b"],
  lane_merge: ["a", "b", "c"],
  memory_fuse: ["a", "b"],
  sdpa: ["a", "b", "c", "d"],
  matmul: ["a", "b"],
  attention_pattern: ["a", "b"],
  pattern_union: ["a", "b"],
  pattern_intersection: ["a", "b"],
};
export function optionalPorts(n: GraphNode) {
  if (n.data.body)
    return n.data.body.nodes
      .filter(
        (v: GraphNode) => v.data.kind === "input" && v.data.params?.optional,
      )
      .map((v: GraphNode) =>
        String.fromCharCode(97 + Number(v.data.params.index || 0)),
      );
  return (
    {
      attention: ["b"],
      cross_attention: ["c"],
      sdpa: ["d"],
      attention_pattern: ["b"],
    }[n.data.kind] || []
  );
}
export function nodePorts(n: GraphNode) {
  return n.data.kind === "custom" && n.data.params?.input_shapes?.length
    ? n.data.params.input_shapes.map((_, i: number) =>
        String.fromCharCode(97 + i),
      )
    : n.data.body
      ? n.data.body.nodes
          .filter((n: GraphNode) => n.data.kind === "input")
          .map((_, i: number) => String.fromCharCode(97 + i))
      : ports[n.data.kind] || ["a"];
}

export function parameterGroups(project: Project) {
  const groups = new Map<string, { label: string; members: NodeLocation[] }>();
  for (const item of allNodes(project)) {
    const group = item.node.data.share;
    if (!group) continue;
    let entry = groups.get(group);
    if (!entry) {
      entry = { label: "θ" + (groups.size + 1), members: [] };
      groups.set(group, entry);
    }
    entry.members.push(item);
  }
  return groups;
}
interface ViewOptions {
  report: ValidationReport | null;
  prefix: string;
  groups: ReturnType<typeof parameterGroups>;
  selection: string | null;
}
export function presentNodes(
  graph: Graph,
  options: ViewOptions & { onOpen: (id: string) => void },
): DisplayNode[] {
  const { report, prefix, groups, selection, onOpen } = options;
  return graph.nodes.map((n) => ({
    ...n,
    data: {
      ...n.data,
      _focused: n.id === selection,
      _ports: nodePorts(n),
      _optional: optionalPorts(n),
      _role: inferRole(n.data),
      _groupLabel: n.data.share ? groups.get(n.data.share)?.label : undefined,
      _shape: report?.shapes[prefix + n.id],
      _count: report?.counts[prefix + n.id],
      _error: report?.errors.some(
        (e) =>
          e.node === prefix + n.id || e.node?.startsWith(prefix + n.id + "/"),
      ),
      _open: onOpen,
    },
  }));
}
export function presentEdges(graph: Graph, options: ViewOptions): Edge[] {
  const { report, prefix, groups, selection } = options;
  return [
    ...graph.edges.map((edge, index) => ({
      ...edge,
      type: "tensor",
      data: {
        ...edge.data,
        _hasSelection: !!selection,
        _related: edge.source === selection || edge.target === selection,
        _routingLane: graph.edges
          .slice(0, index)
          .filter((e) => e.source === edge.source).length,
      },
      label:
        report?.shapes[prefix + edge.source]?.shape.map(String).join(" × ") ||
        "",
    })),
    ...Array.from(groups.entries()).flatMap(([group, info]) => {
      const members = graph.nodes.filter((n) => n.data.share === group);
      return members.slice(1).map((node) => ({
        id: "weights:" + group + ":" + node.id,
        source: members[0].id,
        target: node.id,
        sourceHandle: "weights-out",
        targetHandle: "weights-in",
        type: "weightLink",
        selectable: false,
        deletable: false,
        reconnectable: false,
        data: { label: info.label, visualOnly: true },
      }));
    }),
  ];
}

export function graphScope(project: Project, path: string[]): Graph {
  let graph = project.graph;
  for (const id of path) {
    const body = graph.nodes.find((n) => n.id === id)?.data.body;
    if (!body) throw new Error("The containing block is no longer available.");
    graph = body;
  }
  return graph;
}
