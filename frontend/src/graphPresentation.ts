/** Shared presentation of a graph document; never changes its computation. */
import type { Edge } from "@xyflow/react";
import type {
  DisplayNode,
  Parameters,
  Graph,
  NodeLocation,
  Project,
  ValidationReport,
} from "./types";
import { nodePorts, optionalPorts } from "./nodePorts";
import { outputPorts, edgeContract } from "./lanePorts";
import { previewParameters } from "./graphVariables";
import { allNodes, inferRole } from "./graph";

/** Clear React Flow's stored multi-selection as well as the inspector focus. */
export function clearGraphSelection(graph: Graph) {
  for (const node of graph.nodes) {
    if (node.selected) node.selected = false;
    if (node.data.body) clearGraphSelection(node.data.body);
  }
  for (const edge of graph.edges) if (edge.selected) edge.selected = false;
}

export { nodePorts } from "./nodePorts";

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
  options: ViewOptions & {
    onOpen: (id: string) => void;
    variables?: Parameters;
  },
): DisplayNode[] {
  const { report, prefix, groups, selection, onOpen } = options;
  return graph.nodes.map((n) => ({
    ...n,
    data: {
      ...n.data,
      _focused: n.id === selection,
      _previewParams: previewParameters(n.data.params, options.variables ?? {}),
      _ports: nodePorts({
        ...n,
        data: {
          ...n.data,
          params: previewParameters(n.data.params, options.variables ?? {}),
        },
      }),
      _outputs: outputPorts({
        ...n.data,
        params: previewParameters(n.data.params, options.variables ?? {}),
      }),
      _optional: optionalPorts(n),
      _role: inferRole(n.data),
      _groupLabel: n.data.share ? groups.get(n.data.share)?.label : undefined,
      _shape: report?.shapes[prefix + n.id],
      _tokenizer:
        report?.tokenizers?.[prefix + n.id] ?? n.data.tokenizer ?? undefined,
      _count: report?.counts[prefix + n.id],
      _error: report?.errors.some(
        (e) =>
          e.node === prefix + n.id || e.node?.startsWith(prefix + n.id + "/"),
      ),
      _portErrors: Object.fromEntries(
        (report?.errors || [])
          .filter((issue) => issue.node === prefix + n.id && issue.port)
          .map((issue) => [issue.port, issue.message]),
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
        edgeContract(report, prefix, edge)?.shape.map(String).join(" × ") || "",
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
