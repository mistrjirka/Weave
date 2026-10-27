import type { ElkNode, ElkExtendedEdge, ElkPort } from "elkjs/lib/elk-api";
import type { Graph, GraphNode } from "./types";
import type { XYPosition, Node, Edge } from "@xyflow/react";
import type { LabelSize } from "./labels";
import { nodeRectangles } from "./routing.ts";

export interface PlannedEdge {
  points: XYPosition[];
  label?: XYPosition;
}
export interface GraphLayout {
  signature: string;
  engine: "elk";
  routes: Record<string, PlannedEdge>;
}
export interface LayoutLabel extends LabelSize {
  text: string;
}
export function layoutSignature(graph: {
  nodes: Node[];
  edges: Edge[];
}): string {
  return JSON.stringify([
    nodeRectangles(graph.nodes).map((n) => [n.id, n.x, n.y, n.width, n.height]),
    graph.edges.map((e) => [
      e.id,
      e.source,
      e.sourceHandle || "out",
      e.target,
      e.targetHandle || "a",
    ]),
  ]);
}
function portId(node: string, handle: string, direction: "in" | "out") {
  return JSON.stringify([node, handle, direction]);
}
function dimensions(node: GraphNode) {
  return {
    width: node.measured?.width ?? (node.data.body ? 300 : 248),
    height: node.measured?.height ?? 240,
  };
}
export function buildLayoutGraph(
  graph: Graph,
  labels: Map<string, LayoutLabel>,
): ElkNode {
  const nodes = graph.nodes.map((node) => {
    const { width, height } = dimensions(node);
    const incoming = [
      ...new Set(
        graph.edges
          .filter((e) => e.target === node.id)
          .map((e) => e.targetHandle || "a"),
      ),
    ];
    const ports: ElkPort[] = incoming.map((handle) => ({
      id: portId(node.id, handle, "in"),
      x: -10.5,
      y: 69 + 28 * Math.max(0, handle.charCodeAt(0) - 97),
      width: 0,
      height: 0,
      layoutOptions: { "elk.port.side": "WEST" },
    }));
    if (graph.edges.some((e) => e.source === node.id))
      ports.push({
        id: portId(node.id, "out", "out"),
        x: width + 10.5,
        y: 69,
        width: 0,
        height: 0,
        layoutOptions: { "elk.port.side": "EAST" },
      });
    return {
      id: node.id,
      width,
      height,
      ports,
      layoutOptions: { "elk.portConstraints": "FIXED_POS" },
    };
  });
  const ids = new Set(nodes.map((n) => n.id));
  const edges: ElkExtendedEdge[] = graph.edges
    .filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((edge) => {
      const label = labels.get(edge.id);
      return {
        id: edge.id,
        sources: [portId(edge.source, "out", "out")],
        targets: [portId(edge.target, edge.targetHandle || "a", "in")],
        ...(label
          ? {
              labels: [
                {
                  id: edge.id + ":label",
                  text: label.text,
                  width: label.width,
                  height: label.height,
                  layoutOptions: { "elk.edgeLabels.placement": "CENTER" },
                },
              ],
            }
          : {}),
      };
    });
  return {
    id: "canvas",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.padding": "[top=80,left=48,bottom=80,right=48]",
      "elk.spacing.nodeNode": "72",
      "elk.layered.spacing.nodeNodeBetweenLayers": "100",
      "elk.layered.spacing.edgeNodeBetweenLayers": "26",
      "elk.spacing.edgeNode": "22",
      "elk.spacing.edgeEdge": "18",
      "elk.spacing.edgeLabel": "10",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.crossingMinimization.greedySwitch.type": "TWO_SIDED",
      "elk.layered.thoroughness": "20",
      "elk.randomSeed": "1",
    },
    children: nodes,
    edges,
  };
}
export function applyLayout(graph: Graph, result: ElkNode): Graph {
  const next = structuredClone(graph),
    positions = new Map(result.children?.map((n) => [n.id, n]));
  for (const node of next.nodes) {
    const position = positions.get(node.id);
    if (position?.x !== undefined && position.y !== undefined)
      node.position = { x: position.x, y: position.y };
  }
  const routes: Record<string, PlannedEdge> = {};
  for (const edge of result.edges || []) {
    const section = edge.sections?.[0];
    if (!section) continue;
    const label = edge.labels?.[0];
    routes[edge.id] = {
      points: [
        section.startPoint,
        ...(section.bendPoints || []),
        section.endPoint,
      ],
      ...(label?.x !== undefined && label.y !== undefined
        ? {
            label: {
              x: label.x + (label.width || 0) / 2,
              y: label.y + (label.height || 0) / 2,
            },
          }
        : {}),
    };
  }
  next.layout = { engine: "elk", signature: layoutSignature(next), routes };
  return next;
}
