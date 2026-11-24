import type { Dispatch, SetStateAction } from "react";
import { addEdge, applyEdgeChanges, applyNodeChanges, type Edge, type ReactFlowProps } from "@xyflow/react";
import type { Graph, GraphNode } from "../types";
import type { ConnectionAttempt } from "../connectionFeedback";
import { newId } from "../ui";

type Flow = ReactFlowProps<GraphNode, Edge>;

export function useCanvasMutations(
  projectId: string,
  path: string[],
  changeGraph: (change: (graph: Graph) => void, record?: boolean, markDirty?: boolean) => void,
  setSelection: Dispatch<SetStateAction<string | null>>,
  setSelectedEdge: Dispatch<SetStateAction<string | null>>,
  setConnectionAttempt: Dispatch<SetStateAction<ConnectionAttempt | null>>,
) {
  const onNodesChange: NonNullable<Flow["onNodesChange"]> = (changes) => {
    changeGraph((graph) => {
      graph.nodes = applyNodeChanges(changes, graph.nodes);
      if (changes.some((change) => change.type === "select")) {
        const nodes = graph.nodes.filter((node) => node.selected);
        setSelection((current) => nodes.some((node) => node.id === current) ? current : (nodes.at(-1)?.id ?? null));
        if (nodes.length) setSelectedEdge(null);
      }
    }, changes.some((change) => change.type === "remove"), changes.some((change) => ["remove", "add", "position"].includes(change.type)));
  };
  const onEdgesChange: NonNullable<Flow["onEdgesChange"]> = (changes) => {
    changeGraph((graph) => {
      graph.edges = applyEdgeChanges(changes.filter((change) => !("id" in change) || !change.id.startsWith("weights:")), graph.edges);
    }, changes.some((change) => change.type === "remove"), changes.some((change) => ["remove", "add", "replace"].includes(change.type)));
  };
  const onConnect: NonNullable<Flow["onConnect"]> = (connection) => {
    const edgeId = newId();
    setConnectionAttempt({ projectId, path: [...path], edgeId });
    changeGraph((graph) => {
      graph.edges = addEdge({ ...connection, id: edgeId, type: "smoothstep" }, graph.edges);
    });
  };
  const onBeforeDelete: NonNullable<Flow["onBeforeDelete"]> = async ({ nodes, edges }) => {
    const nodeIds = new Set(nodes.map((node) => node.id));
    const edgeIds = new Set(edges.map((edge) => edge.id));
    changeGraph((graph) => {
      graph.nodes = graph.nodes.filter((node) => !nodeIds.has(node.id));
      graph.edges = graph.edges.filter((edge) => !edgeIds.has(edge.id) && !nodeIds.has(edge.source) && !nodeIds.has(edge.target));
    });
    setSelection(null);
    setSelectedEdge(null);
    return false;
  };
  return { onNodesChange, onEdgesChange, onConnect, onBeforeDelete };
}
