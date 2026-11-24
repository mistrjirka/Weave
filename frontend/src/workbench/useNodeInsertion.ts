import { useCallback, type Dispatch, type RefObject, type SetStateAction } from "react";
import type { ReactFlowInstance } from "@xyflow/react";
import type { Project, Graph, GraphNode, NodeData, CatalogEntry } from "../types";
import type { WorkspacePanel } from "../workspaceLocation";
import type { ConnectionBlockChoice } from "../ConnectionPicker";
import { catalog } from "../catalog";
import { makeIndependent } from "../graph";
import { clearGraphSelection, nodePorts } from "../graphPresentation";
import { isCompactViewport } from "../compactViewport";
import { newId as uid } from "../ui";
import { catalogNodeData } from "./catalogNodeData";
import type { ConnectionPickerState } from "./useCanvasNavigation";

export function useNodeInsertion(project: Project, path: string[], templates: Project[], customBlocks: NodeData[], connectionPicker: ConnectionPickerState | null, setConnectionPicker: Dispatch<SetStateAction<ConnectionPickerState | null>>, canvasRef: RefObject<HTMLDivElement | null>, reactFlow: ReactFlowInstance, changeGraph: (fn: (graph: Graph) => void) => void, setSelection: Dispatch<SetStateAction<string | null>>, setSelectedEdge: Dispatch<SetStateAction<string | null>>, setMobilePanel: Dispatch<SetStateAction<WorkspacePanel>>, notify: (message: string) => void) {
  const copy = <T,>(value: T): T => structuredClone(value);
  function addNode(
    entry: CatalogEntry,
    screenPosition?: { x: number; y: number },
    source?: { id: string; handle: string; targetHandle?: string },
  ) {
    addGraphNode(catalogNodeData(entry, templates, project, path), screenPosition, source);
  }
  function addGraphNode(
    data: NodeData,
    screenPosition?: { x: number; y: number },
    source?: { id: string; handle: string; targetHandle?: string },
  ) {
    const id = uid();
    const position = reactFlow.screenToFlowPosition(
      screenPosition ?? {
        x: window.innerWidth / 2 - 70,
        y: window.innerHeight / 2,
      },
    );
    changeGraph((graph) => {
      if (source && !graph.nodes.some((node) => node.id === source.id))
        throw new Error("The source node is no longer available.");
      clearGraphSelection(graph);
      const node: GraphNode = {
        id,
        type: "architecture",
        position,
        data: copy(data),
        selected: true,
      };
      makeIndependent(node);
      const port = source?.targetHandle ?? nodePorts(node)[0];
      if (port && !nodePorts(node).includes(port))
        throw new Error("The target input port is no longer available.");
      if (source && !port)
        throw new Error("Choose a block with an input port.");
      graph.nodes.push(node);
      if (source)
        graph.edges.push({
          id: uid(),
          source: source.id,
          sourceHandle: source.handle,
          target: id,
          targetHandle: port,
          type: "smoothstep",
        });
    });
    setSelection(id);
    setSelectedEdge(null);
    setMobilePanel(isCompactViewport() ? "properties" : null);
    notify(
      data.label +
        (source ? " added and connected" : " added · connect its ports"),
    );
  }
  const connectionChoices: ConnectionBlockChoice[] = connectionPicker
    ? [
        ...catalog
          .filter((entry) => entry.kind !== "input")
          .map((entry) => ({
            id: "catalog:" + entry.label,
            label: entry.label,
            description: entry.description,
            category: entry.category,
            data: catalogNodeData(entry, templates, project, path),
          })),
        ...customBlocks
          .filter(
            (data) =>
              nodePorts({
                id: "preset",
                type: "architecture",
                position: { x: 0, y: 0 },
                data,
              }).length > 0,
          )
          .map((data, index) => ({
            id: "saved:" + index,
            label: data.label,
            description: data.description ?? "Your reusable block",
            category: "Your blocks",
            data,
          })),
      ]
    : [];
  const closeConnectionPicker = useCallback(() => {
    setConnectionPicker(null);
    canvasRef.current?.focus({ preventScroll: true });
  }, []);
  return { addNode, addGraphNode, connectionChoices, closeConnectionPicker };
}
