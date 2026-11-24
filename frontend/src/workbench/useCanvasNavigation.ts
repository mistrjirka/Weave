import type { Dispatch, RefObject, SetStateAction } from "react";
import type { Edge, ReactFlowProps } from "@xyflow/react";
import { isCompactViewport } from "../compactViewport";
import type { GraphNode, Project } from "../types";
import type { WorkspacePanel } from "../workspaceLocation";
import { graphAt } from "./document";

export type ConnectionPickerState = {
  projectId: string;
  scope: string;
  source: string;
  sourceHandle: string;
  position: { x: number; y: number };
  bounds: { left: number; top: number; right: number; bottom: number };
};
type Flow = ReactFlowProps<GraphNode, Edge>;

export function useCanvasNavigation(
  project: Project, path: string[], canvasRef: RefObject<HTMLDivElement | null>,
  ref: RefObject<Project>, viewPath: RefObject<string[]>,
  setSelection: Dispatch<SetStateAction<string | null>>,
  setSelectedEdge: Dispatch<SetStateAction<string | null>>,
  setMobilePanel: Dispatch<SetStateAction<WorkspacePanel>>,
  setConnectionPicker: Dispatch<SetStateAction<ConnectionPickerState | null>>,
  setHistory: Dispatch<SetStateAction<Project[]>>,
  setFuture: Dispatch<SetStateAction<Project[]>>,
  open: (id: string) => void,
) {
  const onConnectEnd: NonNullable<Flow["onConnectEnd"]> = (event, state) => {
    if (state.isValid || state.toNode || state.fromHandle?.type !== "source" || !state.fromNode || state.fromHandle.id?.startsWith("weights-")) return;
    const point = "changedTouches" in event ? event.changedTouches[0] : event;
    const bounds = canvasRef.current?.getBoundingClientRect();
    if (!point || !bounds || point.clientX < bounds.left || point.clientX > bounds.right || point.clientY < bounds.top || point.clientY > bounds.bottom) return;
    const target = document.elementFromPoint(point.clientX, point.clientY);
    if (target?.closest(".react-flow__controls,.react-flow__panel,.react-flow__minimap,button,input")) return;
    setConnectionPicker({ projectId: project.id, scope: path.join("/"), source: state.fromNode.id, sourceHandle: state.fromHandle.id ?? "out", position: { x: point.clientX, y: point.clientY }, bounds: { left: bounds.left, top: bounds.top, right: bounds.right, bottom: bounds.bottom } });
  };
  const onNodeClick: NonNullable<Flow["onNodeClick"]> = (_, node) => {
    const nodes = graphAt(ref.current, viewPath.current).nodes.filter((item) => item.selected);
    setSelection(nodes.some((item) => item.id === node.id) ? node.id : (nodes.at(-1)?.id ?? null));
    setSelectedEdge(null);
    if (isCompactViewport()) setMobilePanel("properties");
  };
  const onNodeDoubleClick: NonNullable<Flow["onNodeDoubleClick"]> = (_, node) => { if (node.data.body) open(node.id); };
  const onEdgeClick: NonNullable<Flow["onEdgeClick"]> = (_, edge) => {
    if (edge.id.startsWith("weights:")) return;
    setSelection(null);
    setSelectedEdge(edge.id);
    if (isCompactViewport()) setMobilePanel("properties");
  };
  const onPaneClick: NonNullable<Flow["onPaneClick"]> = () => { setSelection(null); setSelectedEdge(null); };
  const onNodeDragStart: NonNullable<Flow["onNodeDragStart"]> = () => {
    setHistory((items) => [...items.slice(-39), structuredClone(project)]);
    setFuture([]);
  };
  return { onConnectEnd, onNodeClick, onNodeDoubleClick, onEdgeClick, onPaneClick, onNodeDragStart };
}
