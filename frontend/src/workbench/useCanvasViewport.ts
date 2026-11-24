import { useEffect, useRef, type RefObject } from "react";
import type { FitViewOptions, ReactFlowInstance } from "@xyflow/react";
import { canvasFitPadding } from "../canvasFitPadding";
import { isCompactViewport } from "../compactViewport";
import { computationOrder } from "../graph";
import type { Project } from "../types";
import type { WorkspaceTab } from "../workspaceLocation";
import { graphAt } from "./document";
import { motion } from "./util";

export function useCanvasViewport(reactFlow: ReactFlowInstance, ref: RefObject<Project>, viewPath: RefObject<string[]>, focusedNode: RefObject<string | null>, tab: WorkspaceTab, path: string[], splitView: boolean) {
  const canvasRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!canvasRef.current || tab !== "graph") return;
    let timer: ReturnType<typeof setTimeout>;
    const observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => fitCanvas(), 150);
    });
    observer.observe(canvasRef.current);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, [tab, path, splitView]);
  // Keep the always-visible variables card clear of fitted nodes and ports.
  function fitGraph(options: Omit<FitViewOptions, "padding"> = {}) {
    return reactFlow.fitView({
      ...options,
      padding: canvasFitPadding(canvasRef.current),
    });
  }
  function fitCanvas() {
    if (
      !isCompactViewport() &&
      (!splitView || (canvasRef.current?.clientWidth ?? 0) >= 720)
    ) {
      fitGraph({ duration: motion(250) });
      return;
    }
    const current = graphAt(ref.current, viewPath.current);
    const first =
      current.nodes.find((n) => n.id === focusedNode.current) ||
      computationOrder(current)[0];
    if (first)
      fitGraph({
        nodes: [{ id: first.id }],
        minZoom: 0.3,
        maxZoom: 0.95,
        duration: motion(200),
      });
  }
  return { canvasRef, fitGraph, fitCanvas };
}
