import { useEffect, useRef, type Dispatch, type RefObject, type SetStateAction } from "react";
import type { ReactFlowInstance } from "@xyflow/react";
import type { Project, Graph } from "../types";
import type { WorkspaceTab } from "../workspaceLocation";
import { canvasFitInsets } from "../canvasFitPadding";
import { graphClipboardType, serializeSelection, parseSelection, pasteSelection } from "../graphClipboard";
import { newId as uid } from "../ui";
import { graphAt } from "./document";

export function useGraphClipboard(project: Project, path: string[], modal: string | null, tab: WorkspaceTab, canvasRef: RefObject<HTMLDivElement | null>, ref: RefObject<Project>, viewPath: RefObject<string[]>, focusedNode: RefObject<string | null>, reactFlow: ReactFlowInstance, changeGraph: (fn: (graph: Graph) => void) => void, setSelection: Dispatch<SetStateAction<string | null>>, setSelectedEdge: Dispatch<SetStateAction<string | null>>, notify: (message: string) => void) {
  const pasteSequence = useRef({ key: "", count: 0 });
  useEffect(() => {
    // Clipboard events work on LAN HTTP too, without Clipboard API permissions.
    const ownsClipboard = (event: ClipboardEvent) => {
      const target = event.target;
      return (
        !modal &&
        tab === "graph" &&
        target instanceof Element &&
        !target.closest(
          "input,textarea,select,[contenteditable],[data-reference-pane]",
        ) &&
        (canvasRef.current?.contains(target) || target === document.body) &&
        (window.getSelection()?.isCollapsed ?? true)
      );
    };
    const onCopy = (event: ClipboardEvent) => {
      if (!ownsClipboard(event) || !event.clipboardData) return;
      const text = serializeSelection(
        graphAt(ref.current, viewPath.current),
        focusedNode.current,
      );
      if (!text) return;
      event.clipboardData.setData(graphClipboardType, text);
      event.clipboardData.setData("text/plain", text);
      event.preventDefault();
      notify("Selected nodes copied");
    };
    const onPaste = (event: ClipboardEvent) => {
      if (!ownsClipboard(event) || !event.clipboardData) return;
      const text =
        event.clipboardData.getData(graphClipboardType) ||
        event.clipboardData.getData("text/plain");
      const source = parseSelection(text);
      const bounds = canvasRef.current?.getBoundingClientRect();
      if (!source || !bounds) return;
      event.preventDefault();
      const key = `${project.id}/${path.join("/")}/${text}`;
      const count =
        pasteSequence.current.key === key ? pasteSequence.current.count + 1 : 0;
      pasteSequence.current = { key, count };
      const inset = canvasFitInsets(canvasRef.current);
      const origin = reactFlow.screenToFlowPosition({
        x: bounds.x + bounds.width / 2,
        y:
          bounds.y +
          inset.top +
          Math.max(0, bounds.height - inset.top - inset.bottom) / 2,
      });
      const xs = source.nodes.map((node) => node.position.x);
      const ys = source.nodes.map((node) => node.position.y);
      // Center the copied group in the current viewport, retaining its arrangement.
      const anchor = {
        x:
          origin.x -
          (Math.max(...xs) - Math.min(...xs) + 280) / 2 +
          (count % 6) * 40,
        y:
          origin.y -
          (Math.max(...ys) - Math.min(...ys) + 240) / 2 +
          (count % 6) * 40,
      };
      changeGraph((graph) => {
        const nodes = pasteSelection(graph, source, anchor, uid);
        setSelection(nodes[0].id);
      });
      setSelectedEdge(null);
      notify(
        `${source.nodes.length} ${source.nodes.length === 1 ? "node" : "nodes"} pasted`,
      );
    };
    window.addEventListener("copy", onCopy);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("copy", onCopy);
      window.removeEventListener("paste", onPaste);
    };
  }, [modal, tab, path, project]);
}
