import { useReactFlow, useViewport, type FitViewOptions } from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import {
  ArrowUp,
  Box,
  ChevronRight,
  Copy,
  LayoutGrid,
  Maximize2,
  MessageSquareText,
  Tags,
} from "lucide-react";
import { isCompactViewport } from "../compactViewport";
import { copyText } from "../ui";
import type { Graph } from "../types";
import type { ReferenceLocation } from "../workspaceLocation";

export function referenceFitPadding(
  withNotes = false,
): NonNullable<FitViewOptions["padding"]> {
  return {
    top: withNotes ? `${Math.min(400, window.innerHeight * 0.42)}px` : "16px",
    bottom: "76px",
    left: "24px",
    right: isCompactViewport() ? "68px" : "24px",
  };
}
function ReferenceZoom() {
  const { zoom } = useViewport();
  const flow = useReactFlow();
  return (
    <button
      aria-label={`Reference zoom to 100 percent · current ${Math.round(zoom * 100)}%`}
      className="zoom-readout"
      title="Reference zoom to 100%"
      onClick={() => flow.zoomTo(1)}
    >
      {Math.round(zoom * 100)}%
    </button>
  );
}
export function ReferenceToolbar({
  path,
  breadcrumbs,
  onNavigate,
  setPath,
  setSelection,
  selection,
  graph,
  arranging,
  arrangeScope,
  showLabels,
  setShowLabels,
  showAnnotations,
  setShowAnnotations,
  referenceClipboard,
  setCopyNotice,
}: {
  path: string[];
  breadcrumbs: string[];
  onNavigate: Dispatch<SetStateAction<ReferenceLocation>>;
  setPath: (path: string[]) => void;
  setSelection: (id: string | null) => void;
  selection: string | null;
  graph: Graph;
  arranging: boolean;
  arrangeScope: (path: string[]) => Promise<void>;
  showLabels: boolean;
  setShowLabels: Dispatch<SetStateAction<boolean>>;
  showAnnotations: boolean;
  setShowAnnotations: Dispatch<SetStateAction<boolean>>;
  referenceClipboard: () => string | null;
  setCopyNotice: Dispatch<SetStateAction<string>>;
}) {
  const flow = useReactFlow();
  return (
    <div className="canvas-toolbar reference-toolbar">
      <div className="breadcrumbs" aria-label="Reference block path">
        {path.length > 0 && (
          <button
            className="graph-up-button"
            aria-label="Up one level"
            title="Return to the enclosing reference graph"
            onClick={() =>
              onNavigate({
                blocks: path.slice(0, -1),
                node: path.at(-1) ?? null,
              })
            }
          >
            <ArrowUp size={15} />
            <span>Up</span>
          </button>
        )}
        <button
          title="Top level reference graph"
          onClick={() => {
            setPath([]);
            setSelection(null);
          }}
        >
          <Box size={13} aria-hidden="true" />{" "}
          {path.length ? "Model" : "Model root"}
        </button>
        {breadcrumbs.map((label, index) => (
          <span key={path.slice(0, index + 1).join("/")}>
            <ChevronRight size={12} aria-hidden="true" />
            <button
              onClick={() => {
                setPath(path.slice(0, index + 1));
                setSelection(null);
              }}
            >
              {label}
            </button>
          </span>
        ))}
      </div>
      <div className="graph-actions">
        <button
          aria-label="Copy reference selection"
          title="Copy selected reference blocks (Ctrl/Cmd+C)"
          disabled={!selection && !graph.nodes.some((node) => node.selected)}
          onClick={async () => {
            const text = referenceClipboard();
            if (!text) return;
            try {
              await copyText(text);
              setCopyNotice("Copied · paste into your graph");
            } catch {
              setCopyNotice("Copy failed · try Ctrl/Cmd+C");
            }
          }}
        >
          <Copy size={15} aria-hidden="true" />
        </button>
        <button
          aria-label="Auto arrange reference"
          title="Arrange reference view"
          disabled={arranging}
          onClick={() => void arrangeScope(path)}
        >
          <LayoutGrid size={15} aria-hidden="true" />
        </button>
        <button
          aria-label={
            showLabels ? "Hide reference labels" : "Show reference labels"
          }
          aria-pressed={showLabels}
          onClick={() => setShowLabels((value) => !value)}
        >
          <Tags size={15} aria-hidden="true" />
        </button>
        <button
          aria-label={
            showAnnotations
              ? "Hide reference design notes"
              : "Show reference design notes"
          }
          title={
            showAnnotations
              ? "Hide reference design notes"
              : "Show reference design notes"
          }
          aria-pressed={showAnnotations}
          onClick={() => setShowAnnotations((value) => !value)}
        >
          <MessageSquareText size={15} aria-hidden="true" />
        </button>
        <button
          aria-label="Fit reference graph"
          title="Fit reference graph"
          onClick={() =>
            void flow.fitView({
              padding: referenceFitPadding(
                showAnnotations &&
                  graph.nodes.some((node) => node.data.annotation?.trim()),
              ),
            })
          }
        >
          <Maximize2 size={15} aria-hidden="true" />
        </button>
        <ReferenceZoom />
      </div>
    </div>
  );
}
