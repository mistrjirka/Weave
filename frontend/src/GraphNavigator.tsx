import { useMemo } from "react";
import { useReactFlow } from "@xyflow/react";
import { ArrowLeft, ArrowRight, ScanSearch } from "lucide-react";
import type { Graph } from "./types";
import { computationOrder } from "./graph";

/** Navigation changes the viewport, never graph topology or stored coordinates. */
export function GraphNavigator({
  graph,
  selection,
  onFocus,
  compact = false,
}: {
  compact?: boolean;
  graph: Graph;
  selection: string | null;
  onFocus: (id: string) => void;
}) {
  const flow = useReactFlow(),
    ordered = useMemo(() => computationOrder(graph), [graph]);
  const index = Math.max(
      0,
      ordered.findIndex((n) => n.id === selection),
    ),
    current = ordered[index];
  if (!current) return null;
  function focus(index: number) {
    const node = ordered[index];
    if (!node) return;
    onFocus(node.id);
    flow.fitView({
      nodes: [{ id: node.id }],
      padding: compact
        ? { top: "16px", bottom: "76px", left: "24px", right: "24px" }
        : 0.4,
      minZoom: compact ? 0.3 : 0.8,
      maxZoom: 0.95,
      duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? 0
        : 200,
    });
  }
  return (
    <div className="mobile-node-nav" aria-label="Follow computation">
      <button
        aria-label="Previous node"
        disabled={index === 0}
        onClick={() => focus(index - 1)}
      >
        <ArrowLeft size={17} />
      </button>
      <button
        className="node-nav-current"
        data-node-id={current.id}
        aria-label={`Focus current node: ${current.data.label} ${index + 1} / ${ordered.length} · follow connections`}
        onClick={() => focus(index)}
      >
        <ScanSearch size={14} />
        <span>
          <b>{current.data.label}</b>{" "}
          <small>
            {index + 1} / {ordered.length} · follow connections
          </small>
        </span>
      </button>
      <button
        aria-label="Next node"
        disabled={index === ordered.length - 1}
        onClick={() => focus(index + 1)}
      >
        <ArrowRight size={17} />
      </button>
    </div>
  );
}
