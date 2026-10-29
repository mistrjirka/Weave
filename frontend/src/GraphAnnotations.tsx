import { ViewportPortal } from "@xyflow/react";
import type { GraphNode } from "./types";

/** Visual notes belong to nodes; they are never computational nodes or edges. */
export function GraphAnnotations({
  nodes,
  visible,
  onEdit,
}: {
  nodes: GraphNode[];
  visible: boolean;
  onEdit?: (id: string) => void;
}) {
  if (!visible) return null;
  return (
    <ViewportPortal>
      {nodes
        .filter((node) => node.data.annotation?.trim())
        .map((node) => {
          const source = safeSource(node.data.annotationSource);
          return (
            <aside
              key={node.id}
              className="graph-annotation nodrag nopan nowheel"
              aria-label={`Design note for ${node.data.label}`}
              style={{
                left: node.position.x + (node.measured?.width ?? 240) / 2,
                top: node.position.y,
              }}
            >
              <div className="graph-annotation-title">Design note</div>
              <p>{node.data.annotation}</p>
              <div className="graph-annotation-actions">
                {source && (
                  <a href={source} target="_blank" rel="noopener noreferrer">
                    Source ↗
                  </a>
                )}
                {onEdit && (
                  <button type="button" onClick={() => onEdit(node.id)}>
                    Edit note
                  </button>
                )}
              </div>
            </aside>
          );
        })}
    </ViewportPortal>
  );
}

function safeSource(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:"
      ? url.href
      : undefined;
  } catch {
    return undefined;
  }
}
