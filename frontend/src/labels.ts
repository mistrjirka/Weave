/** Bounding-box candidate placement for fixed/manual layouts. No node is moved. */
import type { XYPosition } from "@xyflow/react";
import type { Rectangle } from "./routing";
export interface LabelRequest {
  id: string;
  points: XYPosition[];
  width: number;
  height: number;
  priority: number;
  preferred?: XYPosition;
}
export interface LabelPlacement {
  center: XYPosition;
  box: Rectangle;
}
export function rectanglesOverlap(a: Rectangle, b: Rectangle, gap = 0) {
  return (
    a.x < b.x + b.width + gap &&
    a.x + a.width + gap > b.x &&
    a.y < b.y + b.height + gap &&
    a.y + a.height + gap > b.y
  );
}
function boxAt(request: LabelRequest, center: XYPosition): Rectangle {
  return {
    id: request.id,
    x: center.x - request.width / 2,
    y: center.y - request.height / 2,
    width: request.width,
    height: request.height,
  };
}
export function placeLabels(
  requests: LabelRequest[],
  nodes: Rectangle[],
): Map<string, LabelPlacement> {
  const placed = new Map<string, LabelPlacement>(),
    occupied = [...nodes];
  for (const label of [...requests].sort(
    (a, b) => a.priority - b.priority || a.id.localeCompare(b.id),
  )) {
    const candidates: { center: XYPosition; score: number }[] = [];
    if (label.preferred)
      candidates.push({ center: label.preferred, score: -100 });
    for (let i = 1; i < label.points.length; i++) {
      const a = label.points[i - 1],
        b = label.points[i],
        length = Math.abs(a.x - b.x) + Math.abs(a.y - b.y),
        horizontal = Math.abs(a.y - b.y) < 0.1;
      if (length < 8) continue;
      for (const fraction of [0.5, 0.35, 0.65, 0.2, 0.8]) {
        const midpoint = {
          x: a.x + (b.x - a.x) * fraction,
          y: a.y + (b.y - a.y) * fraction,
        };
        for (const offset of [0, -1, 1, -2, 2]) {
          const center = horizontal
            ? {
                x: midpoint.x,
                y: midpoint.y + offset * (label.height / 2 + 10),
              }
            : {
                x: midpoint.x + offset * (label.width / 2 + 10),
                y: midpoint.y,
              };
          candidates.push({
            center,
            score:
              Math.abs(offset) * 32 +
              Math.abs(fraction - 0.5) * 30 +
              (horizontal ? 0 : 15) -
              Math.min(length, 500) / 25,
          });
        }
      }
    }
    candidates.sort((a, b) => a.score - b.score);
    const position = candidates.find(
      (candidate) =>
        !occupied.some((rect) =>
          rectanglesOverlap(boxAt(label, candidate.center), rect, 7),
        ),
    );
    // A fixed drawing may have no free label location. Details remain in the inspector.
    if (!position) continue;
    const box = boxAt(label, position.center);
    placed.set(label.id, { center: position.center, box });
    occupied.push(box);
  }
  return placed;
}

export interface LabelSize {
  width: number;
  height: number;
}
export function estimatedLabelSize(text: string, weight = false): LabelSize {
  return {
    width: text.length * (weight ? 6.7 : 6.1) + (weight ? 24 : 18),
    height: weight ? 27 : 24,
  };
}
/** Batch DOM measurements outside render, using exactly the rendered label CSS. */
export function measureLabels(
  labels: { id: string; text: string; weight?: boolean }[],
): Map<string, LabelSize> {
  if (typeof document === "undefined")
    return new Map(
      labels.map((l) => [l.id, estimatedLabelSize(l.text, l.weight)]),
    );
  const container = document.createElement("div");
  container.style.cssText =
    "position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none";
  const elements = labels.map((label) => {
    const element = document.createElement("span");
    element.className = label.weight
      ? "weight-link-label"
      : "tensor-edge-label";
    element.style.position = "relative";
    element.style.display = "inline-block";
    element.textContent = label.text;
    container.append(element);
    return element;
  });
  document.body.append(container);
  const sizes = new Map(
    labels.map((label, index) => {
      const rect = elements[index].getBoundingClientRect();
      return [label.id, { width: rect.width, height: rect.height }] as const;
    }),
  );
  container.remove();
  return sizes;
}
