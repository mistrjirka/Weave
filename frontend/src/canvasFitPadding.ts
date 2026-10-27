import type { FitViewOptions } from "@xyflow/react";

/** Space for readable variables and canvas controls. */
export function canvasFitInsets(canvas: HTMLElement | null) {
  const card = canvas?.querySelector<HTMLElement>(".graph-variable-card");
  return {
    top:
      Math.max(card?.offsetHeight ?? 280, window.innerWidth <= 760 ? 100 : 0) +
      32,
    left: 24,
    right: 32,
    bottom: 76,
  };
}

/** Fitted nodes leave room for the graph's editable variable card. */
export function canvasFitPadding(
  canvas: HTMLElement | null,
): NonNullable<FitViewOptions["padding"]> {
  const inset = canvasFitInsets(canvas);
  return {
    top: `${inset.top}px`,
    left: `${inset.left}px`,
    right: `${inset.right}px`,
    bottom: `${inset.bottom}px`,
  };
}
