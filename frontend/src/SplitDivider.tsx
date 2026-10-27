import { useRef } from "react";
import type { PointerEvent } from "react";

/** Keyboard and pointer resizing changes pane proportions, never graph positions. */
export function SplitDivider({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  const drag = useRef<{ start: number; width: number; value: number } | null>(
    null,
  );
  const clamp = (value: number) => Math.max(30, Math.min(70, value));
  function start(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    const width =
      event.currentTarget.parentElement?.getBoundingClientRect().width;
    if (!width) return;
    drag.current = { start: event.clientX, width, value };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  return (
    <div
      className="split-divider"
      role="separator"
      aria-label="Resize graph panes"
      title="Drag to resize · Arrow keys to adjust · Double-click to center"
      aria-orientation="vertical"
      aria-valuemin={30}
      aria-valuemax={70}
      aria-valuenow={Math.round(value)}
      aria-valuetext={`${Math.round(value)} percent editable graph`}
      tabIndex={0}
      onPointerDown={start}
      onPointerMove={(event) => {
        if (drag.current)
          onChange(
            clamp(
              drag.current.value +
                ((event.clientX - drag.current.start) / drag.current.width) *
                  100,
            ),
          );
      }}
      onPointerUp={() => {
        drag.current = null;
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
      onLostPointerCapture={() => {
        drag.current = null;
      }}
      onDoubleClick={() => onChange(50)}
      onKeyDown={(event) => {
        const values: Record<string, number> = {
          ArrowLeft: value - 5,
          ArrowRight: value + 5,
          Home: 30,
          End: 70,
          Enter: 50,
        };
        if (event.key in values) {
          event.preventDefault();
          onChange(clamp(values[event.key]));
        }
      }}
    >
      <span aria-hidden="true" />
    </div>
  );
}
