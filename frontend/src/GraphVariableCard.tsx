import { useEffect, useRef, useState } from "react";
import { Braces, GripVertical, Maximize2, Minus, Plus } from "lucide-react";
import { GraphVariables } from "./GraphVariables";

/** An unconnected, movable canvas card; its values remain readable at any zoom. */
export function GraphVariableCard({
  values,
  inherited,
  nested,
  onChange,
  onOpen,
}: {
  values: Record<string, number>;
  inherited: Record<string, number>;
  nested: boolean;
  onChange: (values: Record<string, number>) => void;
  onOpen: () => void;
}) {
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [collapsed, setCollapsed] = useState(false);
  const cardRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const card = cardRef.current;
    const canvas = card?.closest(".flow-wrap");
    if (!card || !canvas) return;
    const observer = new ResizeObserver(() => {
      const maxX = Math.max(0, canvas.clientWidth - card.offsetWidth - 24);
      const maxY = Math.max(0, canvas.clientHeight - card.offsetHeight - 24);
      setOffset((previous) => {
        const x = Math.min(previous.x, maxX),
          y = Math.min(previous.y, maxY);
        return x === previous.x && y === previous.y ? previous : { x, y };
      });
    });
    observer.observe(canvas);
    observer.observe(card);
    return () => observer.disconnect();
  }, []);
  const drag = useRef<{
    x: number;
    y: number;
    startX: number;
    startY: number;
  } | null>(null);
  return (
    <aside
      ref={cardRef}
      className="graph-variable-card nodrag nopan nowheel"
      aria-label="Canvas variables"
      style={{ transform: `translate(${offset.x}px,${offset.y}px)` }}
    >
      <header
        className="variable-card-heading"
        onPointerDown={(event) => {
          if (event.target instanceof Element && event.target.closest("button"))
            return;
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = {
            x: event.clientX,
            y: event.clientY,
            startX: offset.x,
            startY: offset.y,
          };
        }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          const canvas = event.currentTarget
            .closest(".flow-wrap")
            ?.getBoundingClientRect();
          const card =
            event.currentTarget.parentElement?.getBoundingClientRect();
          const maxX = Math.max(
              0,
              (canvas?.width ?? 1000) - (card?.width ?? 280) - 24,
            ),
            maxY = Math.max(
              0,
              (canvas?.height ?? 800) - (card?.height ?? 250) - 24,
            );
          setOffset({
            x: Math.max(
              0,
              Math.min(
                maxX,
                drag.current.startX + event.clientX - drag.current.x,
              ),
            ),
            y: Math.max(
              0,
              Math.min(
                maxY,
                drag.current.startY + event.clientY - drag.current.y,
              ),
            ),
          });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <GripVertical size={13} />
        <Braces size={14} />
        <strong>Variables</strong>
        <button
          aria-label={
            collapsed ? "Expand canvas variables" : "Minimize canvas variables"
          }
          onClick={() => setCollapsed((value) => !value)}
        >
          {collapsed ? <Plus size={13} /> : <Minus size={13} />}
        </button>
        <button aria-label="Open full variable table" onClick={onOpen}>
          <Maximize2 size={13} />
        </button>
      </header>
      {!collapsed && (
        <div className="variable-card-body">
          <GraphVariables
            compact
            values={values}
            inherited={inherited}
            nested={nested}
            onChange={onChange}
          />
        </div>
      )}
    </aside>
  );
}
