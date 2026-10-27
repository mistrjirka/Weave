import {
  createContext,
  useContext,
  useMemo,
  useId,
  useState,
  useCallback,
  useLayoutEffect,
  useRef,
} from "react";
import type { ReactNode, RefObject } from "react";
import { BaseEdge, EdgeLabelRenderer, useViewport } from "@xyflow/react";
import type { EdgeProps, Node, Edge } from "@xyflow/react";
import { nodeRectangles, routeConnection, roundedPath } from "./routing";
import type { Rectangle, Route } from "./routing";
import { placeLabels, estimatedLabelSize } from "./labels";
import type { LabelRequest, LabelPlacement, LabelSize } from "./labels";
import { layoutSignature } from "./layout-core";
import type { GraphLayout, PlannedEdge } from "./layout-core";

interface RoutingContext {
  rectangles: Rectangle[];
  routes: Record<string, PlannedEdge>;
  showLabels: boolean;
}
const Obstacles = createContext<RoutingContext>({
  rectangles: [],
  routes: {},
  showLabels: false,
});
const RegisterLabel = createContext<
  (id: string, label: LabelRequest | null) => void
>(() => {});
const Placements = createContext<Map<string, LabelPlacement>>(new Map());
export function RoutingProvider({
  nodes,
  edges,
  layout,
  showLabels = false,
  children,
}: {
  nodes: Node[];
  edges: Edge[];
  layout?: GraphLayout;
  showLabels?: boolean;
  children: ReactNode;
}) {
  const rectangles = useMemo(() => nodeRectangles(nodes), [nodes]);
  const context = useMemo(
    () => ({
      rectangles,
      routes:
        layout?.signature === layoutSignature({ nodes, edges })
          ? layout.routes
          : {},
      showLabels,
    }),
    [rectangles, nodes, edges, layout, showLabels],
  );
  const [labels, setLabels] = useState<Map<string, LabelRequest>>(new Map());
  const register = useCallback(
    (id: string, label: LabelRequest | null) =>
      setLabels((previous) => {
        if (
          (label &&
            JSON.stringify(previous.get(id)) === JSON.stringify(label)) ||
          (!label && !previous.has(id))
        )
          return previous;
        const next = new Map(previous);
        if (label) next.set(id, label);
        else next.delete(id);
        return next;
      }),
    [],
  );
  const placements = useMemo(
    () => placeLabels([...labels.values()], rectangles),
    [labels, rectangles],
  );
  return (
    <Obstacles.Provider value={context}>
      <RegisterLabel.Provider value={register}>
        <Placements.Provider value={placements}>{children}</Placements.Provider>
      </RegisterLabel.Provider>
    </Obstacles.Provider>
  );
}

function useLabelSize(
  ref: RefObject<HTMLSpanElement | null>,
  text: string,
  weight: boolean,
): LabelSize & { known: boolean } {
  const [measured, setMeasured] = useState<{
    text: string;
    width: number;
    height: number;
  } | null>(null);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const box = entry.borderBoxSize[0];
      const width = box?.inlineSize ?? element.offsetWidth,
        height = box?.blockSize ?? element.offsetHeight;
      setMeasured((previous) =>
        previous?.text === text &&
        Math.abs(previous.width - width) < 0.1 &&
        Math.abs(previous.height - height) < 0.1
          ? previous
          : { text, width, height },
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, text, weight]);
  return measured?.text === text
    ? { ...measured, known: true }
    : { ...estimatedLabelSize(text, weight), known: false };
}
function PlacedLabel({
  id,
  text,
  weight,
  visible,
  route,
  preferred,
  selected,
}: {
  id: string;
  text: string;
  weight: boolean;
  visible: boolean;
  route: Route;
  preferred?: { x: number; y: number };
  selected: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null),
    size = useLabelSize(ref, text, weight),
    register = useContext(RegisterLabel),
    placement = useContext(Placements).get(id);
  useLayoutEffect(() => {
    if (!visible || !size.known) {
      register(id, null);
      return;
    }
    register(id, {
      id,
      points: route.points,
      width: size.width,
      height: size.height,
      priority: selected ? 0 : weight ? 1 : 2,
      preferred: preferred || route.label,
    });
    return () => register(id, null);
  }, [
    id,
    visible,
    size.known,
    size.width,
    size.height,
    register,
    route,
    preferred,
    selected,
    weight,
  ]);
  const center = placement?.center || route.label;
  return (
    <EdgeLabelRenderer>
      <span
        ref={ref}
        data-edge-label={id}
        data-label-visible={
          visible && size.known && !!placement ? "true" : "false"
        }
        className={
          weight
            ? "weight-link-label"
            : "tensor-edge-label " + (visible ? "is-visible" : "")
        }
        style={{
          left: 0,
          top: 0,
          visibility: visible && size.known && placement ? "visible" : "hidden",
          transform: `translate(-50%,-50%) translate(${center.x}px,${center.y}px)`,
        }}
      >
        {text}
      </span>
    </EdgeLabelRenderer>
  );
}
function RoutedEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  selected,
  data,
  label,
  type,
}: EdgeProps) {
  const { rectangles, routes, showLabels } = useContext(Obstacles),
    { zoom } = useViewport();
  const weight = type === "weightLink",
    lane = typeof data?._routingLane === "number" ? data._routingLane : 0,
    planned = routes[id];
  const route = useMemo(() => {
    if (planned && planned.points.length >= 2) {
      const points = planned.points.map((p) => ({ ...p }));
      points[0] = { x: sourceX, y: sourceY };
      points[points.length - 1] = { x: targetX, y: targetY };
      return {
        points,
        path: roundedPath(points),
        label: planned.label || points[Math.floor(points.length / 2)],
        detour: points.length > 4,
        blocked: false,
      };
    }
    return routeConnection(
      { x: sourceX, y: sourceY },
      { x: targetX, y: targetY },
      rectangles,
      source,
      target,
      lane,
      weight,
    );
  }, [
    sourceX,
    sourceY,
    targetX,
    targetY,
    rectangles,
    source,
    target,
    lane,
    weight,
    planned,
  ]);
  const [hovered, setHovered] = useState(false),
    marker = "arrow-" + useId().replace(/:/g, "");
  const color = weight ? "#8563a4" : route.detour ? "#5e7e9c" : "#6a8072";
  const text = weight
    ? `${typeof data?.label === "string" ? data.label : ""} · shared weights`
    : route.blocked
      ? "Separate overlapping cards or use Auto arrange"
      : typeof label === "string"
        ? label
        : "";
  return (
    <>
      {!weight && (
        <defs>
          <marker
            id={marker}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth={Math.min(32, 10 / zoom)}
            markerHeight={Math.min(32, 10 / zoom)}
            markerUnits="userSpaceOnUse"
            orient="auto"
          >
            <path d="M1,1 L9,5 L1,9 Z" fill={color} />
          </marker>
        </defs>
      )}
      <g
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <title>{text}</title>
        <BaseEdge
          id={id}
          path={route.path}
          markerEnd={weight ? undefined : `url(#${marker})`}
          interactionWidth={Math.max(24, 26 / zoom)}
          className={`routed-path ${weight ? "weight-path" : ""} ${route.detour ? "bypass-path" : ""} ${route.blocked ? "route-obstructed" : ""}`}
          style={{
            stroke: color,
            strokeWidth: selected || data?._related ? 2.8 : 1.7,
            strokeOpacity:
              data?._hasSelection && !data?._related && !selected ? 0.4 : 1,
            strokeDasharray: weight ? "7 5" : undefined,
            vectorEffect: "non-scaling-stroke",
          }}
        />
      </g>
      {text && (
        <PlacedLabel
          id={id}
          text={text}
          weight={weight}
          visible={weight || selected || hovered || route.blocked || showLabels}
          route={route}
          preferred={planned?.label}
          selected={!!selected}
        />
      )}
    </>
  );
}
export const edgeTypes = { tensor: RoutedEdge, weightLink: RoutedEdge };
