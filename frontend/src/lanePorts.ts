import type { Edge } from "@xyflow/react";
import type { NodeData, ValidationReport } from "./types";

export const maxLanes = 32;
export const multiOutputLanes = new Set(["lane_fork", "lane_split"]);
export function lanePorts(value: unknown): string[] {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= maxLanes
    ? Array.from({ length: value }, (_, i) => `lane${i}`)
    : [];
}
export function outputPorts(data: NodeData): string[] {
  if (multiOutputLanes.has(data.kind)) return lanePorts(data.params.lanes ?? 4);
  return data.kind === "output" ? [] : ["out"];
}
export function portLabel(port: string) {
  const lane = /^lane(\d+)$/.exec(port);
  return lane ? `lane ${Number(lane[1]) + 1}` : port;
}
export function edgeContract(
  report: ValidationReport | null | undefined,
  prefix: string,
  edge: Pick<Edge, "source" | "sourceHandle">,
) {
  const key = prefix + edge.source;
  const ports = report?.port_shapes?.[key];
  return ports
    ? ports[edge.sourceHandle || "out"]
    : (edge.sourceHandle || "out") === "out"
      ? report?.shapes[key]
      : null;
}

export const laneOperation = (kind: string) =>
  multiOutputLanes.has(kind) || kind === "lane_join";
export const portTop = (kind: string) => (laneOperation(kind) ? 116 : 68);
export function portIndex(port: string, declared: string[]) {
  const lane = /^lane(\d+)$/.exec(port);
  return Math.max(0, lane ? Number(lane[1]) : declared.indexOf(port));
}
