import type { GraphNode } from "./types";
import { lanePorts } from "./lanePorts.ts";

const ports: Record<string, string[]> = {
  input: [],
  output: ["a"],
  add: ["a", "b"],
  multiply: ["a", "b"],
  subtract: ["a", "b"],
  concat: ["a", "b"],
  gate: ["a", "b"],
  attention: ["a", "b"],
  cross_attention: ["a", "b", "c"],
  select: ["a", "b"],
  engram: ["a", "b"],
  lane_mix: ["a", "b"],
  lane_merge: ["a", "b", "c"],
  memory_fuse: ["a", "b"],
  sdpa: ["a", "b", "c", "d"],
  matmul: ["a", "b"],
  attention_pattern: ["a", "b"],
  pattern_union: ["a", "b"],
  pattern_intersection: ["a", "b"],
};
export function optionalPorts(n: GraphNode) {
  if (n.data.body)
    return n.data.body.nodes
      .filter(
        (v: GraphNode) => v.data.kind === "input" && v.data.params?.optional,
      )
      .map((v: GraphNode) =>
        String.fromCharCode(97 + Number(v.data.params.index || 0)),
      );
  return (
    {
      attention: ["b"],
      cross_attention: ["c"],
      sdpa: ["d"],
      attention_pattern: ["b"],
    }[n.data.kind] || []
  );
}
export function nodePorts(n: GraphNode) {
  if (n.data.kind === "lane_join") return lanePorts(n.data.params.lanes ?? 4);
  if (n.data.kind === "moe_combine") {
    const count = n.data.params.experts;
    return [
      "a",
      ...Array.from(
        {
          length:
            typeof count === "number" && Number.isInteger(count)
              ? Math.max(0, Math.min(256, count))
              : 0,
        },
        (_, i) => `e${i}`,
      ),
    ];
  }
  return n.data.kind === "custom" && n.data.params?.input_shapes?.length
    ? n.data.params.input_shapes.map((_, i: number) =>
        String.fromCharCode(97 + i),
      )
    : n.data.body
      ? n.data.body.nodes
          .filter((n: GraphNode) => n.data.kind === "input")
          .map((_, i: number) => String.fromCharCode(97 + i))
      : ports[n.data.kind] || ["a"];
}
