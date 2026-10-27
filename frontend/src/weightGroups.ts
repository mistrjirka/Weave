import type { NodeData, NodeLocation } from "./types";

const moduleKinds = new Set([
  "embedding",
  "linear",
  "head",
  "norm",
  "attention",
  "cross_attention",
  "ffn",
  "lookup",
  "gate",
  "hadamard",
  "monarch",
  "routing",
  "engram_lookup",
  "block",
  "loop",
  "custom",
  "mhc",
  "engram",
]);
export function supportsWeightSharing(data: NodeData): boolean {
  return moduleKinds.has(data.kind) || !!data.share;
}

/** Operator schemas distinguish reusable block types without comparing labels,
 * IDs, coordinates, or adjustable numeric settings. Input interfaces must agree.
 */
export function weightGroupType(data: NodeData): string {
  const body = data.body;
  return JSON.stringify({
    kind: data.kind,
    source: data.params.source,
    body: body
      ? {
          operations: body.nodes
            .map((node) => weightGroupType(node.data))
            .sort(),
          inputs: body.nodes
            .filter((node) => node.data.kind === "input")
            .sort(
              (a, b) =>
                Number(a.data.params.index ?? 0) -
                Number(b.data.params.index ?? 0),
            )
            .map((node) => ({
              dtype: node.data.params.dtype ?? "int64",
              rank: node.data.params.shape?.length ?? 2,
              optional: !!node.data.params.optional,
            })),
        }
      : undefined,
  });
}
export function sameWeightType(a: NodeData, b: NodeData): boolean {
  return weightGroupType(a) === weightGroupType(b);
}
export interface WeightGroupChoice {
  id: string;
  label: string;
  compatible: boolean;
}
export function weightGroupChoices(
  nodes: NodeLocation[],
  selected: NodeData,
): WeightGroupChoice[] {
  const groups = new Map<string, NodeData[]>();
  for (const { node } of nodes) {
    if (!node.data.share) continue;
    const members = groups.get(node.data.share) ?? [];
    members.push(node.data);
    groups.set(node.data.share, members);
  }
  return [...groups].flatMap(([id, members], index) => {
    const compatible = members.every((data) => sameWeightType(selected, data));
    return compatible || id === selected.share
      ? [
          {
            id,
            compatible,
            label: `θ${index + 1} · ${members[0].label} · ${members.length} ${members.length === 1 ? "application" : "applications"}${compatible ? "" : " (incompatible)"}`,
          },
        ]
      : [];
  });
}
