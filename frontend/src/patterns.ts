import type { Parameters } from "./types";
export const patternDefinitions = {
  full: { label: "Full attention pattern", fields: [] },
  causal: { label: "Causal pattern", fields: [] },
  local: { label: "Local window pattern", fields: ["window"] },
  global: {
    label: "Global token pattern",
    fields: ["global_tokens", "global_stride"],
  },
  block: {
    label: "Block-sparse pattern",
    fields: ["block_size", "neighbor_blocks"],
  },
  random: { label: "Random sparse pattern", fields: ["density", "seed"] },
  content_topk: { label: "Content top-k pattern", fields: ["top_k"] },
  custom: { label: "Custom attention matrix", fields: ["matrix"] },
} satisfies Record<string, { label: string; fields: string[] }>;
export type PatternKind = keyof typeof patternDefinitions;
export function isPatternKind(value: unknown): value is PatternKind {
  return typeof value === "string" && Object.hasOwn(patternDefinitions, value);
}
export const defaultMatrix = [
  [true, false, false, false],
  [true, true, false, false],
  [false, true, true, false],
  [false, false, true, true],
];
export const patternDefaults: Parameters = {
  window: 16,
  global_tokens: [0],
  global_stride: 0,
  block_size: 16,
  neighbor_blocks: 0,
  density: 0.1,
  seed: 0,
  top_k: 8,
  causal: false,
  matrix: defaultMatrix,
};
export function changePattern(params: Parameters, value: unknown): Parameters {
  if (!isPatternKind(value)) throw new Error("Unknown attention pattern");
  return { ...structuredClone(patternDefaults), ...params, pattern: value };
}
export function visiblePatternField(params: Parameters, key: string) {
  const kind = params.pattern;
  return (
    key === "pattern" ||
    key === "causal" ||
    (isPatternKind(kind) &&
      patternDefinitions[kind].fields.some((field) => field === key))
  );
}
const predicates: Record<
  PatternKind,
  (q: number, k: number, p: Parameters) => boolean
> = {
  full: () => true,
  causal: (q, k) => k <= q,
  local: (q, k, p) => Math.abs(q - k) < (Number(p.window) || 16),
  global: (q, k, p) =>
    (p.global_tokens || [0]).includes(q) ||
    (p.global_tokens || [0]).includes(k) ||
    (!!Number(p.global_stride) &&
      (q % Number(p.global_stride) === 0 || k % Number(p.global_stride) === 0)),
  block: (q, k, p) =>
    Math.abs(
      Math.floor(q / (Number(p.block_size) || 16)) -
        Math.floor(k / (Number(p.block_size) || 16)),
    ) <= (Number(p.neighbor_blocks) || 0),
  random: (q, k, p) => {
    let hash =
      (q * 1103515245 + k * 12345 + (Number(p.seed) || 0) * 2654435761) >>> 0;
    hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b) >>> 0;
    hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b) >>> 0;
    return (
      ((hash ^ (hash >>> 16)) & 0x7fffffff) <
        (p.density === undefined ? 0.1 : Number(p.density)) * 2147483648 ||
      q === k
    );
  },
  // A static schematic can only illustrate selection; runtime selection depends on input features.
  content_topk: (q, k) =>
    q === k || Math.floor(k / 8) === (Math.floor(q / 8) * 3 + 2) % 8,
  custom: (q, k, p) => {
    const matrix = p.matrix;
    if (!matrix?.length) return false;
    return !!matrix[
      Math.min(matrix.length - 1, Math.floor((q * matrix.length) / 64))
    ]?.[
      Math.min(matrix[0].length - 1, Math.floor((k * matrix[0].length) / 64))
    ];
  },
};
export function previewAllowed(
  row: number,
  column: number,
  params: Parameters,
) {
  const q = row * 8,
    k = column * 8,
    kind = isPatternKind(params.pattern) ? params.pattern : "local";
  return predicates[kind](q, k, params) && (!params.causal || k <= q);
}
