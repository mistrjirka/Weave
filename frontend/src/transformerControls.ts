import type { GraphNode, NodeData, Parameters, JsonValue } from "./types";

export const attentionVariants = [
  { id: "mha", label: "Multi-head attention (MHA)" },
  { id: "gqa", label: "Grouped-query attention (GQA)" },
  { id: "mqa", label: "Multi-query attention (MQA)" },
  { id: "local", label: "Sliding-window attention" },
  { id: "sparse", label: "Local + global sparse attention" },
] as const;
export type AttentionVariant = (typeof attentionVariants)[number]["id"];
export function isAttentionVariant(value: string): value is AttentionVariant {
  return attentionVariants.some((variant) => variant.id === value);
}
const defaults: Parameters = {
  heads: 8,
  kv_heads: 2,
  causal: true,
  rope: true,
  qk_norm: true,
  gated: false,
  window: 0,
  backend: "dense",
  conv_taps: 0,
  theta: 10000,
};
export const transformerSettingKeys = new Set([
  ...Object.keys(defaults),
  "head_dim",
  "value_dim",
  "ffn_width",
]);
export function transformerAttention(data: NodeData): GraphNode | undefined {
  return data.body?.nodes.find((node) => node.data.kind === "attention");
}
export function transformerSettings(data: NodeData): Parameters | null {
  const attention = transformerAttention(data);
  if (!attention) return null;
  const settings: Parameters = {
    ...defaults,
    ...attention.data.params,
    ...data.params,
  };
  settings.ffn_width ??=
    data.body?.nodes.find((node) => node.data.kind === "ffn")?.data.params
      .hidden ?? 256;
  return settings;
}
export function exposeTransformerSettings(data: NodeData) {
  const attention = transformerAttention(data),
    settings = transformerSettings(data);
  if (!attention || !settings) return;
  for (const key of transformerSettingKeys) {
    const value = settings[key];
    if (value === undefined) continue;
    if (data.params[key] === undefined) data.params[key] = value;
    if (key === "ffn_width") {
      for (const node of data.body?.nodes ?? [])
        if (node.data.kind === "ffn") node.data.params.hidden = "$ffn_width";
    } else
      attention.data.params[key] =
        typeof value === "boolean" || key === "backend" ? value : "$" + key;
  }
}
function number(
  value: JsonValue | undefined,
  scope: Parameters,
  fallback: number,
): number {
  const resolved =
    typeof value === "string" && value.startsWith("$")
      ? scope[value.slice(1)]
      : value;
  return typeof resolved === "number" ? resolved : fallback;
}
export function attentionVariant(
  data: NodeData,
  scope: Parameters,
): AttentionVariant | "custom" {
  const settings = transformerSettings(data);
  if (!settings) return "custom";
  if (data.body?.nodes.some((node) => node.data.role === "attention-layout"))
    return "sparse";
  if (number(settings.window, scope, 0) > 0) return "local";
  const heads = number(settings.heads, scope, 8),
    kv = number(settings.kv_heads, scope, heads);
  return heads === kv ? "mha" : kv === 1 ? "mqa" : "gqa";
}
export function setTransformerSetting(
  data: NodeData,
  key: string,
  value: JsonValue,
) {
  exposeTransformerSettings(data);
  const tiedHeads = data.params.kv_heads === data.params.heads;
  data.params[key] = value;
  if (
    key === "heads" &&
    tiedHeads &&
    (typeof value === "number" || typeof value === "string")
  )
    data.params.kv_heads = value;
  if (typeof value === "boolean" || key === "backend") {
    const attention = transformerAttention(data);
    if (attention) attention.data.params[key] = value;
  }
}
export function setAttentionVariant(
  data: NodeData,
  variant: AttentionVariant,
  scope: Parameters,
  newId: () => string,
) {
  const attention = transformerAttention(data);
  if (!attention || !data.body)
    throw new Error(
      "Choose a transformer with compact self-attention. Expanded Q/K/V graphs remain editable inside their block.",
    );
  exposeTransformerSettings(data);
  const graph = data.body;
  const generated = new Set(
    graph.nodes
      .filter((node) => node.data.role === "attention-layout")
      .map((node) => node.id),
  );
  graph.nodes = graph.nodes.filter((node) => !generated.has(node.id));
  graph.edges = graph.edges.filter(
    (edge) => !generated.has(edge.source) && !generated.has(edge.target),
  );
  const heads = number(data.params.heads, scope, 8);
  let kv = number(data.params.kv_heads, scope, 2);
  if (variant === "mha") kv = heads;
  else if (variant === "mqa") kv = 1;
  else if (kv >= heads || heads % kv)
    kv =
      Number.isInteger(heads) && heads > 1 && heads % 2 === 0 ? heads / 2 : 1;
  data.params.kv_heads = kv;
  data.params.window = variant === "local" ? 32 : 0;
  attention.data.label = attentionVariants.find(
    (option) => option.id === variant,
  )!.label;
  if (variant !== "sparse") return;
  const input = graph.edges.find(
    (edge) =>
      edge.target === attention.id && (edge.targetHandle ?? "a") === "a",
  );
  if (!input)
    throw new Error(
      "Connect the attention input before creating its sparse pattern.",
    );
  const local = newId(),
    global = newId(),
    union = newId();
  const y = Math.max(...graph.nodes.map((node) => node.position.y)) + 260;
  const make = (
    id: string,
    kind: string,
    label: string,
    x: number,
    y: number,
    params: Parameters,
  ): GraphNode => ({
    id,
    type: "architecture",
    position: { x, y },
    data: { kind, label, params, role: "attention-layout" },
  });
  graph.nodes.push(
    make(
      local,
      "attention_pattern",
      "Local attention window",
      attention.position.x - 340,
      y,
      { pattern: "local", window: 32, causal: false },
    ),
    make(
      global,
      "attention_pattern",
      "Global anchor tokens",
      attention.position.x - 340,
      y + 240,
      {
        pattern: "global",
        global_tokens: [0],
        global_stride: 0,
        causal: false,
      },
    ),
    make(
      union,
      "pattern_union",
      "Local ∪ global",
      attention.position.x,
      y + 100,
      {},
    ),
  );
  const connect = (source: string, target: string, handle = "a") => ({
    id: newId(),
    source,
    target,
    sourceHandle: "out",
    targetHandle: handle,
    type: "smoothstep",
  });
  graph.edges = graph.edges.filter(
    (edge) =>
      edge.target !== attention.id || (edge.targetHandle ?? "a") !== "b",
  );
  graph.edges.push(
    connect(input.source, local),
    connect(input.source, global),
    connect(local, union),
    connect(global, union, "b"),
    connect(union, attention.id, "b"),
  );
}
