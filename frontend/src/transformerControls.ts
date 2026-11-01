import type {
  Graph,
  GraphNode,
  NodeData,
  Parameters,
  JsonValue,
} from "./types";

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
  "qkv_bias",
  "output_bias",
  "attention_dropout",
  "qk_norm_eps",
  "rope_scaling",
  "rope_factor",
  "rope_original_context",
  "conv_respects_window",
  "ffn_width",
]);
/** Only a single inner transformer can be controlled by a wrapper inspector. */
export function wrappedTransformer(data: NodeData): GraphNode | undefined {
  const layers = data.body?.nodes.filter(
    (node) => node.data.kind !== "input" && node.data.kind !== "output",
  );
  return layers?.length === 1 &&
    layers[0].data.kind === "block" &&
    layers[0].data.role === "transformer"
    ? layers[0]
    : undefined;
}
/** The inner layer owns transformer settings; the loop owns repetition only. */
function moveWrapperSettings(data: NodeData) {
  const child = wrappedTransformer(data);
  if (!child) return;
  for (const key of transformerSettingKeys) {
    const value = data.params[key];
    if (value === undefined) continue;
    if (
      child.data.params[key] === undefined ||
      child.data.params[key] === "$" + key
    )
      child.data.params[key] = data.body?.variables?.[key] ?? value;
    delete data.params[key];
  }
}
function transformerGraph(data: NodeData): Graph | undefined {
  const child = wrappedTransformer(data);
  return child ? transformerGraph(child.data) : data.body;
}
function bodyScope(data: NodeData, scope: Parameters): Parameters {
  return {
    ...scope,
    ...Object.fromEntries(
      Object.entries(data.params).map(([key, value]) => [
        key,
        typeof value === "string" && value.startsWith("$")
          ? (scope[value.slice(1)] ?? value)
          : value,
      ]),
    ),
    ...data.body?.variables,
  };
}

/** Add a navigable boundary without adding operations or changing parameters. */
export function wrapLoopTransformer(data: NodeData) {
  if (data.kind !== "loop" || data.role !== "transformer" || !data.body) return;
  const body = data.body;
  const inputs = structuredClone(
    body.nodes.filter((n) => n.data.kind === "input"),
  );
  const outputs = structuredClone(
    body.nodes.filter((n) => n.data.kind === "output"),
  );
  let id = "transformer_layer";
  while ([...inputs, ...outputs].some((node) => node.id === id)) id += "_";
  inputs.forEach((node, index) => {
    node.position = { x: 0, y: 120 + index * 240 };
  });
  outputs.forEach((node, index) => {
    node.position = { x: 680, y: 120 + index * 240 };
  });
  data.body = {
    nodes: [
      ...inputs,
      {
        id,
        type: "architecture",
        position: { x: 340, y: 120 },
        data: {
          kind: "block",
          role: "transformer",
          label: "Transformer layer",
          params: {},
          body,
        },
      },
      ...outputs,
    ],
    edges: [
      ...inputs.map((node) => ({
        id: `${node.id}-${id}`,
        source: node.id,
        sourceHandle: "out",
        target: id,
        targetHandle: String.fromCharCode(
          97 + Number(node.data.params.index ?? 0),
        ),
        type: "smoothstep",
      })),
      ...outputs.map((node) => ({
        id: `${id}-${node.id}`,
        source: id,
        sourceHandle: "out",
        target: node.id,
        targetHandle: "a",
        type: "smoothstep",
      })),
    ],
  };
  moveWrapperSettings(data);
}

export function transformerAttention(data: NodeData): GraphNode | undefined {
  const nodes = transformerGraph(data)?.nodes ?? [];
  const attention = nodes.filter((node) => node.data.kind === "attention");
  return attention.length === 1 &&
    !nodes.some((node) => !!node.data.body) &&
    nodes.filter((node) => node.data.kind === "ffn").length <= 1
    ? attention[0]
    : undefined;
}
export function transformerSettings(data: NodeData): Parameters | null {
  const child = wrappedTransformer(data);
  if (child) {
    const inner = transformerSettings(child.data);
    if (!inner) return null;
    const scope = bodyScope(data, {});
    return {
      ...data.params,
      ...Object.fromEntries(
        Object.entries(inner).map(([key, value]) => [
          key,
          typeof value === "string" && value.startsWith("$")
            ? (scope[value.slice(1)] ?? value)
            : value,
        ]),
      ),
    };
  }
  const attention = transformerAttention(data);
  if (!attention) return null;
  const settings: Parameters = { ...defaults, ...data.params };
  // Read what the operation actually uses. An inner literal is an intentional
  // override, not a stale copy of an outer control.
  for (const key of transformerSettingKeys) {
    const value = attention.data.params[key];
    if (value !== undefined)
      settings[key] =
        value === "$" + key
          ? (data.body?.variables?.[key] ?? data.params[key] ?? value)
          : value;
  }
  const ffn = data.body?.nodes.find((node) => node.data.kind === "ffn");
  if (ffn) {
    const hidden = ffn.data.params.hidden ?? 256;
    settings.ffn_width =
      hidden === "$ffn_width"
        ? (data.body?.variables?.ffn_width ?? data.params.ffn_width ?? hidden)
        : hidden;
  } else delete settings.ffn_width;
  return settings;
}

export function exposeTransformerSettings(data: NodeData) {
  const child = wrappedTransformer(data);
  if (child) {
    moveWrapperSettings(data);
    exposeTransformerSettings(child.data);
    return;
  }
  const attention = transformerAttention(data),
    settings = transformerSettings(data);
  if (!attention || !settings) return;
  for (const key of transformerSettingKeys) {
    const value = settings[key];
    if (value === undefined) continue;
    data.params[key] = value;
    if (key === "ffn_width") {
      for (const node of transformerGraph(data)?.nodes ?? [])
        if (node.data.kind === "ffn") node.data.params.hidden = "$ffn_width";
    } else attention.data.params[key] = "$" + key;
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
  if (
    transformerGraph(data)?.nodes.some(
      (node) => node.data.role === "attention-layout",
    )
  )
    return "sparse";
  if (number(settings.window, scope, 0) > 0) return "local";
  const heads = number(settings.heads, scope, 8),
    kv = number(settings.kv_heads, scope, heads);
  return heads === kv ? "mha" : kv === 1 ? "mqa" : "gqa";
}
/** Explain fields whose value is determined or inactive in the current scope. */
export function attentionFieldDisabledReason(
  key: string,
  settings: Parameters,
  variant?: AttentionVariant | "custom",
): string | undefined {
  if (key === "kv_heads" && variant === "mha")
    return "MHA uses one key/value head per query head. Change heads to update both.";
  if (key === "kv_heads" && variant === "mqa")
    return "MQA shares one key/value head across all query heads.";
  if (
    ["theta", "rope_scaling", "rope_factor", "rope_original_context"].includes(
      key,
    ) &&
    settings.rope === false
  )
    return "Enable rotary positions to use this setting.";
  if (
    ["rope_factor", "rope_original_context"].includes(key) &&
    settings.rope_scaling !== "yarn"
  )
    return "Select YaRN rotary scaling to use this setting.";
  if (
    key === "conv_respects_window" &&
    (settings.conv_taps === 0 ||
      settings.conv_taps === undefined ||
      settings.window === 0 ||
      settings.window === undefined)
  )
    return "Enable both convolution taps and a local window to use this setting.";
  if (key === "qk_norm_eps" && settings.qk_norm !== true)
    return "Enable query/key normalization to use this setting.";
  return undefined;
}

export function setTransformerSetting(
  data: NodeData,
  key: string,
  value: JsonValue,
  scope: Parameters = {},
) {
  const child = wrappedTransformer(data);
  if (child) {
    moveWrapperSettings(data);
    setTransformerSetting(child.data, key, value, bodyScope(data, scope));
    return;
  }
  if (data.body?.variables && Object.hasOwn(data.body.variables, key))
    throw new Error(
      `${key} is overridden by a local graph variable. Edit it inside this block.`,
    );
  const tiedHeads = attentionVariant(data, scope) === "mha";
  data.params[key] = value;
  if (
    key === "heads" &&
    tiedHeads &&
    (typeof value === "number" || typeof value === "string")
  )
    data.params.kv_heads = value;
  const attention = transformerAttention(data);
  if (key === "ffn_width") {
    const ffn = data.body?.nodes.find((node) => node.data.kind === "ffn");
    if (ffn) ffn.data.params.hidden = "$ffn_width";
  } else if (attention) {
    attention.data.params[key] = "$" + key;
    if (key === "heads" && tiedHeads)
      attention.data.params.kv_heads = "$kv_heads";
  }
}
export function setAttentionVariant(
  data: NodeData,
  variant: AttentionVariant,
  scope: Parameters,
  newId: () => string,
) {
  const child = wrappedTransformer(data);
  if (child) {
    moveWrapperSettings(data);
    setAttentionVariant(child.data, variant, bodyScope(data, scope), newId);
    return;
  }
  if (
    ["heads", "kv_heads", "window"].some((key) =>
      Object.hasOwn(data.body?.variables ?? {}, key),
    )
  )
    throw new Error(
      "Attention settings are overridden by local graph variables. Edit them inside this block.",
    );
  const attention = transformerAttention(data);
  if (!attention || !data.body)
    throw new Error(
      "Choose a transformer with compact self-attention. Expanded Q/K/V graphs remain editable inside their block.",
    );
  const settings = transformerSettings(data)!;
  const graph = transformerGraph(data)!;
  const generated = new Set(
    graph.nodes
      .filter((node) => node.data.role === "attention-layout")
      .map((node) => node.id),
  );
  graph.nodes = graph.nodes.filter((node) => !generated.has(node.id));
  graph.edges = graph.edges.filter(
    (edge) => !generated.has(edge.source) && !generated.has(edge.target),
  );
  const heads = number(settings.heads, scope, 8);
  let kv = number(settings.kv_heads, scope, 2);
  const originalKv = kv;
  if (variant === "mha") kv = heads;
  else if (variant === "mqa") kv = 1;
  else if (
    kv < 1 ||
    kv > heads ||
    heads % kv ||
    (variant === "gqa" && (kv === 1 || kv === heads))
  ) {
    // Prefer a proper grouping when one exists; prime head counts cannot form one.
    kv = 1;
    if (Number.isSafeInteger(heads)) {
      for (
        let factor = 2;
        factor <= Math.sqrt(heads) && factor <= 65536;
        factor++
      ) {
        if (heads % factor === 0) {
          kv = heads / factor;
          break;
        }
      }
    }
  }
  data.params.kv_heads =
    variant === "mha"
      ? (settings.heads ?? kv)
      : variant === "mqa"
        ? 1
        : kv === originalKv
          ? (settings.kv_heads ?? kv)
          : kv;
  data.params.window = variant === "local" ? 32 : 0;
  attention.data.params.kv_heads = "$kv_heads";
  attention.data.params.window = "$window";
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
