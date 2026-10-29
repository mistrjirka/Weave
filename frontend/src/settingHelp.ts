/** Plain-language explanations for settings used by attention and transformer blocks. */
export const attentionSettingHelp: Record<string, string> = {
  heads:
    "How many query heads look for information in parallel. For example, 8 heads can learn 8 different ways to look at the surrounding tokens.",
  kv_heads:
    "How many key/value heads provide information to query heads. MHA uses the same number as query heads; GQA shares them between groups; MQA uses one. The query-head count must be divisible by this number.",
  causal:
    "When on, each token can look at itself and earlier tokens, but not future ones. Use this for a model that predicts the next token. Turn it off when every token may look in both directions.",
  rope: "Rotary position embeddings (RoPE) give attention information about token positions by rotating its queries and keys. They help the model distinguish token order; they do not add a learned embedding table.",
  qk_norm:
    "Query/key normalization adjusts the size of queries and keys before attention scores are computed. It can help keep those scores and training more stable; it is a design choice, not required for every model.",
  gated:
    "Adds a learned gate to the attention output. The gate lets the layer control how much attention information passes through. It adds trainable parameters; it is separate from a gated residual connection.",
  window:
    "Limits attention to nearby tokens. For causal attention, a window of 32 lets a token see itself and up to 31 earlier tokens. Zero means no local window; other attention masks can still restrict visibility.",
  backend:
    "Chooses how attention is computed, not what the layer is meant to learn. Dense works on CPU or GPU; flex uses PyTorch FlexAttention and requires a compatible GPU; auto chooses a supported path. Actual speed depends on your hardware and mask.",
  theta:
    "Controls the range of frequencies used by rotary positions. It affects how positions are represented; the right value depends on the model and its training context.",
  rope_scaling:
    "An optional change to rotary positions, such as YaRN, used by some models for longer contexts. It changes the computation and does not by itself guarantee that a model works beyond its training length.",
  rope_factor:
    "The scaling factor used by YaRN. Use the value expected by the model and its training setup.",
  rope_original_context:
    "The original context length used to calculate YaRN scaling. It is not a guarantee of the usable context length.",
  qk_norm_eps:
    "A small number used by query/key normalization to avoid division by zero. It matters only when query/key normalization is enabled.",
  qkv_bias:
    "Adds learned offsets to the query, key, and value projections. Some architectures use them and others leave them out.",
  output_bias: "Adds a learned offset to the output projection of attention.",
  attention_dropout:
    "Randomly drops some attention weights during training as regularization. It is disabled automatically during evaluation.",
  conv_taps:
    "Number of learned causal convolution taps applied to queries, keys, and values before attention. Zero turns this convolution off.",
  conv_respects_window:
    "When enabled, convolution taps outside a local attention window are ignored. This matters only when both convolution taps and a window are enabled.",
  head_dim:
    "Number of features in each query and key head. If not set, it is normally model width divided by the number of query heads.",
  value_dim:
    "Number of features in each value head. It can be different from the query/key head width in some architectures.",
  ffn_width:
    "The number of features inside the feed-forward network. The layer expands to this size, processes each token, and projects back to the model width.",
};

const common: Record<string, string> = {
  heads:
    "The number of attention heads used when splitting the feature dimension. The width must be divisible by this number.",
  causal: "Blocks attention to future positions when enabled.",
  layer: "The layer index used to initialize lane routing preferences.",
  index:
    "The order of this input or output in the generated model's function. Index 0 is the first argument or output.",
  shape:
    "The tensor dimensions, such as batch B, sequence length T, and feature width. A $name refers to a graph variable.",
  dtype:
    "The kind of data in the tensor. Token IDs usually use int64; learned features usually use float32.",
  output_dtype:
    "The kind of data the custom block says it returns. Weave uses this declaration for shape checking.",
  input_shape:
    "The expected dimensions of the first input to your Python block. Weave checks connections against this declaration.",
  input_shapes:
    "The expected dimensions of all inputs to your Python block, in argument order. When provided, this replaces the single-input shape.",
  output_shape:
    "The dimensions your Python block promises to return. Weave uses this declaration when checking downstream connections.",
  parameter_count:
    "An estimate of the trainable parameters inside your custom Python block, used in the graph's displayed total.",
  kwargs: "Named arguments passed when constructing your custom Python block.",
  source:
    "Python code defining the custom PyTorch module. Its forward method should match the declared inputs and output.",
  optional:
    "When enabled, this input may be left unconnected. The block must support a missing input.",
  width:
    "The number of features in each output vector. Connected blocks need compatible feature widths.",
  vocab:
    "Number of rows in the token embedding table. It must cover every token ID, including added tokens.",
  out: "The number of output features. For a vocabulary head, it is the number of token scores produced at each position.",
  bias: "Adds a learned offset to the output. Some architectures use a bias and others omit it.",
  eps: "A small positive number added during normalization to avoid division by zero.",
  hidden:
    "The intermediate feature width inside this feed-forward layer. It projects back to the input width.",
  probability:
    "The fraction of values randomly dropped during training. Evaluation turns dropout off.",
  affine:
    "Adds learned scale and, optionally, bias parameters after normalization.",
  max_length:
    "The number of positions in the learned position table. Inputs longer than this table cannot be processed by this block.",
  kernel_size:
    "How many neighboring positions the filter or pooling operation looks at in each spatial dimension.",
  stride:
    "How far the filter moves between outputs. Larger strides reduce the output length or image size.",
  padding:
    "How many zero values are added around the input before convolution. Padding can preserve the output size.",
  out_channels:
    "The number of feature channels produced by this image convolution.",
  mode: "Choose the way pooling summarizes each region: max takes the largest value; average takes the mean.",
  approximate:
    "Choose exact GELU or its tanh approximation. Some model architectures specify which one to use.",
  iterations:
    "How many times this container applies its internal graph. Repeating a layer adds computation on each pass.",
  shared:
    "When on, each iteration reuses the same learned weights. When off, each iteration has its own independently learned weights.",
  experts:
    "The number of expert networks available to the router. Open the block to edit their graphs.",
  top_k:
    "The number of choices kept for each token. For an expert router, it is how many experts process each token; for a content pattern, it is how many keys are selected.",
  balance_weight:
    "Strength of the optional expert-balancing loss. Zero disables it; training code must add the auxiliary loss to the task loss.",
  orders:
    "Lengths of the token windows to hash. For example, 2 and 3 represent pairs and triples of neighboring tokens.",
  buckets:
    "Number of entries in each hash table. Different token windows can map to the same bucket.",
  hash_heads: "Number of independent hashes used for each n-gram order.",
  sub_dim:
    "Feature width stored in each n-gram table before the results are combined.",
  dilation:
    "Spacing between positions used by the memory convolution. A value of 3 looks back in steps of three tokens.",
  lanes: "The number of parallel residual streams carried through the block.",
  factor:
    "A fixed multiplier applied to every input value. It changes magnitude but not the tensor shape.",
  length: "How many mask tokens are appended to the input sequence.",
  mask_id:
    "The token ID used for a masked or unknown position. It must belong to the selected tokenizer's vocabulary.",
  strategy:
    "How positions are selected during denoising: confidence favors the most certain predictions, while random chooses positions randomly.",
  temperature:
    "Controls randomness in token selection. Zero uses the highest-scoring token; positive values add randomness.",
  pattern:
    "Choose the rule defining which token pairs are allowed to attend to each other.",
  global_tokens:
    "Positions that are allowed to connect globally in this attention pattern.",
  global_stride:
    "Optionally mark every Nth position as global. Zero disables periodic global positions.",
  block_size: "Number of positions in each attention block.",
  neighbor_blocks: "How many adjacent blocks on each side may be attended to.",
  density:
    "Approximate fraction of token pairs kept in a random attention pattern; self-connections are also kept.",
  seed: "A number that makes the random attention pattern reproducible.",
  repeats:
    "How many times each key/value head is repeated to match the number of query heads.",
  taps: "Number of learned causal filter positions, including the current token. Three taps can look at this token and two earlier positions.",
  active_taps:
    "How many of the stored filter taps actually participate. Inactive taps remain stored but do not affect the output.",
  dim: "The tensor axis used by this operation. Negative axes count backward from the end; -1 means the last axis.",
  dim0: "The first tensor axis to swap. Negative axes count backward from the end.",
  dim1: "The second tensor axis to swap. Negative axes count backward from the end.",
  tie: "The embedding node whose weights are reused for this output head. Dimensions must match.",
  split:
    "Use separate fixed permutations for the two halves of the mixer. It changes the mixer's computation.",
};

const byKind: Record<string, Record<string, string>> = {
  attention_pattern: {
    window:
      "The distance covered by the local attention pattern. With causal masking, a window of 16 includes the current token and up to 15 earlier tokens.",
    causal:
      "When enabled, this pattern also blocks connections to future tokens.",
  },
  input: { index: common.index, shape: common.shape, dtype: common.dtype },
  output: { index: common.index },
  head: { out: common.out, tie: common.tie },
  scale: { factor: common.factor },
};

export function settingHelp(kind: string, name: string): string | undefined {
  const attention = [
    "attention",
    "cross_attention",
    "block",
    "loop",
    "sdpa",
    "rope",
  ].includes(kind);
  return (
    byKind[kind]?.[name] ??
    (attention ? attentionSettingHelp[name] : undefined) ??
    common[name]
  );
}

/** Inactive settings remain visible so their saved values are not lost. */
export function generalFieldDisabledReason(
  kind: string,
  key: string,
  settings: Record<string, unknown>,
): string | undefined {
  if (kind === "layer_norm" && key === "bias" && settings.affine === false)
    return "Enable learned affine parameters to use a bias.";
  return undefined;
}
