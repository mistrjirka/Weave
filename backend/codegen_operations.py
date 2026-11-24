"""Operation-specific source emitters."""
from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class Operation:
    constructor: Callable | None = None
    expression: Callable | None = None
    runtime: tuple[str, ...] = ()
    helpers: tuple[str, ...] = ()
    sparse: tuple[str, ...] = ()


# Builders receive resolved settings, inferred input dimensions and local names.
OPERATIONS = {
    "embedding": Operation(
        constructor=lambda p, c, s: f"nn.Embedding(num_embeddings={p.get('vocab', c.get('vocab'))!r}, embedding_dim={p.get('width', c.get('width', 128))!r})"
    ),
    "linear": Operation(
        constructor=lambda p, c, s: f"nn.Linear(in_features={s[-1]!r}, out_features={p.get('out', c.get('width', 128))!r}, bias={p.get('bias', False)!r})"
    ),
    "head": Operation(
        constructor=lambda p, c, s: f"nn.Linear(in_features={s[-1]!r}, out_features={p.get('out', c.get('vocab'))!r}, bias={p.get('bias', False)!r})"
    ),
    "layer_norm": Operation(
        constructor=lambda p, c, s: f"nn.LayerNorm({s[-1]!r}, eps={p.get('eps', 1e-5)!r}, elementwise_affine={p.get('affine', True)!r}, bias={p.get('bias', True)!r})"
    ),
    "dropout": Operation(
        constructor=lambda p, c, s: f"nn.Dropout(p={p.get('probability', 0.1)!r})"
    ),
    "position_embedding": Operation(
        constructor=lambda p, c, s: f"LearnedPositions(width={s[-1]!r}, max_length={p.get('max_length', 2048)!r})",
        runtime=("LearnedPositions",),
    ),
    "conv1d": Operation(
        constructor=lambda p, c, s: f"SequenceConv1d(in_channels={s[-1]!r}, out_channels={p.get('out', s[-1])!r}, kernel_size={p.get('kernel_size', 3)!r}, stride={p.get('stride', 1)!r}, padding={p.get('padding', 0)!r}, bias={p.get('bias', True)!r})",
        runtime=("SequenceConv1d",),
    ),
    "pool1d": Operation(
        constructor=lambda p, c, s: f"SequencePool1d(kernel_size={p.get('kernel_size', 2)!r}, stride={p.get('stride', 2)!r}, mode={p.get('mode', 'max')!r})",
        runtime=("SequencePool1d",),
    ),
    "conv2d": Operation(
        constructor=lambda p, c, s: f"nn.Conv2d(in_channels={s[1]!r}, out_channels={p.get('out_channels', 32)!r}, kernel_size={p.get('kernel_size', 3)!r}, stride={p.get('stride', 1)!r}, padding={p.get('padding', 0)!r}, bias={p.get('bias', True)!r})"
    ),
    "pool2d": Operation(
        constructor=lambda p, c, s: f"nn.{('MaxPool2d' if p.get('mode', 'max') == 'max' else 'AvgPool2d')}(kernel_size={p.get('kernel_size', 2)!r}, stride={p.get('stride', 2)!r})"
    ),
    "norm": Operation(
        constructor=lambda p, c, s: f"nn.RMSNorm({s[-1]!r}, eps={p.get('eps', 1e-6)!r})"
    ),
    "ffn": Operation(
        constructor=lambda p, c, s: f"FFN(width={s[-1]!r}, hidden={p.get('hidden', s[-1]*4)!r})",
        runtime=("FFN",),
    ),
    "ngram": Operation(
        constructor=lambda p, c, s: f"NGram(orders={p.get('orders', [2,3])!r}, buckets={p.get('buckets', 8192)!r})",
        runtime=("NGram",),
    ),
    "lookup": Operation(
        constructor=lambda p, c, s: f"Lookup(orders={s[-1]!r}, buckets={p.get('buckets', 8192)!r}, width={p.get('width', c.get('width', 128))!r})",
        runtime=("Lookup",),
    ),
    "gate": Operation(constructor=lambda p, c, s: "Gate()", runtime=("Gate",)),
    "hadamard": Operation(
        constructor=lambda p, c, s: f"{'HadamardTwoStage' if p.get('variant') == 'two_stage' else 'Hadamard'}({s[-1]!r})",
        runtime=("Hadamard", "HadamardTwoStage"),
    ),
    "monarch": Operation(
        constructor=lambda p, c, s: f"Monarch(width={s[-1]!r}, split={p.get('split', False)!r})",
        runtime=("Monarch",),
    ),
    "causal_conv": Operation(
        constructor=lambda p, c, s: f"CausalConv(width={s[-1]!r}, taps={p.get('taps', 3)!r}, active_taps={p.get('active_taps')!r})",
        runtime=("CausalConv",),
    ),
    "routing": Operation(
        constructor=lambda p, c, s: f"Routing(width={s[-1]!r}, lanes={p.get('lanes', 4)!r}, layer={p.get('layer', 0)!r})",
        runtime=("Routing",),
    ),
    "engram_hash": Operation(
        constructor=lambda p, c, s: f"EngramHash(orders={p.get('orders', [2,3])!r}, heads={p.get('hash_heads', 2)!r}, slots={p.get('buckets', 8192)!r})",
        runtime=("EngramHash",),
    ),
    "engram_lookup": Operation(
        constructor=lambda p, c, s: f"EngramLookup(width={p.get('width', c.get('width', 128))!r}, orders={p.get('orders', [2,3])!r}, heads={p.get('hash_heads', 2)!r}, slots={p.get('buckets', 8192)!r}, sub_dim={p.get('sub_dim', 128)!r}, dilation={p.get('dilation', 3)!r})",
        runtime=("EngramLookup",),
    ),
    "add": Operation(expression=lambda a, p, c: f"{a[0]} + {a[1]}"),
    "subtract": Operation(expression=lambda a, p, c: f"{a[0]} - {a[1]}"),
    "multiply": Operation(expression=lambda a, p, c: f"{a[0]} * {a[1]}"),
    "concat": Operation(
        expression=lambda a, p, c: f"torch.cat(({a[0]}, {a[1]}), dim=-1)"
    ),
    "gelu": Operation(
        expression=lambda a, p, c: f"F.gelu({a[0]}, approximate={p.get('approximate', 'none')!r})"
    ),
    "relu": Operation(expression=lambda a, p, c: f"F.relu({a[0]})"),
    "silu": Operation(expression=lambda a, p, c: f"F.silu({a[0]})"),
    "sigmoid": Operation(expression=lambda a, p, c: f"torch.sigmoid({a[0]})"),
    "scale": Operation(expression=lambda a, p, c: f"{a[0]} * {p.get('factor', 1.0)!r}"),
    "softmax": Operation(
        expression=lambda a, p, c: f"{a[0]}.softmax(dim={p.get('dim', -1)!r})"
    ),
    "matmul": Operation(expression=lambda a, p, c: f"{a[0]} @ {a[1]}"),
    "transpose": Operation(
        expression=lambda a, p, c: f"{a[0]}.transpose({p.get('dim0', -2)!r}, {p.get('dim1', -1)!r})"
    ),
    "split_heads": Operation(
        expression=lambda a, p, c: f"{a[0]}.reshape({a[0]}.shape[0], {a[0]}.shape[1], {p.get('heads', c.get('heads', 8))!r}, -1).transpose(1, 2)"
    ),
    "merge_heads": Operation(
        expression=lambda a, p, c: f"{a[0]}.transpose(1, 2).contiguous().flatten(-2)"
    ),
    "repeat_heads": Operation(
        expression=lambda a, p, c: f"{a[0]}.repeat_interleave({p.get('repeats', 1)!r}, dim=1)"
    ),
    "rope": Operation(
        expression=lambda a, p, c: f"rotary_positions({a[0]}, theta={p.get('theta', 10000.0)!r}, rope_scaling={p.get('rope_scaling', 'none')!r}, rope_factor={p.get('rope_factor', 1.0)!r}, rope_original_context={p.get('rope_original_context', 32768)!r})",
        helpers=("rotary_positions",),
    ),
    "sdpa": Operation(
        expression=lambda a, p, c: f"attend({', '.join(a[:3])}, mask={a[3]}, causal={p.get('causal', True)!r}, window={p.get('window', 0)!r}, backend={p.get('backend', 'dense')!r}, dropout_p={p.get('attention_dropout', 0.0)!r} if self.training else 0.0)",
        sparse=("attend",),
    ),
    "attention_pattern": Operation(
        expression=lambda a, p, c: f"AttentionPattern({p.get('pattern', 'local')!r}, {a[0]}, {a[1]}, p={p!r})",
        sparse=("AttentionPattern",),
    ),
    "pattern_union": Operation(
        expression=lambda a, p, c: f"combine_patterns({a[0]}, {a[1]})",
        helpers=("combine_patterns",),
    ),
    "pattern_intersection": Operation(
        expression=lambda a, p, c: f"combine_patterns({a[0]}, {a[1]}, intersection=True)",
        helpers=("combine_patterns",),
    ),
    "lane_fork": Operation(
        expression=lambda a, p, c: f"({', '.join([a[0]] * p.get('lanes', 4))},)"
    ),
    "lane_split": Operation(expression=lambda a, p, c: f"torch.unbind({a[0]}, dim=-2)"),
    "lane_join": Operation(
        expression=lambda a, p, c: f"torch.stack(({', '.join(a)},), dim=-2)"
        + (
            {"mean": ".mean(dim=-2)", "sum": ".sum(dim=-2)"}.get(
                p.get("mode", "stack"), ""
            )
        )
    ),
    "lane_expand": Operation(
        expression=lambda a, p, c: f"{a[0]}.unsqueeze(-2).expand(*{a[0]}.shape[:-1], {p.get('lanes', 4)!r}, {a[0]}.shape[-1])"
    ),
    "lane_reduce": Operation(expression=lambda a, p, c: f"{a[0]}.mean(dim=-2)"),
    "lane_mix": Operation(
        expression=lambda a, p, c: f"torch.einsum('btn,btnc->btc', {a[1]}[..., :{p.get('lanes', 4)!r}], {a[0]})"
    ),
    "lane_merge": Operation(
        expression=lambda a, p, c: f"merge_residual_lanes({', '.join(a)}, lanes={p.get('lanes', 4)!r})",
        helpers=("merge_residual_lanes",),
    ),
    "memory_fuse": Operation(
        expression=lambda a, p, c: f"fuse_memory({a[0]}, {a[1]})",
        helpers=("fuse_memory",),
    ),
    "mask": Operation(
        expression=lambda a, p, c: f"F.pad({a[0]}, (0, {p.get('length', 0)!r}), value={p.get('mask_id', c.get('mask_id'))!r})"
    ),
    "select": Operation(
        expression=lambda a, p, c: f"select_tokens({a[0]}, {a[1]}, mask_id={p.get('mask_id', c.get('mask_id'))!r}, step=step, steps=steps, temperature={p.get('temperature', 0)!r}, strategy={p.get('strategy', 'confidence')!r})",
        helpers=("select_tokens",),
    ),
}


def attention_constructor(p, config, shape, *, cross=False):
    width = shape[-1]
    heads = p.get("heads", config.get("heads", 8))
    settings = {
        "width": width,
        "heads": heads,
        "kv_heads": p.get("kv_heads", heads),
        "head_dim": p.get("head_dim"),
        "value_dim": p.get("value_dim"),
        "causal": p.get("causal", not cross),
        "window": p.get("window", 0),
        "rope": p.get("rope", True),
        "theta": p.get("theta", 10000.0),
        "qk_norm": p.get("qk_norm", False),
        "qk_norm_eps": p.get("qk_norm_eps", 1e-6),
        "qkv_bias": p.get("qkv_bias", False),
        "output_bias": p.get("output_bias", False),
        "attention_dropout": p.get("attention_dropout", 0.0),
        "conv_respects_window": p.get("conv_respects_window", False),
        "rope_scaling": p.get("rope_scaling", "none"),
        "rope_factor": p.get("rope_factor", 1.0),
        "rope_original_context": p.get("rope_original_context", 32768),
        "gated": p.get("gated", False),
        "conv_taps": p.get("conv_taps", 0),
        "backend": p.get("backend", "dense"),
    }
    defaults = {
        "head_dim": None,
        "value_dim": None,
        "causal": True,
        "window": 0,
        "rope": True,
        "theta": 10000.0,
        "qk_norm": False,
        "qk_norm_eps": 1e-6,
        "qkv_bias": False,
        "output_bias": False,
        "attention_dropout": 0.0,
        "conv_respects_window": False,
        "rope_scaling": "none",
        "rope_factor": 1.0,
        "rope_original_context": 32768,
        "gated": False,
        "conv_taps": 0,
        "backend": "dense",
    }
    arguments = [
        f"{key}={value!r}"
        for key, value in settings.items()
        if key not in defaults or value != defaults[key]
    ]
    return "Attention(" + ", ".join(arguments) + ")"


OPERATIONS["attention"] = Operation(
    constructor=attention_constructor, runtime=("Attention",)
)
OPERATIONS["cross_attention"] = Operation(
    constructor=lambda p, c, s: attention_constructor(p, c, s, cross=True),
    runtime=("Attention",),
)
