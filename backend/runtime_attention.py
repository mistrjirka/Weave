"""Portable attention implementation for generated models."""
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .sparse_runtime import AttentionPattern, as_pattern, attend
except ImportError:
    from sparse_runtime import AttentionPattern, as_pattern, attend


def shift(x, offset):
    if not offset:return x
    return torch.cat((torch.zeros_like(x[:,:min(offset,x.shape[1])]),x[:,:max(0,x.shape[1]-offset)]),dim=1)


class Attention(nn.Module):
    def __init__(
        self,
        width,
        heads,
        kv_heads,
        head_dim=None,
        value_dim=None,
        causal=True,
        window=0,
        rope=True,
        theta=10000.0,
        qk_norm=False,
        gated=False,
        conv_taps=0,
        backend="dense",
        qk_norm_eps=1e-6,
        qkv_bias=False,
        rope_scaling="none",
        rope_factor=1.0,
        rope_original_context=32768,
        output_bias=False,
        attention_dropout=0.0,
        conv_respects_window=False,
    ):
        super().__init__()
        self.h, self.kv = heads, kv_heads
        self.hd, self.vd = head_dim or width // heads, value_dim or width // heads
        self.causal, self.window, self.rope, self.theta = causal, window, rope, theta
        self.attention_dropout = attention_dropout
        self.conv_respects_window = conv_respects_window
        self.backend = backend
        # Transient tables: never serialize position caches into checkpoints.
        self._rope_cache = None
        self.rope_scaling, self.rope_factor, self.rope_original_context = (
            rope_scaling,
            rope_factor,
            rope_original_context,
        )
        self.q = nn.Linear(width, heads * self.hd, bias=qkv_bias)
        self.k = nn.Linear(width, kv_heads * self.hd, bias=qkv_bias)
        self.v = nn.Linear(width, kv_heads * self.vd, bias=qkv_bias)
        self.o = nn.Linear(heads * self.vd, width, bias=output_bias)
        self.q_norm = nn.RMSNorm(self.hd, eps=qk_norm_eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd, eps=qk_norm_eps) if qk_norm else nn.Identity()
        self.gate = nn.Linear(width, heads * self.vd, bias=False) if gated else None
        self.taps = nn.ParameterList()
        for d in (
            [heads * self.hd, kv_heads * self.hd, kv_heads * self.vd]
            if conv_taps
            else []
        ):
            taps = torch.zeros(conv_taps, d)
            taps[0] = 1
            self.taps.append(nn.Parameter(taps))

    def rotate(self, x):
        key = (
            x.device,
            x.dtype,
            x.shape[-2],
            self.hd,
            self.theta,
            self.rope_scaling,
            self.rope_factor,
            self.rope_original_context,
        )
        cached = None if torch.compiler.is_compiling() else self._rope_cache
        if cached is not None and cached[0] == key:
            cos, sin, scale = cached[1:]
            a, b = x.chunk(2, dim=-1)
            return scale * (x * cos + torch.cat((-b, a), -1) * sin)
        pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        inv = self.theta ** (
            -torch.arange(0, self.hd, 2, device=x.device).float() / self.hd
        )
        scale = 1.0
        if self.rope_scaling == "yarn":

            def correction(rotations):
                return (
                    self.hd
                    * math.log(self.rope_original_context / (rotations * 2 * math.pi))
                    / (2 * math.log(self.theta))
                )

            low, high = max(math.floor(correction(32)), 0), min(
                math.ceil(correction(1)), self.hd - 1
            )
            if low == high:
                high += 0.001
            ramp = (
                (torch.arange(self.hd // 2, device=x.device, dtype=torch.float32) - low)
                / (high - low)
            ).clamp(0, 1)
            inv = inv * ((1 - ramp) + ramp / self.rope_factor)
            scale = 1 + 0.1 * math.log(self.rope_factor)
        freq = torch.outer(pos, inv)
        # Compute trigonometry before casting: FP16 cannot represent every
        # integer position beyond 2048 and would rotate long sequences wrongly.
        freq = torch.cat((freq, freq), -1)
        cos, sin = freq.cos().to(x.dtype), freq.sin().to(x.dtype)
        # Compiled graphs specialize on shape; avoid mutating a Python cache
        # while Dynamo is tracing or replaying a graph.
        if not torch.compiler.is_compiling():
            self._rope_cache = (key, cos, sin, scale)
        a, b = x.chunk(2, dim=-1)
        return scale * (x * cos + torch.cat((-b, a), -1) * sin)

    def _apply(self, fn, recurse=True):
        self._rope_cache = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, x, memory=None, mask=None, **_):
        memory = x if memory is None else memory
        b, t, _ = x.shape
        s = memory.shape[1]
        q, k, v = self.q(x), self.k(memory), self.v(memory)
        if self.taps:
            q, k, v = [
                sum(
                    taps[j] * shift(z, j)
                    for j in range(
                        min(taps.shape[0], self.window)
                        if self.conv_respects_window and self.window
                        else taps.shape[0]
                    )
                )
                for z, taps in zip([q, k, v], self.taps)
            ]
        q = self.q_norm(q.view(b, t, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(k.view(b, s, self.kv, self.hd)).transpose(1, 2)
        v = v.view(b, s, self.kv, self.vd).transpose(1, 2)
        if self.rope:
            q, k = self.rotate(q), self.rotate(k)
        if self.h != self.kv:
            k = k.repeat_interleave(self.h // self.kv, dim=1)
            v = v.repeat_interleave(self.h // self.kv, dim=1)
        y = attend(
            q,
            k,
            v,
            mask,
            self.causal,
            self.window,
            self.backend,
            self.attention_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return self.o(y)
