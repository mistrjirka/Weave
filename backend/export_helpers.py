"""Named tensor operations used by exported PyTorch models.

These functions contain no graph interpreter or module factory. Their arguments
are ordinary tensors and explicit operation settings.
"""

import math

import torch
from torch.nn import functional as F

try:
    from .sparse_runtime import AttentionPattern, as_pattern
except ImportError:
    from sparse_runtime import AttentionPattern, as_pattern


def rotary_positions(x, theta=10000.0):
    width = x.shape[-1]
    positions = torch.arange(x.shape[-2], device=x.device).float()
    frequencies = theta ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1).to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * angles.cos() + rotated * angles.sin()


def fuse_memory(hidden_states, memory):
    keys, values = memory.chunk(2, dim=-1)
    width = hidden_states.shape[-1]
    normalized_hidden = F.rms_norm(hidden_states, [width], eps=1e-6)
    normalized_keys = F.rms_norm(keys, [width], eps=1e-6)
    scores = (normalized_hidden * normalized_keys).sum(-1, keepdim=True)
    gate = torch.sigmoid(scores / math.sqrt(width))
    return hidden_states + gate * values


def merge_residual_lanes(residuals, update, coefficients, lanes=4):
    post_gate = coefficients[..., lanes : 2 * lanes]
    residual_mix = coefficients[..., 2 * lanes :].reshape(
        *residuals.shape[:2], lanes, lanes
    )
    mixed = torch.einsum("btij,btjc->btic", residual_mix, residuals)
    return mixed + post_gate.unsqueeze(-1) * update.unsqueeze(-2)


def combine_patterns(first, second, *, intersection=False):
    if not isinstance(first, AttentionPattern) and not isinstance(
        second, AttentionPattern
    ):
        return first & second if intersection else first | second
    base = first if isinstance(first, AttentionPattern) else second
    children = [as_pattern(first, base.q, base.k), as_pattern(second, base.q, base.k)]
    return AttentionPattern(
        "intersection" if intersection else "union", base.q, base.k, children=children
    )


def select_tokens(
    logits,
    token_ids,
    *,
    mask_id,
    step=0,
    steps=1,
    temperature=0,
    strategy="confidence",
):
    """One discrete sampling step; train on predictor logits before this operation."""
    masked = token_ids.eq(mask_id)
    scores = logits.clone()
    scores[..., mask_id] = -torch.inf
    if temperature > 0:
        scores = scores / temperature - torch.empty_like(scores).exponential_().log()
    predictions = scores.argmax(-1)
    confidence = scores.softmax(-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    if strategy == "random":
        confidence = torch.rand_like(confidence)
    confidence = confidence.masked_fill(~masked, -torch.inf)
    result = token_ids.clone()
    for batch_index in range(token_ids.shape[0]):
        remaining = int(masked[batch_index].sum())
        count = math.ceil(remaining / max(1, steps - step))
        if count:
            positions = confidence[batch_index].topk(count).indices
            result[batch_index, positions] = predictions[batch_index, positions]
    return result
