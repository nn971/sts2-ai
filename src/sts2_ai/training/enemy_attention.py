"""Torch counterpart of public-only v13 enemy attention.

Inputs: [batch, enemy_slot, hidden], valid-enemy mask [batch, enemy_slot].
Targets continue to use instance-slot pointers from the original v8 codec.
"""
from __future__ import annotations

import math
from typing import Any

from sts2_ai.models.enemy_attention import ATTENTION_HEADS


def add_attention_parameters(
    params: dict[str, Any], hidden: int, torch: Any, *,
    prefix: str = "attn_", heads: int = ATTENTION_HEADS,
) -> None:
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("Attention hidden size must be divisible by head count")
    scale = 1.0 / math.sqrt(hidden)
    for name in ("query", "key", "value", "output"):
        params[f"{prefix}{name}"] = (
            torch.randn(hidden, hidden) * scale
        ).requires_grad_()
    # Zero gate lets a newly stacked attention layer begin as an identity.
    params[f"{prefix}gate"] = torch.zeros(hidden, requires_grad=True)


def contextualize_torch(
    embeddings: Any, mask: Any, params: dict[str, Any], torch: Any, *,
    prefix: str = "attn_", heads: int = ATTENTION_HEADS,
) -> Any:
    if f"{prefix}gate" not in params:
        return embeddings  # original v8 path
    batch, count, width = embeddings.shape
    if width % heads:
        raise ValueError("Attention hidden width must be divisible by head count")
    size = width // heads
    linear = torch.nn.functional.linear

    def heads(tensor: Any) -> Any:
        return tensor.reshape(batch, count, heads, size).permute(0, 2, 1, 3)

    q = heads(linear(embeddings, params[f"{prefix}query"]))
    k = heads(linear(embeddings, params[f"{prefix}key"]))
    v = heads(linear(embeddings, params[f"{prefix}value"]))
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(size)
    # Mask both queries and keys, and exclude the diagonal. A masked row
    # has no neighbors (e.g. 0/1 enemies), so its output must be exactly zero.
    valid = (
        mask[:, None, :, None] & mask[:, None, None, :]
        & ~torch.eye(count, dtype=torch.bool, device=mask.device)[None, None, :, :]
    )
    probabilities = torch.softmax(scores.masked_fill(~valid, -1e9), dim=-1)
    probabilities = probabilities * valid
    attended = torch.matmul(probabilities, v)
    mixed = attended.permute(0, 2, 1, 3).reshape(batch, count, width)
    updates = linear(mixed, params[f"{prefix}output"])
    return embeddings + params[f"{prefix}gate"] * updates * mask.unsqueeze(-1)
