"""Torch counterpart of public-only v13 enemy attention.

Inputs: [batch, enemy_slot, hidden], valid-enemy mask [batch, enemy_slot].
Targets continue to use instance-slot pointers from the original v8 codec.
"""
from __future__ import annotations

import math
from typing import Any

from sts2_ai.models.enemy_attention import ATTENTION_HEADS


def add_attention_parameters(params: dict[str, Any], hidden: int, torch: Any) -> None:
    if hidden <= 0 or hidden % ATTENTION_HEADS:
        raise ValueError("Attention hidden size must be a multiple of four")
    scale = 1.0 / math.sqrt(hidden)
    for name in ("attn_query", "attn_key", "attn_value", "attn_output"):
        params[name] = (torch.randn(hidden, hidden) * scale).requires_grad_()
    # Zero gate guarantees function-identical starting predictions to v8.
    # Random Q/K/V/O let the gate receive a useful gradient on the first update.
    params["attn_gate"] = torch.zeros(hidden, requires_grad=True)


def contextualize_torch(
    embeddings: Any, mask: Any, params: dict[str, Any], torch: Any,
) -> Any:
    if "attn_gate" not in params:
        return embeddings  # original v8 path
    batch, count, width = embeddings.shape
    if width % ATTENTION_HEADS:
        raise ValueError("Attention hidden width must be divisible by four")
    size = width // ATTENTION_HEADS
    linear = torch.nn.functional.linear

    def heads(tensor: Any) -> Any:
        return tensor.reshape(batch, count, ATTENTION_HEADS, size).permute(0, 2, 1, 3)

    q = heads(linear(embeddings, params["attn_query"]))
    k = heads(linear(embeddings, params["attn_key"]))
    v = heads(linear(embeddings, params["attn_value"]))
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
    updates = linear(mixed, params["attn_output"])
    return embeddings + params["attn_gate"] * updates * mask.unsqueeze(-1)
