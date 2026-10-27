"""Portable permutation-equivariant enemy-to-enemy attention (v13).

Four heads attend to OTHER living enemies (no self edge). An empty or
singleton enemy set receives exactly zero relational update. The learned
residual gate starts at zero so a v8 combat policy can be extended without
changing a single initial action logit or value estimate.
"""
from __future__ import annotations

import math
from typing import Any, TypeVar

_Key = TypeVar("_Key", int, str)

ATTENTION_HEADS = 4
ATTENTION_FIELDS = (
    "attn_query", "attn_key", "attn_value", "attn_output", "attn_gate",
)


def contextualize_python(
    embeddings: dict[_Key, list[float]],
    params: dict[str, Any], *, prefix: str = "attn_", heads: int = ATTENTION_HEADS,
) -> dict[_Key, list[float]]:
    """Use IDs only as mapping keys; never embed or order them numerically."""
    if len(embeddings) < 2:
        return embeddings
    ids = list(embeddings)
    width = len(embeddings[ids[0]])
    if heads <= 0 or width % heads:
        raise ValueError("Attention hidden width must be divisible by head count")
    depth = width // heads

    def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]

    queries = {i: matvec(params[f"{prefix}query"], embeddings[i]) for i in ids}
    keys = {i: matvec(params[f"{prefix}key"], embeddings[i]) for i in ids}
    values = {i: matvec(params[f"{prefix}value"], embeddings[i]) for i in ids}
    result: dict[_Key, list[float]] = {}
    for i in ids:
        joined: list[float] = []
        for head in range(heads):
            start = head * depth
            stop = start + depth
            opponents = [j for j in ids if j != i]
            scores = [
                sum(queries[i][k] * keys[j][k] for k in range(start, stop))
                / math.sqrt(depth)
                for j in opponents
            ]
            maximum = max(scores)
            exponents = [math.exp(score - maximum) for score in scores]
            denominator = sum(exponents)
            probabilities = [value / denominator for value in exponents]
            joined.extend(
                sum(prob * values[j][k] for prob, j in zip(probabilities, opponents,
                                                           strict=True))
                for k in range(start, stop)
            )
        update = matvec(params[f"{prefix}output"], joined)
        result[i] = [
            embeddings[i][k] + params[f"{prefix}gate"][k] * update[k]
            for k in range(width)
        ]
    return result


DEEP_MACRO_FIELDS = (
    "attn2_query", "attn2_key", "attn2_value", "attn2_output", "attn2_gate",
    "macro_ffn_in", "macro_ffn_in_bias", "macro_ffn_out",
    "macro_ffn_out_bias", "macro_ffn_gate",
)


def contextualize_deep_python(
    embeddings: dict[str, list[float]], params: dict[str, Any],
) -> dict[str, list[float]]:
    """Portable second eight-head attention and width-4 gated feedforward."""
    updated = contextualize_python(embeddings, params, prefix="attn2_", heads=8)
    result: dict[str, list[float]] = {}
    for key, values in updated.items():
        middle = [
            max(0.0, bias + sum(a * b for a, b in zip(row, values, strict=True)))
            for row, bias in zip(params["macro_ffn_in"],
                                 params["macro_ffn_in_bias"], strict=True)
        ]
        residual = [
            bias + sum(a * b for a, b in zip(row, middle, strict=True))
            for row, bias in zip(params["macro_ffn_out"],
                                 params["macro_ffn_out_bias"], strict=True)
        ]
        result[key] = [
            value + gate * delta
            for value, gate, delta in zip(
                values, params["macro_ffn_gate"], residual, strict=True,
            )
        ]
    return result
