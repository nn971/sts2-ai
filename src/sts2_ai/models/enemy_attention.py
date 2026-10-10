"""Portable permutation-equivariant enemy-to-enemy attention (v13).

Four heads attend to OTHER living enemies (no self edge). An empty or
singleton enemy set receives exactly zero relational update. The learned
residual gate starts at zero so a v8 combat policy can be extended without
changing a single initial action logit or value estimate.
"""
from __future__ import annotations

import math
from typing import Any

ATTENTION_HEADS = 4
ATTENTION_FIELDS = (
    "attn_query", "attn_key", "attn_value", "attn_output", "attn_gate",
)


def contextualize_python(
    embeddings: dict[int, list[float]],
    params: dict[str, Any],
) -> dict[int, list[float]]:
    """Use IDs only as mapping keys; never embed or order them numerically."""
    if len(embeddings) < 2:
        return embeddings
    ids = list(embeddings)
    width = len(embeddings[ids[0]])
    if width % ATTENTION_HEADS:
        raise ValueError("Attention hidden width must be divisible by four")
    depth = width // ATTENTION_HEADS

    def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]

    queries = {i: matvec(params["attn_query"], embeddings[i]) for i in ids}
    keys = {i: matvec(params["attn_key"], embeddings[i]) for i in ids}
    values = {i: matvec(params["attn_value"], embeddings[i]) for i in ids}
    result: dict[int, list[float]] = {}
    for i in ids:
        joined: list[float] = []
        for head in range(ATTENTION_HEADS):
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
        update = matvec(params["attn_output"], joined)
        result[i] = [
            embeddings[i][k] + params["attn_gate"][k] * update[k]
            for k in range(width)
        ]
    return result
