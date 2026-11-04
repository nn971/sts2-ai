"""Pure Python diagnostics for exact public/player/PUCT trace replay drift.

These helpers identify the *first* divergent trace location. The benchmark
still fails closed on every mismatch, including floating-point differences.
"""
from __future__ import annotations

import math
from typing import Any


def first_trace_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    """Return the first exact mismatch with a compact type-aware explanation."""
    if expected == actual:
        return None
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            return {
                "path": path, "kind": "different_keys",
                "expected_only": missing[:12], "actual_only": extra[:12],
            }
        for key in sorted(expected):
            difference = first_trace_difference(
                expected[key], actual[key], f"{path}.{key}",
            )
            if difference is not None:
                return difference
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return {
                "path": path, "kind": "different_list_length",
                "expected": len(expected), "actual": len(actual),
            }
        for index, (before, after) in enumerate(zip(expected, actual, strict=True)):
            difference = first_trace_difference(
                before, after, f"{path}[{index}]",
            )
            if difference is not None:
                return difference
    elif (
        type(expected) in (float, int)
        and type(actual) in (float, int)
        and math.isfinite(expected)
        and math.isfinite(actual)
    ):
        return {
            "path": path, "kind": "numeric_difference",
            "expected": expected, "actual": actual,
            "absolute_difference": abs(actual - expected),
            "relative_difference": abs(actual - expected) / max(
                abs(actual), abs(expected), 1e-12,
            ),
        }
    return {
        "path": path, "kind": "different_value",
        "expected": repr(expected)[:300], "actual": repr(actual)[:300],
    }
