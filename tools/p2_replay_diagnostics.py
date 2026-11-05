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


def classify_trace_difference(
    expected: dict[str, Any], actual: dict[str, Any],
) -> dict[str, Any]:
    """Separate decision/teacher drift from stochastic search-work accounting.

    This is a *diagnosis*, never an acceptance or tolerance rule: callers
    must still reject any mismatched trace, including a one-step work drift.
    """
    first = first_trace_difference(expected, actual)
    if first is None:
        return {
            "kind": "exact_replay",
            "semantic_parity": True,
            "search_work_parity": True,
            "first_difference": None,
        }
    semantic_keys = ("actions", "final_public_frame", "outcome", "searches")
    semantic_differences = {
        key: first_trace_difference(
            expected.get(key), actual.get(key), f"$.{key}",
        )
        for key in semantic_keys
        if expected.get(key) != actual.get(key)
    }
    semantic_parity = not semantic_differences
    work_difference = first_trace_difference(
        expected.get("transitions"), actual.get("transitions"),
        "$.transitions",
    )
    work_parity = work_difference is None
    result: dict[str, Any] = {
        "kind": (
            "search_work_only" if semantic_parity and not work_parity
            else "semantic_or_teacher_drift"
        ),
        "semantic_parity": semantic_parity,
        "search_work_parity": work_parity,
        "first_difference": first,
        "semantic_differences": semantic_differences,
        "search_work_difference": work_difference,
        "action_counts": {
            "expected": len(expected.get("actions", ())),
            "actual": len(actual.get("actions", ())),
        },
    }
    if type(expected.get("transitions")) is int and type(actual.get("transitions")) is int:
        before, after = expected["transitions"], actual["transitions"]
        result["search_work"] = {
            "expected_transitions": before,
            "actual_transitions": after,
            "absolute_drift": abs(after - before),
            "relative_drift": abs(after - before) / max(before, after, 1),
        }
    return result
