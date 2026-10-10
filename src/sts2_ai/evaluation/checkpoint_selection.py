"""Pure, fail-closed analysis of paired certified-Act-1 checkpoint evaluations.

Model choice uses a *selection* cohort only. An independently seeded final
cohort is reserved for reporting the selected checkpoint against round 40.
All results use public, certified Act-1 boundaries, never floor guesses.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

REPORT_SCHEMA = "sts2-certified-act1-checkpoint-evaluation-v1"
JOURNAL_SCHEMA = "sts2-certified-act1-evaluation-journal-v1"
GOAL = "native-act1-boss-v1"


def is_certified_clear(row: dict[str, Any]) -> bool:
    if row.get("episode_goal_version") != GOAL:
        raise ValueError("An evaluation used a different episode goal")
    if row.get("censored") is not False:
        raise ValueError("Cannot include censored/unknown episodes in checkpoint selection")
    if type(row.get("act1_cleared")) is not bool:
        raise ValueError("Act-1 clearance must be an explicit certified boolean")
    if row.get("outcome") not in ("victory", "defeat"):
        raise ValueError("Unexpected nonterminal evaluation outcome")
    return row["act1_cleared"] is True


def exact_mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided conditional binomial test for discordant paired outcomes."""
    if min(only_a, only_b) < 0:
        raise ValueError("Negative paired count")
    n = only_a + only_b
    if n == 0:
        return 1.0
    # Avoid large binomial coefficients by accumulating probabilities.
    # n is at most the number of seeds (usually 256 or 512).
    lower = min(only_a, only_b)
    term = 0.5 ** n
    cumulative = term
    for k in range(lower):
        term *= (n - k) / (k + 1)
        cumulative += term
    return min(1.0, 2.0 * cumulative)


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0 or wins < 0 or wins > n:
        raise ValueError("Invalid wins / sample count")
    proportion = wins / n
    factor = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / factor
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / factor
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No evaluation rows")
    wins = sum(is_certified_clear(row) for row in rows)
    progresses = [float(row["frontier_progress"]) for row in rows]
    if not all(math.isfinite(x) for x in progresses):
        raise ValueError("Nonfinite frontier progress")
    bosses = [row["boss_progress"] for row in rows if row.get("boss_progress") is not None]
    for boss in bosses:
        if boss["initial_hp"] <= 0 or boss["remaining_hp"] < 0:
            raise ValueError("Malformed boss progress")
    return {
        "evaluated": len(rows),
        "act1_clears": wins,
        "act1_clear_rate": wins / len(rows),
        "wilson_95": wilson_interval(wins, len(rows)),
        "mean_frontier_progress": statistics.fmean(progresses),
        "boss_encounters": len(bosses),
        "boss_conversions": sum(is_certified_clear(row) for row in rows
                                if row.get("boss_progress") is not None),
        "near_misses_80pct_boss_hp_removed": sum(
            row.get("boss_progress") is not None
            and not is_certified_clear(row)
            and row["boss_progress"]["remaining_hp"] * 5
                <= row["boss_progress"]["initial_hp"]
            for row in rows
        ),
    }


def paired_comparison(
    a: list[dict[str, Any]], b: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        raise ValueError("Paired runs must have matching nonempty lengths")
    both = only_a = only_b = neither = 0
    for x, y in zip(a, b, strict=True):
        if x.get("seed") != y.get("seed"):
            raise ValueError("Pairing seed mismatch")
        ax, by = is_certified_clear(x), is_certified_clear(y)
        if ax and by:
            both += 1
        elif ax:
            only_a += 1
        elif by:
            only_b += 1
        else:
            neither += 1
    return {
        "both_win": both, "only_first_wins": only_a,
        "only_second_wins": only_b, "neither_win": neither,
        "clear_rate_delta_first_minus_second": (only_a - only_b) / len(a),
        "mcnemar_exact_two_sided_p": exact_mcnemar(only_a, only_b),
    }


def analyze(
    grouped_rows: dict[str, list[dict[str, Any]]],
    *, mode: str, reference_label: str,
) -> dict[str, Any]:
    if not grouped_rows or reference_label not in grouped_rows:
        raise ValueError("Evaluation requires the nominated round-40 reference")
    lengths = {len(rows) for rows in grouped_rows.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("Model evaluation counts do not match")
    # The single reference model always wins exact ties after progress tie break.
    stats = {name: summarize_model(rows) for name, rows in grouped_rows.items()}
    ranked = sorted(stats, key=lambda name: (
        -stats[name]["act1_clears"],
        -stats[name]["mean_frontier_progress"],
        name != reference_label,
        name,
    ))
    best = ranked[0] if mode == "selection" else None
    return {
        "models": stats,
        "ranking": ranked if mode == "selection" else None,
        "selected_label": best,
        "pairwise_vs_reference": {
            label: paired_comparison(rows, grouped_rows[reference_label])
            for label, rows in grouped_rows.items()
            if label != reference_label
        },
    }
