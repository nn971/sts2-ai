"""Balanced Ascension-10 v8 difficulty screens and source-clustered strata.

All assignments precede combat and depend exclusively on immutable source IDs.
H1's original source namespace and default benchmark execution are untouched.
"""
from __future__ import annotations

import random
import statistics

from sts2_ai.nature.combat_scenarios import (
    SCENARIOS, deck_for_seed, scenario_for_seed,
)


def configuration_key(scenario_id: str, deck_preset_id: str) -> str:
    return f"{scenario_id}::{deck_preset_id}"


def balanced_h2_source_seeds(
    split: str, count: int, scenario_pool: tuple[str, ...],
    deck_pool: tuple[str, ...],
) -> tuple[str, ...]:
    """Exactly equal source counts in each preselected A10 enemy/deck cell.

    Rejection is based ONLY on the public source seed and fixed context hashes,
    never on initial draws, player performance or Nature's actions.
    """
    if split not in {"selection", "confirmation"} or count < 1:
        raise ValueError("Invalid H2 benchmark source split/count")
    if (not scenario_pool or not deck_pool
            or len(set(scenario_pool)) != len(scenario_pool)
            or len(set(deck_pool)) != len(deck_pool)
            or any(key not in SCENARIOS or SCENARIOS[key].ascension != 10
                   for key in scenario_pool)):
        raise ValueError("Balanced H2 benchmark requires pinned A10 scenarios")
    combinations = len(scenario_pool) * len(deck_pool)
    if count % combinations:
        raise ValueError("H2 source count must be divisible by configuration count")
    # Fail closed if a caller supplies an unknown deck.
    deck_for_seed("h2-deck-validation", deck_pool)
    quota = count // combinations
    totals = {
        configuration_key(scenario, deck): 0
        for scenario in scenario_pool for deck in deck_pool
    }
    chosen = []
    index = 0
    while len(chosen) < count:
        seed = f"nature-h2-v8-{split}-{index:08d}"
        index += 1
        key = configuration_key(
            scenario_for_seed(seed, scenario_pool).scenario_id,
            deck_for_seed(seed, deck_pool),
        )
        if totals[key] < quota:
            chosen.append(seed)
            totals[key] += 1
    return tuple(chosen)


def reference_by_configuration(
    cohorts: list[tuple], *, bootstrap_seed: int = 61673,
) -> dict[str, dict]:
    """Summarize exact replayed reference combats by joint enemy/deck cell.

    Bootstrap resamples *source* scores averaged over reference replicates.
    Changed draws and changed scores use the first two reference replicates,
    on the identical source opening, to estimate sensitivity to the draw law.
    """
    if len(cohorts) < 2 or not cohorts[0]:
        raise ValueError("H2 requires two or more reference replicates")
    n = len(cohorts[0])
    first = cohorts[0]
    for cohort in cohorts:
        if len(cohort) != n or any(
            a.source_seed != b.source_seed
            or a.scenario_id != b.scenario_id
            or a.deck_preset_id != b.deck_preset_id
            or a.source_opening_fingerprint != b.source_opening_fingerprint
            or b.draw_mode != "reference" or not b.terminal_replay_verified
            for a, b in zip(first, cohort, strict=True)
        ):
            raise ValueError("H2 reference cohorts do not match source worlds")

    result = {}
    for key in sorted({
        configuration_key(e.scenario_id, e.deck_preset_id) for e in first
    }):
        indices = [
            i for i, e in enumerate(first)
            if configuration_key(e.scenario_id, e.deck_preset_id) == key
        ]
        members = [cohort[i] for cohort in cohorts for i in indices]
        source_scores = tuple(
            statistics.mean(cohort[i].score for cohort in cohorts)
            for i in indices
        )
        rng = random.Random(bootstrap_seed + sum(map(ord, key)))
        source_n = len(indices)
        boot = sorted(
            statistics.mean(
                source_scores[rng.randrange(source_n)] for _ in indices
            )
            for _ in range(1500)
        )
        changed_draws = []
        changed_scores = []
        abs_score_delta = []
        for i in indices:
            a, b = cohorts[0][i], cohorts[1][i]
            trace_a = tuple(
                d.selected_id for d in a.decisions
                if d.phase == "nature:draw_card"
            )
            trace_b = tuple(
                d.selected_id for d in b.decisions
                if d.phase == "nature:draw_card"
            )
            changed_draws.append(trace_a != trace_b)
            delta = abs(a.score - b.score)
            abs_score_delta.append(delta)
            changed_scores.append(delta > 1e-12)
        example = first[indices[0]]
        result[key] = {
            "scenario_id": example.scenario_id,
            "deck_preset_id": example.deck_preset_id,
            "source_count": source_n,
            "combat_count": len(members),
            "win_rate": statistics.mean(
                e.outcome == "victory" for e in members
            ),
            "mean_exit_hp": statistics.mean(e.exit_hp for e in members),
            "mean_score": statistics.mean(e.score for e in members),
            "mean_score_source_bootstrap_95pct": [
                boot[37], boot[1462],
            ],
            "nontrivial_draws_per_combat": statistics.mean(
                e.nontrivial_draw_count for e in members
            ),
            "physical_draws_per_combat": statistics.mean(
                e.controlled_draw_count for e in members
            ),
            "reference_draw_changed_pair_fraction": statistics.mean(
                changed_draws
            ),
            "score_changed_pair_fraction": statistics.mean(changed_scores),
            "mean_abs_draw_replicate_score_delta": statistics.mean(
                abs_score_delta
            ),
        }
    return result
