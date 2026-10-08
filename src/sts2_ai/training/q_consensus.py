"""Conservative cross-budget agreement gate for oracle-exact root-Q teachers.

UCT visits are for exploration; our acting search policy chooses the highest
estimated mean action value. At small budgets these rankings may flip. Require
both searches to choose the same *semantic* Q winner on an identical exact
root before using an evidence-gated high-budget softmax target.

This is agreement filtering, NOT a calibrated confidence interval or an
independent outcome measurement: both searches share the emulator state,
heuristic cutoff and often correlated rollout paths.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

from sts2_ai.models.hashed_linear import state_dict

from .q_targets import QTeacherConfig, distill_one_q_target
from .targets import TrainingExample
from .teacher_quality import conservative_semantic_group


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    high_q: QTeacherConfig = QTeacherConfig()
    min_low_semantic_visits: int = 1
    min_low_value_gap: float = 0.01

    def __post_init__(self) -> None:
        if self.min_low_semantic_visits < 1:
            raise ValueError("Minimum low-budget visits must be positive")
        if not math.isfinite(self.min_low_value_gap) or not (
            0 <= self.min_low_value_gap <= 2
        ):
            raise ValueError("Low-budget Q gap must lie within [0,2]")

    @property
    def mode(self) -> str:
        return (
            "cross-budget-q-consensus-v1:"
            + self.high_q.mode
            + f":low-v{self.min_low_semantic_visits}"
            + f"-g{self.min_low_value_gap.hex()}"
        )


def _indexed(
    roots: tuple[TrainingExample, ...],
) -> dict[tuple[str, str, str, str], TrainingExample]:
    index: dict[tuple[str, str, str, str], TrainingExample] = {}
    for root in roots:
        if root.policy_target_mode != "uct-visits-v1":
            raise ValueError("Consensus input requires original UCT-visit evidence")
        if not root.source_state_hash:
            raise ValueError("Consensus input requires exact-state hashes")
        key = (
            root.source_state_hash,
            root.information_policy,
            root.emulator_revision,
            root.game_build,
        )
        if key in index:
            raise ValueError("Duplicate exact root within one teacher dataset")
        index[key] = root
    return index


def _winner(
    root: TrainingExample, min_visits: int, min_gap: float
) -> tuple[str | None, str]:
    state = state_dict(root.observation_json)
    sum_q: dict[str, float] = defaultdict(float)
    visits: dict[str, int] = defaultdict(int)
    for action in root.policy_targets:
        if action.visits < 0 or not math.isfinite(action.search_value):
            return None, "invalid-low-q"
        if action.visits == 0:
            continue
        if abs(action.search_value) > 1.000001:
            return None, "invalid-low-q"
        label = conservative_semantic_group(state, action)
        sum_q[label] += action.visits * action.search_value
        visits[label] += action.visits
    # Do not assume that unvisited alternatives are inferior.
    all_groups = {
        conservative_semantic_group(state, a) for a in root.policy_targets
    }
    if len(all_groups) < 2:
        return None, "single-semantic-action"
    if any(visits[group] < min_visits for group in all_groups):
        return None, "underexplored-low-q"
    ordered = sorted(
        ((sum_q[group] / visits[group], group) for group in all_groups),
        reverse=True,
    )
    if ordered[0][0] - ordered[1][0] <= min_gap:
        return None, "ambiguous-low-q"
    return ordered[0][1], "resolved"


def distill_consensus_targets(
    low: tuple[TrainingExample, ...],
    high: tuple[TrainingExample, ...],
    config: ConsensusConfig = ConsensusConfig(),
) -> tuple[tuple[TrainingExample, ...], dict[str, Any]]:
    """Export only high-budget action-Q targets whose low-budget winner agrees."""
    low_index, high_index = _indexed(low), _indexed(high)
    matched = sorted(low_index.keys() & high_index.keys())
    if not matched:
        raise ValueError("No identical exact roots in low/high teacher datasets")

    accepted: list[TrainingExample] = []
    reasons: Counter[str] = Counter()
    comparable = 0
    agreeing = 0
    accepted_q_gaps = []
    for key in matched:
        before, after = low_index[key], high_index[key]
        if before.search_budget >= after.search_budget:
            raise ValueError("Low teacher budget must be strictly below high budget")
        if before.observation_hash != after.observation_hash:
            raise ValueError("Matched exact root has conflicting fair observation")
        low_actions = {
            (p.action_id, p.action_kind, p.action_payload_json)
            for p in before.policy_targets
        }
        high_actions = {
            (p.action_id, p.action_kind, p.action_payload_json)
            for p in after.policy_targets
        }
        if low_actions != high_actions:
            raise ValueError("Matched exact root has conflicting legal actions")

        winner_low, reason = _winner(
            before, config.min_low_semantic_visits, config.min_low_value_gap
        )
        if winner_low is None:
            reasons[reason] += 1
            continue
        winner_high, reason = _winner(after, config.high_q.min_semantic_visits, 0.0)
        if winner_high is None:
            reasons["high-" + reason] += 1
            continue
        comparable += 1
        if winner_low != winner_high:
            reasons["q-winner-disagreement"] += 1
            continue
        agreeing += 1
        # Check action representation, leading-Q gap and the deliberately
        # uncalibrated uncertainty proxy with the existing Q distiller.
        trained, rejection, gap = distill_one_q_target(after, config.high_q)
        if trained is None:
            reasons["high-" + rejection] += 1
            continue
        accepted_q_gaps.append(gap if gap is not None else 0.0)
        accepted.append(
            replace(trained, policy_target_mode=config.mode)
        )
        reasons["accepted"] += 1

    report: dict[str, Any] = {
        "low_examples": len(low),
        "high_examples": len(high),
        "matched_exact_roots": len(matched),
        "unmatched_low": len(low) - len(matched),
        "unmatched_high": len(high) - len(matched),
        "comparable_resolved_roots": comparable,
        "q_winner_agreement_fraction": agreeing / comparable if comparable else None,
        "accepted": len(accepted),
        "accepted_fraction_of_matched": len(accepted) / len(matched),
        "accepted_mean_q_gap": fmean(accepted_q_gaps) if accepted_q_gaps else None,
        "reason_counts": dict(sorted(reasons.items())),
        "policy_target_mode": config.mode,
        "source_run_seeds": sorted({
            seed for example in accepted for seed in example.source_run_seeds
        }),
        "warning": (
            "Both exact-state searches are oracle-exact, share rollout biases, and "
            "Q gaps are uncalibrated; matching their winners is not proof of "
            "better decisions. Require independent held-out gameplay tests."
        ),
    }
    return tuple(accepted), report
