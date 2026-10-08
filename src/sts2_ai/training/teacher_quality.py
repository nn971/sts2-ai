"""Policy-teacher quality checks with semantic-action aggregation.

These are diagnostics and selection heuristics, *not* statistical confidence
intervals on game outcomes. A UCT root search is oracle-exact, and its visits
may reflect exploration rather than objective value differences.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from sts2_ai.models.hashed_linear import semantic_action_label, state_dict

from .targets import TrainingExample


@dataclass(frozen=True, slots=True)
class TeacherFilter:
    min_budget: int = 32
    min_visits_per_semantic_action: int = 2
    min_semantic_top_margin: float = 0.10
    min_semantic_value_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.min_budget < 0 or self.min_visits_per_semantic_action < 0:
            raise ValueError("Teacher budget and visit thresholds must be nonnegative")
        if not 0.0 <= self.min_semantic_top_margin < 1.0:
            raise ValueError("Teacher probability margin must be in [0,1)")
        if not 0.0 <= self.min_semantic_value_gap <= 2.0:
            raise ValueError("Teacher value gap must lie within [0,2]")


@dataclass(frozen=True, slots=True)
class TeacherRootQuality:
    source_search_id: str
    observation_hash: str
    budget: int
    legal_actions: int
    semantic_actions: int
    root_visits: int
    semantic_normalized_entropy: float
    literal_normalized_entropy: float
    semantic_top_probability: float
    semantic_top_margin: float
    best_semantic_value_gap: float | None
    semantic_min_visits: int
    value_visit_agreement: bool | None
    eligible: bool
    reasons: tuple[str, ...]


_DEFAULT_FILTER = TeacherFilter()


def _normalized(weights: list[float]) -> list[float]:
    if not weights:
        return []
    total = sum(weights)
    return [v / total for v in weights] if total > 0.0 else [1.0 / len(weights)] * len(weights)


def _entropy_normalized(probabilities: list[float]) -> float:
    if len(probabilities) < 2:
        return 0.0
    value = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return value / math.log(len(probabilities))


def grade_teacher_root(
    example: TrainingExample, config: TeacherFilter = _DEFAULT_FILTER
) -> TeacherRootQuality:
    if not example.policy_targets:
        raise ValueError("Root is missing legal action targets")
    if len({p.action_id for p in example.policy_targets}) != len(example.policy_targets):
        raise ValueError("Duplicate action identifiers in searched root")

    state = state_dict(example.observation_json)
    literal = _normalized([max(0.0, p.probability) for p in example.policy_targets])
    groups: dict[str, list[int]] = defaultdict(list)
    for i, action in enumerate(example.policy_targets):
        label = semantic_action_label(state, action.action_kind, action.action_payload_json)
        # Never declare two choices equivalent solely because the semantic
        # resolver lacks that action type or fails to resolve its card/target.
        # Unknown shop/event decisions can have very different outcomes.
        if (
            label == action.action_kind
            or "unknown-card" in label
            or "unknown-enemy" in label
            or label.endswith("room=None")
        ):
            label = f"unresolved:{action.action_id}"
        groups[label].append(i)

    semantic_labels = sorted(groups)
    semantic_weights = [
        sum(max(0.0, example.policy_targets[i].probability) for i in groups[label])
        for label in semantic_labels
    ]
    semantic_probs = _normalized(semantic_weights)
    total_visits = sum(max(0, p.visits) for p in example.policy_targets)
    semantic_visits = [
        sum(max(0, example.policy_targets[i].visits) for i in groups[label])
        for label in semantic_labels
    ]

    # Aggregate action means by sample count. Unvisited actions do not contribute
    # a fabricated value estimate; a group with zero visits is unmeasured.
    semantic_values: list[float | None] = []
    for label, visits in zip(semantic_labels, semantic_visits, strict=True):
        value_sum = sum(
            example.policy_targets[i].search_value
            * max(0, example.policy_targets[i].visits)
            for i in groups[label]
        )
        semantic_values.append(value_sum / visits if visits > 0 else None)

    ranked = sorted(
        (
            (value, idx)
            for idx, value in enumerate(semantic_values)
            if value is not None
        ),
        reverse=True,
    )
    gap = ranked[0][0] - ranked[1][0] if len(ranked) >= 2 else None
    visit_max = max(range(len(semantic_labels)), key=lambda i: (semantic_probs[i], -i))
    best_q = ranked[0][1] if ranked else None
    agrees = visit_max == best_q if best_q is not None else None
    top = max(semantic_probs)
    margin = top - 1.0 / len(semantic_labels)
    reasons = []
    if len(semantic_labels) < 2:
        reasons.append("one-semantic-action")
    if example.search_budget < config.min_budget:
        reasons.append("low-budget")
    if min(semantic_visits) < config.min_visits_per_semantic_action:
        reasons.append("underexplored-semantic-action")
    if margin < config.min_semantic_top_margin:
        reasons.append("flat-semantic-policy")
    if gap is None or gap <= config.min_semantic_value_gap:
        reasons.append("no-measured-value-gap")
    if agrees is not True:
        reasons.append("visits-disagree-with-best-value")

    return TeacherRootQuality(
        source_search_id=example.source_search_id,
        observation_hash=example.observation_hash,
        budget=example.search_budget,
        legal_actions=len(example.policy_targets),
        semantic_actions=len(semantic_labels),
        root_visits=total_visits,
        semantic_normalized_entropy=_entropy_normalized(semantic_probs),
        literal_normalized_entropy=_entropy_normalized(literal),
        semantic_top_probability=top,
        semantic_top_margin=margin,
        best_semantic_value_gap=gap,
        semantic_min_visits=min(semantic_visits),
        value_visit_agreement=agrees,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def teacher_quality_report(
    examples: tuple[TrainingExample, ...],
    config: TeacherFilter = _DEFAULT_FILTER,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("Teacher audit requires searched roots")
    qualities = tuple(grade_teacher_root(e, config) for e in examples)
    reasons = Counter(reason for item in qualities for reason in item.reasons)
    multi = [q for q in qualities if q.semantic_actions > 1]
    accepted = [q for q in qualities if q.eligible]
    agreements = [q.value_visit_agreement for q in multi if q.value_visit_agreement is not None]
    gaps = [q.best_semantic_value_gap for q in multi if q.best_semantic_value_gap is not None]
    return {
        "examples": len(examples),
        "eligible_examples": len(accepted),
        "eligible_fraction": len(accepted) / len(examples),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "mean_semantic_actions": fmean(q.semantic_actions for q in qualities),
        "mean_literal_actions": fmean(q.legal_actions for q in qualities),
        "mean_literal_normalized_entropy": fmean(
            q.literal_normalized_entropy for q in qualities
        ),
        "mean_semantic_normalized_entropy_multi_action": (
            fmean(q.semantic_normalized_entropy for q in multi) if multi else None
        ),
        "mean_semantic_top_margin_multi_action": (
            fmean(q.semantic_top_margin for q in multi) if multi else None
        ),
        "mean_best_semantic_value_gap": fmean(gaps) if gaps else None,
        "value_visit_agreement_fraction": (
            sum(agreements) / len(agreements) if agreements else None
        ),
        "single_semantic_action_fraction": (
            sum(q.semantic_actions == 1 for q in qualities) / len(qualities)
        ),
        "eligible_phase_counts": dict(sorted(Counter(
            str(json.loads(e.observation_json).get("phase", "?"))
            for e, quality in zip(examples, qualities, strict=True)
            if quality.eligible
        ).items())),
        "teacher_filter": {
            "min_budget": config.min_budget,
            "min_visits_per_semantic_action": config.min_visits_per_semantic_action,
            "min_semantic_top_margin": config.min_semantic_top_margin,
            "min_semantic_value_gap": config.min_semantic_value_gap,
        },
        "warning": (
            "Action semantics aggregate interchangeable legal actions. Eligibility is "
            "a heuristic signal-quality gate, NOT a calibrated statistical confidence "
            "measure or proof of correct action ranking."
        ),
    }


def curate_teacher_examples(
    examples: tuple[TrainingExample, ...],
    config: TeacherFilter = _DEFAULT_FILTER,
) -> tuple[TrainingExample, ...]:
    """Return unchanged selected examples; never fabricate sharpened labels."""
    return tuple(example for example in examples if grade_teacher_root(example, config).eligible)
