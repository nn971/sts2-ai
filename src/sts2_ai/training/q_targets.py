"""Optional policy targets from searched action values, not UCT visits.

The acting oracle MCTS agent selects the highest sampled mean return.
Distilling its *visit distribution* instead teaches exploration. This module
exposes an explicit, evidence-gated alternative without changing the original
records, and reports rejection counts rather than inventing confidence.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

from sts2_ai.models.hashed_linear import semantic_action_label, state_dict

from .targets import TrainingExample
from .teacher_quality import conservative_semantic_group


@dataclass(frozen=True, slots=True)
class QTeacherConfig:
    min_budget: int = 64
    min_semantic_visits: int = 2
    min_value_gap: float = 0.05
    uncertainty_scale: float = 0.25
    temperature: float = 0.15

    def __post_init__(self) -> None:
        if self.min_budget < 0 or self.min_semantic_visits < 1:
            raise ValueError("Budget must be nonnegative and visits positive")
        if not 0.0 <= self.min_value_gap <= 2.0:
            raise ValueError("Minimum Q gap must lie within [0,2]")
        if not 0.0 <= self.uncertainty_scale <= 10.0:
            raise ValueError("Uncertainty scale must lie within [0,10]")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("Softmax temperature must be finite and positive")

    @property
    def mode(self) -> str:
        return (
            f"q-softmax-v2-t{self.temperature.hex()}-b{self.min_budget}"
            f"-v{self.min_semantic_visits}-g{self.min_value_gap.hex()}"
            f"-u{self.uncertainty_scale.hex()}"
        )


_DEFAULT_Q_CONFIG = QTeacherConfig()


def distill_one_q_target(
    example: TrainingExample, config: QTeacherConfig
) -> tuple[TrainingExample | None, str, float | None]:
    if example.search_budget < config.min_budget:
        return None, "low-budget", None
    if len(example.policy_targets) < 2:
        return None, "forced-choice", None

    state = state_dict(example.observation_json)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, target in enumerate(example.policy_targets):
        groups[conservative_semantic_group(state, target)].append(i)
    if len(groups) < 2:
        return None, "one-semantic-action", None

    labels = sorted(groups)
    # The v2 neural policy head only sees kind plus semantic-action label.
    # It cannot distinguish two distinct map nodes of the same room type,
    # unresolved event choices, etc. Do not train it on contradictory labels.
    representable_groups: dict[str, set[str]] = defaultdict(set)
    for group_label, indices in groups.items():
        for index in indices:
            action = example.policy_targets[index]
            signature = (
                action.action_kind + ":" + semantic_action_label(
                    state, action.action_kind, action.action_payload_json
                )
            )
            representable_groups[signature].add(group_label)
    if any(len(group_set) > 1 for group_set in representable_groups.values()):
        return None, "model-cannot-distinguish-actions", None
    means = []
    uncertainties = []
    for label in labels:
        actions = [example.policy_targets[i] for i in groups[label]]
        visits = sum(max(0, a.visits) for a in actions)
        if visits < config.min_semantic_visits:
            return None, "insufficient-semantic-visits", None
        if any(
            a.visits < 0 or not math.isfinite(a.search_value)
            or abs(a.search_value) > 1.000001
            for a in actions
        ):
            return None, "invalid-q-evidence", None
        mean = sum(a.visits * a.search_value for a in actions) / visits
        means.append(mean)
        # Explicitly *not* a calibrated confidence interval: search samples
        # are correlated through a shared tree and often deterministic.
        uncertainties.append(math.sqrt(max(0.0, 1.0 - mean * mean) / visits))

    ranked = sorted(range(len(labels)), key=lambda i: (means[i], labels[i]), reverse=True)
    winner, runner = ranked[:2]
    gap = means[winner] - means[runner]
    required = max(
        config.min_value_gap,
        config.uncertainty_scale * (uncertainties[winner] + uncertainties[runner]),
    )
    if gap <= required:
        return None, "indistinguishable-action-values", gap

    shifted = [(m - max(means)) / config.temperature for m in means]
    weights = [math.exp(x) for x in shifted]
    normalizer = sum(weights)
    if not normalizer > 0.0:
        return None, "numerical-underflow", gap

    targets = list(example.policy_targets)
    for label, weight in zip(labels, weights, strict=True):
        per_action = weight / normalizer / len(groups[label])
        for index in groups[label]:
            targets[index] = replace(targets[index], probability=per_action)
    assert math.isclose(sum(t.probability for t in targets), 1.0, abs_tol=1e-9)
    return (
        replace(example, policy_targets=tuple(targets), policy_target_mode=config.mode),
        "accepted",
        gap,
    )


def distill_q_targets(
    examples: tuple[TrainingExample, ...],
    config: QTeacherConfig = _DEFAULT_Q_CONFIG,
) -> tuple[tuple[TrainingExample, ...], dict[str, Any]]:
    if not examples:
        raise ValueError("Q-target distillation needs searched-root data")
    kept = []
    reasons: Counter[str] = Counter()
    gaps = []
    for example in examples:
        target, reason, gap = distill_one_q_target(example, config)
        reasons[reason] += 1
        if target is not None:
            kept.append(target)
            if gap is not None:
                gaps.append(gap)
    report = {
        "source_examples": len(examples),
        "accepted": len(kept),
        "accepted_fraction": len(kept) / len(examples),
        "reason_counts": dict(sorted(reasons.items())),
        "mean_accepted_best_runner_value_gap": fmean(gaps) if gaps else None,
        "policy_target_mode": config.mode,
        "warning": (
            "Q-softmax policy labels imitate the root action MEAN-VALUE ranking, "
            "not UCT exploration visits. Their Q gaps and uncertainty proxies "
            "are uncalibrated. Rejected roots must not be relabeled as confident."
        ),
    }
    return tuple(kept), report
