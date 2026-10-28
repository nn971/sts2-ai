"""Minimal learning-progress sampler for H9 rich combat fixtures.

Only training structural indices 0..1023 are used; the immutable v2
generator and reserved zero-shot indices 1024..1279 remain untouched.
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from sts2_ai.nature.research_combat_fixtures import (
    RICH_ENCOUNTERS, fixture_from_source, rich_fixture_for_index,
)

ADAPTIVE_SCHEMA = "sts2-h9-rich-learning-progress-v1"
FAMILIES = tuple(item[0] for item in RICH_ENCOUNTERS)
TRAIN_SPECS = 1024


@lru_cache(maxsize=4)
def training_pools(curriculum_seed: int) -> dict[str, tuple[int, ...]]:
    """Source-index partitions; no new game mechanics or fixture rewriting."""
    pools: dict[str, list[int]] = {name: [] for name in FAMILIES}
    for index in range(TRAIN_SPECS):
        fixture = rich_fixture_for_index(index, curriculum_seed=curriculum_seed)
        pools[fixture["encounter_id"]].append(index)
    if any(not entries for entries in pools.values()):
        raise ValueError("Rich training corpus omits an encounter family")
    return {name: tuple(entries) for name, entries in pools.items()}


def _random_seed(study_seed: int, domain: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"h9-adaptive:{study_seed}:{domain}".encode()
    ).digest()[:8], "big")


def probe_sources(
    *, study_seed: int, curriculum_seed: int, per_family: int = 6,
) -> tuple[str, ...]:
    """Same per-encounter *training* fixtures and chance seeds at each probe."""
    if per_family < 1 or study_seed < 0:
        raise ValueError("Invalid adaptive probe configuration")
    pools = training_pools(curriculum_seed)
    seeds: list[str] = []
    for name in FAMILIES:
        indices = list(pools[name])
        rng = random.Random(_random_seed(study_seed, "probe:" + name))
        rng.shuffle(indices)
        for ordinal in range(per_family):
            index = indices[ordinal % len(indices)]
            # Fixed unique source identity even if a tiny family repeats.
            seeds.append(
                f"h9-rich-train-v2-{study_seed}-0-{index}-{2000000 + len(seeds)}"
            )
    return tuple(seeds)


def weighted_train_sources(
    count: int, *, round_index: int, study_seed: int,
    curriculum_seed: int, weights: dict[str, float],
) -> tuple[str, ...]:
    """Draw encounter families then distinct fixtures per family when possible."""
    if count < 1 or round_index < 0 or study_seed < 0:
        raise ValueError("Invalid adaptive train source request")
    if (set(weights) != set(FAMILIES)
            or any(not math.isfinite(w) or w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8)):
        raise ValueError("Invalid adaptive encounter sampling weights")
    pools = training_pools(curriculum_seed)
    rng = random.Random(_random_seed(study_seed, f"train:{round_index}"))
    chosen = rng.choices(FAMILIES, weights=[weights[n] for n in FAMILIES], k=count)
    shuffled: dict[str, list[int]] = {
        name: rng.sample(list(pools[name]), k=len(pools[name]))
        for name in FAMILIES
    }
    offsets = {name: 0 for name in FAMILIES}
    result = []
    for ordinal, name in enumerate(chosen):
        indices = shuffled[name]
        offset = offsets[name]
        if offset and offset % len(indices) == 0:
            rng.shuffle(indices)
        index = indices[offset % len(indices)]
        offsets[name] += 1
        result.append(
            f"h9-rich-train-v2-{study_seed}-{round_index}-{index}-{ordinal}"
        )
    return tuple(result)


def encounter_mean_rewards(
    episodes: Sequence[Any], *, curriculum_seed: int,
    expected_per_family: int,
) -> dict[str, float]:
    by_family: dict[str, list[float]] = {name: [] for name in FAMILIES}
    for episode in episodes:
        scenario, _, _ = fixture_from_source(
            episode.source_seed, curriculum_seed=curriculum_seed,
        )
        if episode.outcome not in ("victory", "defeat"):
            raise ValueError("Adaptive probes require terminal combats")
        score = float(episode.score)
        if not math.isfinite(score):
            raise ValueError("Nonfinite adaptive probe score")
        by_family[scenario.encounter_id].append(score)
    if any(len(scores) != expected_per_family for scores in by_family.values()):
        raise ValueError("Adaptive probes must cover every encounter equally")
    return {name: statistics.mean(scores) for name, scores in by_family.items()}


@dataclass
class AdaptiveRichProgress:
    """Reward-change EMA, with a uniform floor; no difficulty heuristics."""

    beta: float = 0.7
    alpha: float = 0.5
    epsilon: float = 0.01
    previous: dict[str, float] | None = None
    progress: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    evaluations: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.beta < 1 and 0 < self.alpha <= 1
                and math.isfinite(self.epsilon) and self.epsilon > 0):
            raise ValueError("Invalid learning-progress coefficients")
        if not self.progress:
            self.progress = {name: 0.0 for name in FAMILIES}
        if not self.weights:
            self.weights = {name: 1.0 / len(FAMILIES) for name in FAMILIES}

    def update(self, mean_rewards: dict[str, float]) -> dict[str, Any]:
        if (set(mean_rewards) != set(FAMILIES)
                or any(not math.isfinite(float(x))
                       for x in mean_rewards.values())):
            raise ValueError("Adaptive feedback requires all finite family means")
        means = {name: float(mean_rewards[name]) for name in FAMILIES}
        changes = {name: 0.0 for name in FAMILIES}
        if self.previous is not None:
            for name in FAMILIES:
                changes[name] = abs(means[name] - self.previous[name])
                self.progress[name] = (
                    self.alpha * changes[name]
                    + (1 - self.alpha) * self.progress[name]
                )
            total = sum(self.epsilon + self.progress[name] for name in FAMILIES)
            self.weights = {
                name: ((1 - self.beta) / len(FAMILIES) +
                       self.beta * (self.epsilon + self.progress[name]) / total)
                for name in FAMILIES
            }
        self.previous = means
        self.evaluations += 1
        return {
            "means": means, "absolute_change": changes,
            "smoothed_progress": dict(self.progress),
            "weights": dict(self.weights), "evaluations": self.evaluations,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTIVE_SCHEMA, "beta": self.beta, "alpha": self.alpha,
            "epsilon": self.epsilon, "previous": self.previous,
            "progress": dict(self.progress), "weights": dict(self.weights),
            "evaluations": self.evaluations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveRichProgress":
        if not isinstance(data, dict) or data.get("schema") != ADAPTIVE_SCHEMA:
            raise ValueError("Adaptive progress checkpoint schema mismatch")
        restored = cls(
            beta=float(data["beta"]), alpha=float(data["alpha"]),
            epsilon=float(data["epsilon"]), previous=data["previous"],
            progress=dict(data["progress"]), weights=dict(data["weights"]),
            evaluations=int(data["evaluations"]),
        )
        if (set(restored.progress) != set(FAMILIES)
                or set(restored.weights) != set(FAMILIES)
                or (restored.previous is not None
                    and set(restored.previous) != set(FAMILIES))
                or any(not math.isfinite(x) or x < 0
                       for x in restored.progress.values())
                or any(not math.isfinite(x) or x <= 0
                       for x in restored.weights.values())
                or not math.isclose(sum(restored.weights.values()), 1.0,
                                    abs_tol=1e-8)
                or restored.evaluations < 1):
            raise ValueError("Corrupt adaptive progress checkpoint")
        return restored
