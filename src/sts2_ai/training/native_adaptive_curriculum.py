"""H9 learning-progress curriculum, adapted to real native A10 combat entries.

Family = full sorted enemy composition (including multiplicity). No manufactured
deck/enemy/HP combination; all recipes come from the pinned native source run.
Fixed, deterministic greedy probes from the TRAINING corpus estimate absolute
change in H9 total-nonminion-HP return. EMA and uniform floor mirror the
earlier H9 adaptive-rich controller. No held-out run is used for curriculum.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe

SCHEMA = "sts2-h15-native-encounter-learning-progress-v1"


def family_key(recipe: CombatSnapshotRecipe) -> str:
    recipe.validate()
    return json.dumps(
        sorted(recipe.enemy_ids), separators=(",", ":"), ensure_ascii=True,
    )


def family_pools(
    recipes: Sequence[CombatSnapshotRecipe],
) -> dict[str, tuple[CombatSnapshotRecipe, ...]]:
    if not recipes:
        raise ValueError("Native adaptive sampler requires training entries")
    pools: dict[str, list[CombatSnapshotRecipe]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for recipe in recipes:
        key = (recipe.seed, recipe.exact_hash)
        if key in seen:
            raise ValueError("Native adaptive sampling cannot contain duplicate entries")
        seen.add(key)
        pools[family_key(recipe)].append(recipe)
    return {
        name: tuple(sorted(rows, key=lambda r: (r.seed, r.exact_hash)))
        for name, rows in sorted(pools.items())
    }


def _seed(study_seed: int, domain: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"h15-native-adaptive:{study_seed}:{domain}".encode(),
    ).digest()[:8], "big")


def fixed_probe_recipes(
    recipes: Sequence[CombatSnapshotRecipe], *, per_family: int,
    study_seed: int,
) -> tuple[CombatSnapshotRecipe, ...]:
    """Deterministic per-family, fixed native training entries.

    Small pools use as many distinct entries as exist, without inventing
    additional states or counting the same entry repeatedly as independent.
    """
    if per_family < 1 or study_seed < 0:
        raise ValueError("Invalid native adaptive probe budget")
    pools = family_pools(recipes)
    result: list[CombatSnapshotRecipe] = []
    for name, rows in pools.items():
        rng = random.Random(_seed(study_seed, "probes:" + name))
        result.extend(rng.sample(list(rows), min(per_family, len(rows))))
    return tuple(result)


def feedback_means(
    probes: Sequence[CombatSnapshotRecipe],
    outcomes: Sequence[Any],
) -> dict[str, float]:
    if len(probes) != len(outcomes) or not probes:
        raise ValueError("Native fixed probes/results are missing")
    pooled: dict[str, list[float]] = defaultdict(list)
    for recipe, outcome in zip(probes, outcomes, strict=True):
        if (outcome.source_seed != recipe.seed or
                outcome.entry_hash != recipe.exact_hash or
                outcome.outcome not in ("victory", "defeat")):
            raise ValueError("Probe outcome differs from selected native entry")
        score = float(outcome.reward)
        if not math.isfinite(score) or not 0 <= score <= 2:
            raise ValueError("Native probe returned an invalid H9 score")
        pooled[family_key(recipe)].append(score)
    return {name: statistics.mean(scores) for name, scores in sorted(pooled.items())}


@dataclass
class NativeAdaptiveProgress:
    """Same beta/alpha/epsilon update semantics as H9 AdaptiveRichProgress."""

    families: tuple[str, ...]
    beta: float = 0.7
    alpha: float = 0.5
    epsilon: float = 0.01
    previous: dict[str, float] | None = None
    progress: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    evaluations: int = 0

    def __post_init__(self) -> None:
        if (not self.families or len(set(self.families)) != len(self.families)
                or tuple(sorted(self.families)) != self.families
                or not 0 <= self.beta < 1
                or not 0 < self.alpha <= 1
                or not math.isfinite(self.epsilon) or self.epsilon <= 0
                or self.evaluations < 0):
            raise ValueError("Invalid native adaptive curriculum configuration")
        if not self.progress:
            self.progress = {name: 0.0 for name in self.families}
        if not self.weights:
            self.weights = {name: 1 / len(self.families) for name in self.families}
        if (set(self.progress) != set(self.families)
                or set(self.weights) != set(self.families)
                or (self.previous is not None and
                    set(self.previous) != set(self.families))
                or any(not math.isfinite(x) or x < 0
                       for x in self.progress.values())
                or any(not math.isfinite(w) or w <= 0
                       for w in self.weights.values())
                or not math.isclose(sum(self.weights.values()), 1, abs_tol=1e-8)):
            raise ValueError("Invalid native adaptive curriculum checkpoint")

    def update(self, mean_rewards: dict[str, float]) -> dict[str, Any]:
        if (set(mean_rewards) != set(self.families)
                or any(not math.isfinite(float(x))
                       for x in mean_rewards.values())):
            raise ValueError("Native adaptive feedback omits encounter family")
        means = {name: float(mean_rewards[name]) for name in self.families}
        changes = {name: 0.0 for name in self.families}
        if self.previous is not None:
            for name in self.families:
                changes[name] = abs(means[name] - self.previous[name])
                self.progress[name] = (
                    self.alpha * changes[name] +
                    (1 - self.alpha) * self.progress[name]
                )
            total = sum(self.epsilon + self.progress[name]
                        for name in self.families)
            self.weights = {
                name: (
                    (1 - self.beta) / len(self.families) +
                    self.beta * (self.epsilon + self.progress[name]) / total
                ) for name in self.families
            }
        self.previous = means
        self.evaluations += 1
        return {
            "means": means, "absolute_change": changes,
            "smoothed_progress": dict(self.progress),
            "weights": dict(self.weights),
            "evaluations": self.evaluations,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "families": list(self.families),
            "beta": self.beta, "alpha": self.alpha, "epsilon": self.epsilon,
            "previous": self.previous, "progress": dict(self.progress),
            "weights": dict(self.weights), "evaluations": self.evaluations,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> NativeAdaptiveProgress:
        if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
            raise ValueError("Native adaptive checkpoint schema mismatch")
        return cls(
            families=tuple(obj["families"]), beta=obj["beta"],
            alpha=obj["alpha"], epsilon=obj["epsilon"],
            previous=obj["previous"], progress=dict(obj["progress"]),
            weights=dict(obj["weights"]), evaluations=obj["evaluations"],
        )


def weighted_native_selection(
    recipes: Sequence[CombatSnapshotRecipe], *,
    count: int, study_seed: int, round_index: int,
    weights: dict[str, float],
) -> tuple[CombatSnapshotRecipe, ...]:
    """Adaptive encounter-family draws, then diverse progress/seed snapshots.

    Each family cycles over its observed public progress bands and distinct
    source runs before revisiting a recipe. Distinct small families may repeat
    when count exceeds the available training entries, without fabrication.
    """
    if count < 1 or round_index < 0 or study_seed < 0:
        raise ValueError("Invalid adaptive native cohort size or seed")
    pools = family_pools(recipes)
    if (set(weights) != set(pools) or
            any(not math.isfinite(w) or w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1, abs_tol=1e-8)):
        raise ValueError("Native adaptive encounter weights are invalid")
    rng = random.Random(_seed(study_seed, f"train:{round_index}"))
    families = sorted(pools)
    chosen = rng.choices(
        families, weights=[weights[f] for f in families], k=count,
    )
    # Round-robin progress strata within each encounter ensures broad source
    # states without manually synthesizing combinations.
    ordered: dict[str, list[CombatSnapshotRecipe]] = {}
    offsets: dict[str, int] = {}
    for family in families:
        by_progress: dict[str, list[CombatSnapshotRecipe]] = defaultdict(list)
        for row in pools[family]:
            by_progress[row.progress].append(row)
        for rows in by_progress.values():
            rng.shuffle(rows)
        keys = sorted(by_progress)
        rng.shuffle(keys)
        positions = {key: 0 for key in keys}
        merged: list[CombatSnapshotRecipe] = []
        while len(merged) < len(pools[family]):
            for key in keys:
                i = positions[key]
                if i < len(by_progress[key]):
                    merged.append(by_progress[key][i])
                    positions[key] += 1
        ordered[family] = merged
        offsets[family] = 0
    selected: list[CombatSnapshotRecipe] = []
    for family in chosen:
        candidates = ordered[family]
        offset = offsets[family]
        if offset and offset % len(candidates) == 0:
            rng.shuffle(candidates)
        selected.append(candidates[offset % len(candidates)])
        offsets[family] += 1
    return tuple(selected)
