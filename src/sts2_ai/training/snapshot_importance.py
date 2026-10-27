"""Fixed-target, adaptive-proposal importance sampling for authentic combat entries.

The *target* distribution defines the desired combat objective. The *proposal*
only changes which entries receive simulation budget; every draw carries p/q.
Initially the priority is a transparent HP/encounter proxy. Future pilots may
supply independently estimated gradient/RMS priorities via `priority_scores`.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe
from sts2_ai.training.snapshot_curriculum import source_key

SnapshotIdentity = tuple[str, int, str, str]
TIERS = {"weak": 0.0, "normal": 0.5, "elite": 1.0, "boss": 1.5}


def snapshot_identity(recipe: CombatSnapshotRecipe) -> SnapshotIdentity:
    """Source-run provenance plus exact combat-entry fingerprint."""
    seed, ascension, policy = source_key(recipe)
    return seed, ascension, policy, recipe.exact_hash


def proxy_priority(recipe: CombatSnapshotRecipe) -> float:
    """Cheap, positive difficulty proxy; NOT a claim of gradient optimality."""
    recipe.validate()
    return 1.0 + TIERS[recipe.tier] + 2.0 * (1.0 - recipe.hp / recipe.max_hp)


@dataclass(frozen=True, slots=True)
class ImportanceDraw:
    recipe: CombatSnapshotRecipe
    target_probability: float
    proposal_probability: float
    weight: float
    priority: float


@dataclass(frozen=True, slots=True)
class ImportanceDistribution:
    recipes: tuple[CombatSnapshotRecipe, ...]
    target_probabilities: tuple[float, ...]
    proposal_probabilities: tuple[float, ...]
    priorities: tuple[float, ...]
    target_balance: float
    priority_mix: float

    def draw(self, *, count: int, seed: int) -> tuple[ImportanceDraw, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError("Importance draw count must be positive")
        indices = random.Random(seed).choices(
            range(len(self.recipes)), weights=self.proposal_probabilities, k=count,
        )
        return tuple(
            ImportanceDraw(
                self.recipes[i], self.target_probabilities[i],
                self.proposal_probabilities[i],
                self.target_probabilities[i] / self.proposal_probabilities[i],
                self.priorities[i],
            )
            for i in indices
        )


def build_importance_distribution(
    recipes: Sequence[CombatSnapshotRecipe], *,
    target_balance: float = 0.5,
    priority_mix: float = 0.7,
    priority_scores: Mapping[SnapshotIdentity, float] | None = None,
) -> ImportanceDistribution:
    """Construct an exact finite-corpus p/q scheme with full support.

    p = (1-beta) * uniform_entries + beta * uniform_observed_strata,
    where a stratum is (act progress, encounter tier).
    q = (1-lambda) * p + lambda * normalized(p * priority).
    Hence p/q <= 1/(1-lambda) for lambda < 1.

    Priorities cannot change the target. All input ordering is canonicalized,
    and repetition is *sampling* repetition, not independent environment RNG.
    """
    if not recipes:
        raise ValueError("Importance sampling requires a nonempty snapshot corpus")
    if not (math.isfinite(target_balance) and 0 <= target_balance <= 1):
        raise ValueError("target_balance must lie in [0, 1]")
    if not (math.isfinite(priority_mix) and 0 <= priority_mix < 1):
        raise ValueError("priority_mix must lie in [0, 1)")
    ordered = tuple(sorted(
        recipes, key=lambda r: (source_key(r), len(r.action_ids), r.exact_hash),
    ))
    identifiers = [snapshot_identity(r) for r in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicated exact snapshot entry")
    for recipe in ordered:
        recipe.validate()
    strata = Counter((r.progress, r.tier) for r in ordered)
    n, k = len(ordered), len(strata)
    p = tuple(
        (1 - target_balance) / n
        + target_balance / (k * strata[(r.progress, r.tier)])
        for r in ordered
    )
    priorities = tuple(
        proxy_priority(r) if priority_scores is None
        else priority_scores.get(snapshot_identity(r), proxy_priority(r))
        for r in ordered
    )
    if any(not math.isfinite(s) or s <= 0 for s in priorities):
        raise ValueError("Importance priorities must be finite and positive")
    norm = math.fsum(a * b for a, b in zip(p, priorities, strict=True))
    q = tuple(
        (1 - priority_mix) * a + priority_mix * a * b / norm
        for a, b in zip(p, priorities, strict=True)
    )
    if not math.isclose(math.fsum(p), 1.0, abs_tol=1e-12):
        raise AssertionError("Invalid target distribution")
    if not math.isclose(math.fsum(q), 1.0, abs_tol=1e-12):
        raise AssertionError("Invalid proposal distribution")
    return ImportanceDistribution(
        ordered, p, q, priorities, target_balance, priority_mix,
    )


def importance_diagnostics(draws: Sequence[ImportanceDraw]) -> dict[str, float]:
    if not draws:
        raise ValueError("Cannot diagnose empty importance batch")
    weights = [d.weight for d in draws]
    total = math.fsum(weights)
    ess = total * total / math.fsum(w * w for w in weights)
    return {
        "weight_min": min(weights), "weight_max": max(weights),
        "weight_mean": total / len(weights),
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / len(weights),
    }
