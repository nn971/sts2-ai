"""Reference-law importance correction for joint player/Nature PPO.

For a fixed player policy, enemy intents and player actions have identical
conditional probabilities under the reference law p and learned Nature q;
only deferred physical draws contribute a likelihood ratio. The H3 collector
uses a deterministic allocation of source worlds to p and q arms with known
relative cohort shares alpha and 1-alpha. Balance-heuristic multiple IS is
the marginal likelihood ratio for that mixture:

    w(tau) = p(tau) / (alpha*p(tau) + (1-alpha)*q(tau)).

It is positive and bounded by 1/alpha. It is **not** the raw p/q
product and does not require arbitrary ratio clipping. Reference-arm
trajectories MUST also have q scored at every physical draw.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


class NatureImportanceError(ValueError):
    """Invalid provenance for reference-law correction."""


def log_q_over_p(episode: Any) -> tuple[float, int]:
    """Compute full-path draw-only log density ratio from a genuine transcript."""
    if episode.draw_mode not in ("learned", "reference"):
        raise NatureImportanceError("Unknown Nature draw arm")
    if not episode.terminal_replay_verified:
        raise NatureImportanceError("Cannot importance-weight censored combat")
    total = 0.0
    count = 0
    for step in episode.decisions:
        if not step.phase.startswith("nature:"):
            continue
        if step.phase not in ("nature:draw_card", "nature:enemy_intent"):
            raise NatureImportanceError("Unknown controlled stochastic site")
        if step.log_p is None or step.log_q is None:
            raise NatureImportanceError("Both p and q logs required on every Nature site")
        if not (math.isfinite(step.log_p) and math.isfinite(step.log_q)
                and step.log_p <= 1e-10 and step.log_q <= 1e-10):
            raise NatureImportanceError("Invalid selected-outcome log probabilities")
        if step.phase == "nature:enemy_intent":
            if abs(step.log_q - step.log_p) > 1e-9:
                raise NatureImportanceError("Enemy intent law unexpectedly changed")
        else:
            count += 1
            total += step.log_q - step.log_p
    if count != episode.controlled_draw_count:
        raise NatureImportanceError("Missing physical draw from likelihood transcript")
    return total, count


def balance_weight(log_qp: float, reference_fraction: float) -> float:
    """Bounded likelihood ratio using a numerically stable mixture denominator."""
    if not (math.isfinite(log_qp) and 0 < reference_fraction < 1):
        raise NatureImportanceError("Invalid q/p density ratio or arm mixture")
    alpha = reference_fraction
    if log_qp >= 0:
        rho = math.exp(-log_qp)
        weight = rho / (alpha * rho + 1 - alpha)
    else:
        rho = math.exp(log_qp)
        weight = 1 / (alpha + (1 - alpha) * rho)
    if not (math.isfinite(weight) and 0 < weight <= 1 / alpha + 1e-10):
        raise NatureImportanceError("Nonfinite balance-heuristic importance weight")
    return weight


def reference_target_weights(
    episodes: Sequence[Any], *, reference_count: int,
    min_effective_fraction: float = 0.1,
) -> tuple[tuple[float, ...], dict[str, float]]:
    """One positive weight per combat; fail closed if variance is excessive.

    The reference-only fallback is a valid on-distribution PPO update, but
    uses fewer player samples. It is explicitly reported, never silently
    confused with successful adversarial experience reuse.
    """
    count = len(episodes)
    if (count < 2 or not 0 < reference_count < count
            or not 0 < min_effective_fraction <= 1):
        raise NatureImportanceError("Invalid mixed-arm importance sample")
    observed_reference = sum(ep.draw_mode == "reference" for ep in episodes)
    if observed_reference != reference_count:
        raise NatureImportanceError("Reference arm allocation does not match provenance")
    alpha = reference_count / count
    logs = [log_q_over_p(ep)[0] for ep in episodes]
    weights = [balance_weight(v, alpha) for v in logs]
    total = math.fsum(weights)
    squares = math.fsum(w * w for w in weights)
    ess = total * total / squares
    fallback = ess / count < min_effective_fraction
    if fallback:
        # Reference-law samples require no IS correction; reject
        # high-variance adversarial samples for the player only.
        weights = [1.0 if ep.draw_mode == "reference" else 0.0 for ep in episodes]
        ess = float(reference_count)
    return tuple(weights), {
        "player_is_effective_samples": ess,
        "player_is_effective_fraction": ess / count,
        "player_is_reference_only_fallback": float(fallback),
        "player_is_raw_mean_weight": total / count,
        "player_is_raw_max_weight": max(balance_weight(v, alpha) for v in logs),
        "player_is_fraction_reference": alpha,
        "player_is_nonreference_used": float(sum(
            ep.draw_mode == "learned" and w > 0
            for ep, w in zip(episodes, weights, strict=True)
        )),
        "player_is_mean_log_q_over_p": math.fsum(logs) / count,
    }
