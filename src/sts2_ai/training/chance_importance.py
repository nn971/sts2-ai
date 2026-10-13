"""Chance-event importance correction, separate from snapshot sampling v15o.

Only exact synthetic laws or independently audited native reference laws can
supply p. UNVERIFIED emulator request metadata is never a probability law.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

REFERENCE_STATUSES = frozenset({"SYNTHETIC_EXACT", "AUDITED_NATIVE"})


@dataclass(frozen=True, slots=True)
class ChanceLaw:
    law_id: str
    event_id: str
    probabilities: tuple[float, ...]
    status: str

    def __post_init__(self) -> None:
        if not self.law_id or not self.event_id:
            raise ValueError("Reference law requires identity")
        if self.status not in REFERENCE_STATUSES:
            raise ValueError("Unverified/unsupported reference chance law")
        _check_probabilities(self.probabilities, full_support=False)


def _check_probabilities(probs: Sequence[float], *, full_support: bool) -> None:
    if not probs or any(
        not isinstance(p, (float, int)) or not math.isfinite(p)
        or p < 0 or (full_support and p <= 0)
        for p in probs
    ):
        raise ValueError("Invalid chance probability support")
    if not math.isclose(math.fsum(probs), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Chance probabilities do not sum to one")


def log_chance_ratio(
    reference: ChanceLaw, proposal: Sequence[float], *,
    event_id: str, selected_index: int,
) -> float:
    """log(p/q), rejecting missing support and unverified reference laws."""
    if reference.event_id != event_id:
        raise ValueError("Chance law event/provenance mismatch")
    _check_probabilities(proposal, full_support=False)
    if len(proposal) != len(reference.probabilities):
        raise ValueError("Chance proposal and target support differ")
    if any(p > 0 and q <= 0 for p, q in
           zip(reference.probabilities, proposal, strict=True)):
        raise ValueError("Proposal lacks positive target support")
    if type(selected_index) is not int or not 0 <= selected_index < len(proposal):
        raise ValueError("Selected chance index outside support")
    if proposal[selected_index] <= 0:
        raise ValueError("Selected chance outcome has zero proposal probability")
    p = reference.probabilities[selected_index]
    return math.log(p) - math.log(proposal[selected_index]) if p else -math.inf


def trajectory_log_weight(
    steps: Sequence[tuple[ChanceLaw, Sequence[float], str, int]],
) -> float:
    """Complete trajectory chance weight, with no PPO/minibatch normalization."""
    return math.fsum(
        log_chance_ratio(p, q, event_id=event_id, selected_index=index)
        for p, q, event_id, index in steps
    )


def effective_sample_size(log_weights: Sequence[float]) -> float:
    """Numerically stable trajectory ESS; zero-weight trajectories are allowed."""
    if not log_weights or any(
        not isinstance(w, (float, int)) or math.isnan(w) or w == math.inf
        for w in log_weights
    ):
        raise ValueError("Invalid log-weight population")
    finite = [w for w in log_weights if w != -math.inf]
    if not finite:
        raise ValueError("All importance weights are zero")
    center = max(finite)
    weights = [math.exp(w - center) if w != -math.inf else 0.0 for w in log_weights]
    return math.fsum(weights) ** 2 / math.fsum(x * x for x in weights)


def mixture_proposal(
    reference: ChanceLaw, learned: Sequence[float], *, alpha: float,
) -> tuple[float, ...]:
    if not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0, 1)")
    if len(learned) != len(reference.probabilities):
        raise ValueError("Mismatched chance proposal support")
    _check_probabilities(learned, full_support=False)
    combined = tuple(
        (1 - alpha) * p + alpha * q
        for p, q in zip(reference.probabilities, learned, strict=True)
    )
    # Support of q must include every event with nonzero p.
    if any(p > 0 and q <= 0 for p, q in
           zip(reference.probabilities, combined, strict=True)):
        raise ValueError("Proposal lost positive reference support")
    return combined
