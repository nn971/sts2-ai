"""Read-only public chance laws and the fail-closed fair-root contract.

The draw law is valid when the supplied card-type counts really are the
remaining *unknown, exchangeable* draw pile conditioned on public history.
Known top cards, ordered observations and hidden correlations require a richer
conditioning model before this law can be used as a run-level sampler.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Protocol, runtime_checkable

from .protocol import LegalAction, Observation, StateHandle

CHANCE_LAW_VERSION = "known-deck-draw-v1"
FAIR_CONTINUATION_CAPABILITY_ID = "history-conditioned-fair-v1"


class FairContinuationUnavailable(RuntimeError):
    """The backend has no validated public-history-consistent sampler."""


@dataclass(frozen=True, slots=True)
class PublicHistoryStep:
    observation: Observation
    chosen_action: LegalAction | None


@runtime_checkable
class FairContinuationSampler(Protocol):
    """Optional emulator capability; each root conditions on complete public history.

    Randomness is supplied independently by search; no RNG cursor from a
    live engine state is an acceptable implementation of this protocol.
    """

    @property
    def fair_continuation_capability_id(self) -> str: ...

    def sample_fair_continuations(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        count: int,
    ) -> tuple[StateHandle, ...]: ...


def require_fair_sampler(backend: object) -> FairContinuationSampler:
    """Fail closed rather than silently substituting oracle-exact forks."""
    if (
        not isinstance(backend, FairContinuationSampler)
        or backend.fair_continuation_capability_id != FAIR_CONTINUATION_CAPABILITY_ID
    ):
        raise FairContinuationUnavailable(
            "Backend has no validated history-conditioned fair sampler. "
            "Exact-state fork/expand and reseeding the live hidden state are oracle-only."
        )
    return backend


@dataclass(frozen=True, slots=True)
class DrawOutcome:
    """Unordered draw multiplicities; cards remain ordered lexicographically."""

    counts: tuple[tuple[str, int], ...]
    probability: float


@dataclass(frozen=True, slots=True)
class KnownDeckDrawLaw:
    """Multivariate-hypergeometric draws from a publicly justified multiset."""

    counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> KnownDeckDrawLaw:
        for card, count in counts.items():
            if not isinstance(card, str) or not card:
                raise ValueError("card identifiers must be nonempty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("card counts must be nonnegative integers")
        return cls(tuple(sorted((key, value) for key, value in counts.items() if value)))

    @property
    def remaining(self) -> int:
        return sum(count for _, count in self.counts)

    def _validate_draw_count(self, draws: int) -> None:
        if isinstance(draws, bool) or not isinstance(draws, int):
            raise ValueError("draw count must be an integer")
        if draws < 0 or draws > self.remaining:
            raise ValueError("draw count must be between zero and remaining cards")

    def distribution(self, draws: int, *, max_outcomes: int = 4096) -> tuple[DrawOutcome, ...]:
        """Enumerate exact small-support multiset probabilities, or refuse."""
        self._validate_draw_count(draws)
        if max_outcomes < 1:
            raise ValueError("max_outcomes must be positive")
        denom = math.comb(self.remaining, draws)
        outcomes: list[DrawOutcome] = []
        ranges = [range(min(count, draws) + 1) for _, count in self.counts]
        for drawn in product(*ranges):
            if sum(drawn) != draws:
                continue
            if len(outcomes) >= max_outcomes:
                raise ValueError("draw support exceeds max_outcomes; use sampling")
            numerator = math.prod(
                math.comb(available, taken)
                for (_, available), taken in zip(self.counts, drawn, strict=True)
            )
            outcomes.append(
                DrawOutcome(
                    counts=tuple(
                        (card, count)
                        for (card, _), count in zip(self.counts, drawn, strict=True)
                        if count
                    ),
                    probability=numerator / denom,
                )
            )
        return tuple(outcomes)

    def sample_ordered(self, draws: int, *, rng: random.Random) -> tuple[str, ...]:
        """Draw without replacement using exclusively the caller's search RNG."""
        self._validate_draw_count(draws)
        available = [count for _, count in self.counts]
        result: list[str] = []
        remaining = self.remaining
        for _ in range(draws):
            ticket = rng.randrange(remaining)
            for index, (card, _) in enumerate(self.counts):
                if ticket < available[index]:
                    result.append(card)
                    available[index] -= 1
                    break
                ticket -= available[index]
            remaining -= 1
        return tuple(result)
