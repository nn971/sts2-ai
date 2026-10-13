"""Fixed Nature controls for isolated mechanically legal choices.

Uniform means uniform over primitive index tickets, NOT an assertion about
the original game probability of semantic enemy moves.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from .protocol import ChanceEvent, NatureProposal, canonical_sha256


@dataclass(frozen=True, slots=True)
class UniformNature:
    seed: int
    policy_id: str = "nature-uniform-index-v1"

    def propose(self, event: ChanceEvent) -> NatureProposal:
        # Each occurrence owns its own RNG; worker scheduling cannot change it.
        if type(self.seed) is not int:
            raise ValueError("Nature seed must be an integer")
        material = f"{self.seed}|{event.event_id}".encode()
        local_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
        index = random.Random(local_seed).randrange(event.support_count)
        result = NatureProposal(index, self.policy_id, -math.log(event.support_count))
        result.validate(event)
        return result


@dataclass(frozen=True, slots=True)
class FirstLegalNature:
    policy_id: str = "nature-first-legal-v1"

    def propose(self, event: ChanceEvent) -> NatureProposal:
        proposal = NatureProposal(0, self.policy_id)
        proposal.validate(event)
        return proposal


@dataclass(frozen=True, slots=True)
class LastLegalNature:
    policy_id: str = "nature-last-legal-v1"

    def propose(self, event: ChanceEvent) -> NatureProposal:
        proposal = NatureProposal(event.support_count - 1, self.policy_id)
        proposal.validate(event)
        return proposal


@dataclass(slots=True)
class ScriptedNature:
    """Strict ordered replay of (event id, selected index) pairs."""

    selections: tuple[tuple[str, int], ...]
    policy_id: str = "nature-scripted-index-v1"
    _cursor: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_selections(
        cls, selections: Iterable[tuple[str, int]],
    ) -> ScriptedNature:
        return cls(tuple(selections))

    def propose(self, event: ChanceEvent) -> NatureProposal:
        if self._cursor >= len(self.selections):
            raise ValueError("Nature replay exhausted")
        event_id, index = self.selections[self._cursor]
        if event_id != event.event_id:
            raise ValueError("Nature replay diverged in event/support/order")
        result = NatureProposal(index, self.policy_id)
        result.validate(event)
        self._cursor += 1
        return result

    def assert_complete(self) -> None:
        if self._cursor != len(self.selections):
            raise ValueError("Unconsumed scripted Nature outcomes")


def enumerate_finite_choices(
    event: ChanceEvent, *, limit: int = 32,
) -> tuple[NatureProposal, ...]:
    """Bounded exhaustive legal index set; reject enormous shuffles/tickets."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("Enumeration limit must be positive")
    if event.support_count > limit:
        raise ValueError("Nature support exceeds safe enumeration limit")
    return tuple(
        NatureProposal(i, "nature-enumerated-index-v1")
        for i in range(event.support_count)
    )


def policy_fingerprint(policy_id: str, config: dict) -> str:
    if not policy_id:
        raise ValueError("Missing Nature policy identity")
    return canonical_sha256([policy_id, config])
