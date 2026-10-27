"""Schedule-stable semantic Nature proposals and fail-closed replay."""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from .semantic import SemanticEvent, SemanticProposal


def _rng(seed: int, event: SemanticEvent) -> random.Random:
    if type(seed) is not int:
        raise ValueError("Semantic Nature seed must be an integer")
    material = f"{seed}|{event.replay_identity}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


@dataclass(frozen=True, slots=True)
class UniformSemanticNature:
    seed: int
    policy_id: str = "nature-semantic-uniform-v1"

    def propose(self, event: SemanticEvent) -> SemanticProposal:
        eligible = (
            tuple(i for i, w in enumerate(event.raw_weights or ()) if w > 0)
            if event.operation == "enemy_intent"
            else tuple(range(event.support_count))
        )
        index = eligible[_rng(self.seed, event).randrange(len(eligible))]
        result = SemanticProposal(
            index, event.candidate_ids[index], self.policy_id, -math.log(len(eligible)),
        )
        result.validate(event)
        return result


@dataclass(frozen=True, slots=True)
class SourceWeightedIntentNature:
    """A *proposal* using emulator-supplied raw weights; not native p calibration."""

    seed: int
    policy_id: str = "nature-semantic-source-weight-proposal-v1"

    def propose(self, event: SemanticEvent) -> SemanticProposal:
        weights = event.normalized_intent_weights
        draw = _rng(self.seed, event).random()
        cumulative = 0.0
        selected = None
        for i, p in enumerate(weights):
            if p <= 0:
                continue
            cumulative += p
            if draw < cumulative:
                selected = i
                break
        if selected is None:
            selected = next(i for i in reversed(range(len(weights))) if weights[i] > 0)
        result = SemanticProposal(
            selected, event.candidate_ids[selected], self.policy_id,
            math.log(weights[selected]),
        )
        result.validate(event)
        return result


@dataclass(slots=True)
class ScriptedSemanticNature:
    """Replay exact ordered (event replay identity, selected instance ID)."""

    decisions: tuple[tuple[str, str], ...]
    policy_id: str = "nature-semantic-scripted-v1"
    _cursor: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_decisions(
        cls, decisions: Iterable[tuple[str, str]],
    ) -> ScriptedSemanticNature:
        return cls(tuple(decisions))

    def propose(self, event: SemanticEvent) -> SemanticProposal:
        if self._cursor >= len(self.decisions):
            raise ValueError("Semantic replay exhausted")
        replay_identity, selected_id = self.decisions[self._cursor]
        if event.replay_identity != replay_identity:
            raise ValueError("Semantic event, support or constraints diverged")
        try:
            selected_index = event.candidate_ids.index(selected_id)
        except ValueError as exc:
            raise ValueError("Selected instance/intent disappeared from support") from exc
        result = SemanticProposal(selected_index, selected_id, self.policy_id)
        result.validate(event)
        self._cursor += 1
        return result

    def assert_complete(self) -> None:
        if self._cursor != len(self.decisions):
            raise ValueError("Unconsumed semantic replay decisions")
