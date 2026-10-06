from __future__ import annotations

import random
from collections.abc import Sequence

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation


class RandomAgent:
    """Deterministic-under-seed baseline useful for smoke testing plumbing."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose(self, observation: Observation, legal_actions: Sequence[LegalAction]) -> Decision:
        del observation
        if not legal_actions:
            raise ValueError("Cannot choose from an empty legal-action set")
        action = self._rng.choice(list(legal_actions))
        return Decision(action=action, policy_name="random-v1")
