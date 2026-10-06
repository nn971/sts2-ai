from __future__ import annotations

from typing import Protocol

from sts2_ai.emulator import Observation


class StateEvaluator(Protocol):
    """Assign a scalar value to one player-facing observation."""

    def evaluate(self, observation: Observation, *, terminal: bool) -> float: ...
