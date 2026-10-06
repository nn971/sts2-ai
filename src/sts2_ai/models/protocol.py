from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sts2_ai.emulator import LegalAction, Observation


@dataclass(frozen=True, slots=True)
class PolicyValueEstimate:
    action_logits: tuple[float, ...]
    value: float


class PolicyValueModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def evaluate(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate: ...
