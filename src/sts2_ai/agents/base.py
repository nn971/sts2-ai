from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from sts2_ai.emulator import LegalAction, Observation


@dataclass(frozen=True, slots=True)
class Decision:
    action: LegalAction
    policy_name: str
    metadata_json: str = "{}"


class Agent(Protocol):
    def choose(self, observation: Observation, legal_actions: Sequence[LegalAction]) -> Decision: ...
