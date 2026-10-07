from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sts2_ai.emulator import LegalAction, Observation, StateHandle


@dataclass(frozen=True, slots=True)
class Decision:
    action: LegalAction
    policy_name: str
    metadata_json: str = "{}"


class Agent(Protocol):
    def choose(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> Decision: ...


@runtime_checkable
class ExactStateAgent(Protocol):
    """Agent that intentionally receives an exact emulator handle.

    This interface is for explicitly labelled oracle-exact search. Fair agents should
    implement Agent and consume only the observation plus legal actions.
    """

    def choose_state(
        self,
        state: StateHandle,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> Decision: ...
