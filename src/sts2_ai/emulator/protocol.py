from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

type StateHandle = str


@dataclass(frozen=True, slots=True)
class LegalAction:
    """Binding-neutral description of one mechanically legal action."""

    action_id: str
    kind: str
    payload_json: str = "{}"


@dataclass(frozen=True, slots=True)
class InformationPolicy:
    """Versioned declaration of what an agent is allowed to observe."""

    policy_id: str


@dataclass(frozen=True, slots=True)
class Observation:
    policy_id: str
    payload_json: str
    observation_hash: str


@dataclass(frozen=True, slots=True)
class Transition:
    parent: StateHandle
    action: LegalAction
    child: StateHandle
    terminal: bool
    exact_hash: str | None = None


@dataclass(frozen=True, slots=True)
class StepFrame:
    """A transition bundled with the resulting player-facing frame."""

    transition: Transition
    observation: Observation
    legal_actions: tuple[LegalAction, ...]


class EmulatorBackend(Protocol):
    """Research-facing contract expected from sts2-emulator bindings."""

    @property
    def binding_version(self) -> str: ...

    @property
    def emulator_revision(self) -> str: ...

    def reset(self, seed: str, ascension: int = 0) -> StateHandle: ...

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]: ...

    def step(self, state: StateHandle, action: LegalAction) -> Transition: ...

    def batch_step(
        self,
        items: Sequence[tuple[StateHandle, LegalAction]],
    ) -> tuple[Transition, ...]: ...

    def batch_step_frame(
        self,
        items: Sequence[tuple[StateHandle, LegalAction]],
        policy: InformationPolicy,
    ) -> tuple[StepFrame, ...]: ...

    def batch_rollout_step_frame(
        self,
        items: Sequence[tuple[StateHandle, LegalAction]],
        policy: InformationPolicy,
    ) -> tuple[StepFrame, ...]: ...

    def fork(self, state: StateHandle) -> StateHandle: ...

    def expand(self, state: StateHandle) -> Sequence[Transition]: ...

    def batch_expand(
        self,
        states: Sequence[StateHandle],
    ) -> tuple[tuple[Transition, ...], ...]: ...

    def release_many(self, states: Sequence[StateHandle]) -> int: ...

    def exact_hash(self, state: StateHandle) -> str: ...

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation: ...

    def batch_observe(
        self,
        states: Sequence[StateHandle],
        policy: InformationPolicy,
    ) -> tuple[Observation, ...]: ...

    def is_terminal(self, state: StateHandle) -> bool: ...

    def close(self) -> None: ...
