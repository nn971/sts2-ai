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


class EmulatorBackend(Protocol):
    """Minimal research-facing contract expected from sts2-emulator bindings."""

    @property
    def binding_version(self) -> str: ...

    @property
    def emulator_revision(self) -> str: ...

    def reset(self, seed: str) -> StateHandle: ...

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]: ...

    def step(self, state: StateHandle, action: LegalAction) -> Transition: ...

    def expand(self, state: StateHandle) -> Sequence[Transition]: ...

    def fork(self, state: StateHandle) -> StateHandle: ...

    def exact_hash(self, state: StateHandle) -> str: ...

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation: ...

    def is_terminal(self, state: StateHandle) -> bool: ...
