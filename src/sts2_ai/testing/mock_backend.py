from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from sts2_ai.emulator import (
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
    Transition,
)


class MockLinearBackend:
    """Tiny deterministic backend for parent-repo tests.

    State handles are integers encoded as strings. Each nonterminal state offers +1 and +2.
    This is deliberately unrelated to Slay the Spire mechanics.
    """

    def __init__(self, terminal_at: int = 5) -> None:
        self._terminal_at = terminal_at

    @property
    def binding_version(self) -> str:
        return "mock-v1"

    @property
    def emulator_revision(self) -> str:
        return "mock"

    def reset(self, seed: str) -> StateHandle:
        del seed
        return "0"

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]:
        if self.is_terminal(state):
            return ()
        return (
            LegalAction(action_id="inc-1", kind="increment", payload_json='{"amount":1}'),
            LegalAction(action_id="inc-2", kind="increment", payload_json='{"amount":2}'),
        )

    def step(self, state: StateHandle, action: LegalAction) -> Transition:
        value = int(state)
        amount = json.loads(action.payload_json)["amount"]
        child = str(value + int(amount))
        return Transition(
            parent=state,
            action=action,
            child=child,
            terminal=self.is_terminal(child),
        )

    def fork(self, state: StateHandle) -> StateHandle:
        return str(state)

    def exact_hash(self, state: StateHandle) -> str:
        return hashlib.sha256(state.encode()).hexdigest()

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation:
        payload = json.dumps({"value": int(state)}, sort_keys=True)
        digest = hashlib.sha256((policy.policy_id + payload).encode()).hexdigest()
        return Observation(
            policy_id=policy.policy_id,
            payload_json=payload,
            observation_hash=digest,
        )

    def is_terminal(self, state: StateHandle) -> bool:
        return int(state) >= self._terminal_at

    def release_many(self, states: Sequence[StateHandle]) -> int:
        return len(states)
