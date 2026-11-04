"""CPU-only preflight contracts; exact public equality is weaker than RNG receipts."""
from __future__ import annotations

import hashlib
import json

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation
from sts2_ai.emulator.protocol import StepFrame, Transition
from tools.probe_p2_exact_fork_prefix import verify_exact_fork_prefix


class NativeForkStub:
    def __init__(self, *, drift: str | None = None):
        self.states: dict[str, tuple[int, str]] = {"root": (0, "rng-0")}
        self.next_id = 0
        self.drift = drift
        self.fork_count = 0
        self.released: list[str] = []

    def _create(self, value: tuple[int, str]) -> str:
        self.next_id += 1
        key = f"handle-{self.next_id}"
        self.states[key] = value
        return key

    def exact_hash(self, state: str) -> str:
        depth, hidden = self.states[state]
        return hashlib.sha256(f"{depth}|{hidden}".encode()).hexdigest()

    def observe(self, state: str, policy) -> Observation:
        assert policy.policy_id == FAIR_POLICY_ID
        depth, _ = self.states[state]
        payload = json.dumps({"combat": {"turn": depth}, "hp": 70})
        return Observation(
            policy_id=FAIR_POLICY_ID, payload_json=payload,
            observation_hash=hashlib.sha256(payload.encode()).hexdigest(),
        )

    def legal_actions(self, state: str):
        return (
            (LegalAction("first", "play_card", "{}"),)
            if self.states[state][0] < 2 else ()
        )

    def is_terminal(self, state: str) -> bool:
        return self.states[state][0] >= 2

    def fork(self, state: str) -> str:
        self.fork_count += 1
        depth, hidden = self.states[state]
        if self.drift == "fork" and self.fork_count == 2:
            hidden = "wrong-private-rng"
        return self._create((depth, hidden))

    def batch_rollout_step_frame(self, items, policy):
        answer = []
        for pos, (source, action) in enumerate(items):
            depth, hidden = self.states[source]
            assert action == self.legal_actions(source)[0]
            if self.drift == "step" and pos == 1:
                hidden = "different-hidden-rng-after-draw"
            child = self._create((depth + 1, hidden))
            answer.append(StepFrame(
                Transition(source, action, child, self.is_terminal(child)),
                self.observe(child, policy), self.legal_actions(child),
            ))
        return tuple(answer)

    def release_many(self, handles):
        for h in handles:
            assert h in self.states
            assert h != "root"
            del self.states[h]
            self.released.append(h)
        return len(handles)


def test_exact_fork_prefix_two_steps_and_unconditional_cleanup():
    backend = NativeForkStub()
    result = verify_exact_fork_prefix(backend, "root", max_steps=8)
    assert result["verified"]
    assert result["steps_checked"] == 2
    assert result["forks_checked"] == 4
    assert result["owned_handles_released"] == 8
    assert result["trace"][-1]["terminal"] is True
    assert result["independent_rng_receipt_available"] is False
    assert result["ready_for_p2_1_caching"] is False
    assert backend.states == {"root": (0, "rng-0")}


@pytest.mark.parametrize(("mode", "pattern"), [
    ("fork", "hash diverged on fork"),
    ("step", "hash diverged after action"),
])
def test_identical_public_projection_does_not_hide_private_drift(mode, pattern):
    backend = NativeForkStub(drift=mode)
    with pytest.raises(ValueError, match=pattern):
        verify_exact_fork_prefix(backend, "root", max_steps=2)
    assert backend.states == {"root": (0, "rng-0")}
    assert len(backend.released) > 0


def test_preflight_has_bounded_steps_and_never_releases_input_root():
    backend = NativeForkStub()
    result = verify_exact_fork_prefix(backend, "root", max_steps=1)
    assert result["steps_checked"] == 1
    assert result["trace"][-1]["terminal"] is False
    assert backend.states == {"root": (0, "rng-0")}
    with pytest.raises(ValueError, match="positive integer"):
        verify_exact_fork_prefix(backend, "root", max_steps=0)


def test_stale_fused_observation_fails_with_cleanup():
    class StaleFrame(NativeForkStub):
        def batch_rollout_step_frame(self, items, policy):
            frames = list(super().batch_rollout_step_frame(items, policy))
            second = frames[1]
            frames[1] = StepFrame(
                second.transition, self.observe("root", policy),
                second.legal_actions,
            )
            return tuple(frames)

    backend = StaleFrame()
    with pytest.raises(ValueError, match="Public continuation"):
        verify_exact_fork_prefix(backend, "root", max_steps=2)
    assert backend.states == {"root": (0, "rng-0")}
