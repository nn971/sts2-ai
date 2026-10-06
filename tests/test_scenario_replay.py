from __future__ import annotations

import pytest

from sts2_ai.scenarios import ReplayScenario, replay_scenario
from sts2_ai.testing import MockLinearBackend


def _scenario(backend: MockLinearBackend, expected_hash: str) -> ReplayScenario:
    return ReplayScenario(
        scenario_id="replay-test",
        run_seed="ignored",
        action_history=("inc-2", "inc-1"),
        decision_index=2,
        exact_state_hash=expected_hash,
        game_build="test",
        emulator_revision=backend.emulator_revision,
        information_policy="fair-test",
        reason="test",
        value_margin=0.0,
        max_uncertainty=None,
        actions=(),
    )


def test_replay_scenario_reconstructs_exact_hash() -> None:
    backend = MockLinearBackend(terminal_at=10)
    expected_hash = backend.exact_hash("3")

    state = replay_scenario(backend, _scenario(backend, expected_hash))
    try:
        assert state == "3"
        assert backend.exact_hash(state) == expected_hash
    finally:
        backend.release_many((state,))


def test_replay_scenario_rejects_hash_mismatch() -> None:
    backend = MockLinearBackend(terminal_at=10)

    with pytest.raises(RuntimeError, match="replay hash mismatch"):
        replay_scenario(backend, _scenario(backend, "wrong"))
