"""Pinned bridge test for the first eligible full-game reward boundary."""
from __future__ import annotations

import shutil

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlBridgeError
from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend


def test_pristine_reward_transition_matches_ordinary_engine_replay() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned hypothetical reward bridge requires dotnet")

    streams = {
        "map": 11, "combat": 29, "combat_targets": 5,
        "reward": 7, "shop": 9, "event": 13,
    }
    with JsonlEmulatorBackend(build=True) as backend:
        assert backend.pristine_reward_proposal_schema == (
            "prototype-pristine-reward-branch-v1"
        )
        policy = InformationPolicy(FAIR_POLICY_ID)
        state = backend.reset_factorized_hypothetical(streams)
        owned = [state]
        try:
            with pytest.raises(JsonlBridgeError):
                backend.propose_pristine_reward(
                    state, backend.legal_actions(state)[0],
                    reward_initial_state=7,
                )
            found_reward = False
            for step in range(100):
                menu = tuple(backend.legal_actions(state))
                assert menu, "Fixture terminated before reward"
                before_phase = int(__import__("json").loads(
                    backend.observe(state, policy).payload_json
                )["phase"])
                if before_phase == 1:
                    action = next(a for a in menu if a.kind == "start_run")
                elif before_phase == 2:
                    action = next(a for a in menu if a.kind == "choose_map_node")
                else:
                    action = next(
                        (a for a in menu if a.kind == "play_card"),
                        next((a for a in menu if a.kind == "end_turn"), menu[0]),
                    )
                child = backend.step(state, action).child
                owned.append(child)
                after_phase = int(__import__("json").loads(
                    backend.observe(child, policy).payload_json
                )["phase"])
                if before_phase == 3 and after_phase == 5:
                    replay = backend.propose_pristine_reward(
                        state, action, reward_initial_state=7,
                    )
                    owned.append(replay)
                    assert backend.exact_hash(replay) == backend.exact_hash(child)
                    assert backend.observe(replay, policy) == backend.observe(child, policy)
                    assert tuple(backend.legal_actions(replay)) == tuple(
                        backend.legal_actions(child)
                    )
                    other = backend.propose_pristine_reward(
                        state, action, reward_initial_state=9,
                    )
                    owned.append(other)
                    assert int(__import__("json").loads(
                        backend.observe(other, policy).payload_json
                    )["phase"]) == 5
                    with pytest.raises(JsonlBridgeError):
                        backend.propose_pristine_reward(
                            child, action, reward_initial_state=7,
                        )
                    found_reward = True
                    break
                state = child
            assert found_reward, "Fixture never reached first reward boundary"
        finally:
            backend.release_many(owned)
