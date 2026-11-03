"""H18 generated Overgrowth A0 routing and experiment isolation."""
from __future__ import annotations

import json
import random

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition
from sts2_ai.emulator.run_environment import (
    EXPERIMENTAL_FIXED_ROUTE_PROFILE,
    NATIVE_MAP_PROFILE,
    NATIVE_OVERGROWTH,
    NATIVE_RESET_SCHEMA,
    reset_training_run,
)
from sts2_ai.training.h16_joint_act_pilot import collect_act_episode
from tools.train_h16_joint_act_pilot import seed_namespace, selected_map_profile


def _observation(frame: dict) -> Observation:
    return Observation(FAIR_POLICY_ID, json.dumps(frame), "public-hash")


def _frame(*, hp: int = 70, map_column: int = 0) -> dict:
    return {
        "act": 1, "floor": 0, "hp": hp, "max_hp": 70,
        "completed_rooms": [], "combat": None,
        "deck": [], "relics": [], "potions": [],
        "map": [
            {"node_id": f"1:1:{map_column}", "floor": 1, "next_node_ids": ["1:2:0"]},
            {"node_id": "1:2:0", "floor": 2, "next_node_ids": ["1:16:3"]},
        ],
    }


def test_map_selection_and_seed_namespace_keep_h17_unchanged() -> None:
    assert selected_map_profile("generated") is None
    assert selected_map_profile("fixed-route") == EXPERIMENTAL_FIXED_ROUTE_PROFILE
    assert seed_namespace(10, "fixed-route") == "h16-fixedroute"
    assert seed_namespace(0, "generated") == "h18-generated-a0"
    assert seed_namespace(10, "generated") != seed_namespace(0, "generated")
    assert seed_namespace(0, "fixed-route") != seed_namespace(0, "generated")
    with pytest.raises(ValueError, match="Unknown Act 1 map"):
        selected_map_profile("invented")
    with pytest.raises(ValueError, match="Ascension"):
        seed_namespace(11, "generated")


def test_generated_a0_reset_invokes_existing_default_native_operation() -> None:
    class FakeNativeBackend:
        native_overgrowth_reset_schema = NATIVE_RESET_SCHEMA

        def __init__(self):
            self.reset_args = []
            self.released = []

        # Deliberately no map_profile_id keyword: generated mode must omit it.
        def reset_native_overgrowth(self, seed, ascension=0):
            self.reset_args.append((seed, ascension))
            return "generated-start"

        def observe(self, state, policy):
            frame = _frame()
            frame.update(
                map_generation_profile_id=NATIVE_MAP_PROFILE,
                event_id="proto.native.event.neow",
                map=[{"floor": 1, "node_id": "1:1:0"},
                     {"floor": 16, "node_id": "1:16:3"}],
                deck=[{"card_id": "proto.silent.strike"}] * 12
                + [{"card_id": "proto.common.restlessness"}],
            )
            return _observation(frame)

        def release_many(self, states):
            self.released.extend(states)
            return len(states)

    backend = FakeNativeBackend()
    state = reset_training_run(
        backend, "h18-generated-a0-sample", NATIVE_OVERGROWTH, 0,
        map_profile_id=selected_map_profile("generated"),
    )
    assert state == "generated-start"
    assert backend.reset_args == [("h18-generated-a0-sample", 0)]
    assert backend.released == []


@pytest.mark.parametrize("map_profile", [None, EXPERIMENTAL_FIXED_ROUTE_PROFILE])
def test_episode_collector_propagates_map_selection(monkeypatch, map_profile) -> None:
    import sts2_ai.training.h16_joint_act_pilot as pilot

    observed = []

    def reset_spy(backend, seed, environment, ascension, *, map_profile_id):
        observed.append((environment, ascension, map_profile_id))
        return "map-choice"

    monkeypatch.setattr(pilot, "reset_training_run", reset_spy)

    class MapBackend:
        def observe(self, state, policy):
            return _observation(_frame(hp=70 if state == "map-choice" else 0))

        def legal_actions(self, state):
            assert state == "map-choice"
            return (
                LegalAction("left", "choose_map_node", '{"NodeId":"1:1:0"}'),
                LegalAction("right", "choose_map_node", '{"NodeId":"1:1:1"}'),
            )

        def is_terminal(self, state):
            return state == "defeat"

        def step(self, state, action):
            assert state == "map-choice" and action.kind == "choose_map_node"
            return Transition(state, action, "defeat", True)

        def release_many(self, states):
            return len(states)

    class Policy:
        temperature = 0.85

        def infer(self, observation, actions):
            return (tuple(0.0 for _ in actions), 0.2,
                    tuple(0.0 for _ in actions))

    episode = collect_act_episode(
        MapBackend(), Policy(), seed="generated-a0-smoke", ascension=0,
        map_profile_id=map_profile, max_decisions=4,
        search_probability=0.0, max_searches=0,
        simulations=16, search_depth=None, rng=random.Random(1),
        rollout_transport="legacy",
    )
    assert observed == [(NATIVE_OVERGROWTH, 0, map_profile)]
    assert episode.outcome == "defeat"
    assert len(episode.decisions) == 1
    assert episode.decisions[0].legal_actions[
        episode.decisions[0].chosen_index
    ].kind == "choose_map_node"


def test_worker_passes_generated_map_and_a0_into_collector(monkeypatch):
    import tools.train_h16_joint_act_pilot as trainer
    from sts2_ai.training.h16_joint_act_pilot import PilotEpisode

    class Backend:
        def reset_operation_profile(self):
            pass

        def operation_profile(self):
            return {}

    class Policy:
        inference_calls = 0
        inference_seconds = 0.0

    expected = PilotEpisode(
        "h18-seed", (), (_frame(),), "defeat", 0, 0,
    )
    seen = []

    def fake_collect(backend, policy, **kwargs):
        seen.append(kwargs)
        return expected

    config = {
        "ascension": 0,
        "map_profile_id": None,
        "study_seed": 20261120,
        "max_decisions": 4096,
        "search_probability": 0.0,
        "max_searches": 0,
        "simulations": 16,
        "search_depth": None,
        "rollout_transport": "fused",
        "max_simulation_steps": 512,
        "adaptive_search": None,
        "card_exposure": {},
    }
    monkeypatch.setattr(
        trainer, "_WORKER",
        {"backend": Backend(), "policy": Policy(), "config": config},
    )
    monkeypatch.setattr(trainer, "collect_act_episode", fake_collect)
    result = trainer._worker_run("h18-seed")
    assert result.outcome == "defeat"
    assert seen[0]["ascension"] == 0
    assert seen[0]["map_profile_id"] is None
    assert seen[0]["seed"] == "h18-seed"
