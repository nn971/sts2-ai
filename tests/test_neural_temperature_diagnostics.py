"""Temperature and boss diagnostics use only the fair public interface."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence

import pytest

from sts2_ai.agents.neural_temperature import (
    NeuralTemperatureAgent,
    temperature_probabilities,
)
from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.emulator.run_environment import NATIVE_MAP_PROFILE, NATIVE_RESET_SCHEMA
from sts2_ai.evaluation.neural_temperature import diagnose_neural_temperatures
from sts2_ai.models.neural import NeuralPolicyValueModel

A = LegalAction("first", "end_turn")
B = LegalAction("second", "play_card", '{"card_id":"toy.card"}')


def model() -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": "sts2-neural-policy-value-v2-semantic-action",
        "dimension": 16,
        "hidden": 4,
        "model_id": "toy-diagnostic-model",
        "value_head_trained": False,
        "state_weight": [[0.0] * 16 for _ in range(4)],
        "state_bias": [0.0] * 4,
        "action_weight": [[0.0] * 16 for _ in range(4)],
        "action_bias": [0.0] * 4,
        "policy_weight": [0.0] * 4,
        "policy_bias": 0.0,
        "value_weight": [0.0] * 4,
        "value_bias": 0.0,
    })


class BossToyBackend:
    """Native-shaped public frames with two boss decisions and real defeat."""

    native_overgrowth_reset_schema = NATIVE_RESET_SCHEMA
    emulator_revision = "test-boss-emulator"

    def __init__(self) -> None:
        self.states: dict[str, int] = {}
        self.next_id = 0
        self.exact_state_calls = 0

    def _put(self, stage: int) -> str:
        self.next_id += 1
        handle = f"state-{self.next_id}"
        self.states[handle] = stage
        return handle

    def reset_native_overgrowth(self, _seed: str, ascension: int = 0) -> str:
        assert ascension == 0
        return self._put(0)

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        assert policy.policy_id == "prototype-fair-v0"
        stage = self.states[state]
        if stage == 0:
            payload = {
                "act": 1,
                "floor": 0,
                "hp": 70,
                "max_hp": 70,
                "event_id": "proto.native.event.neow",
                "map_generation_profile_id": NATIVE_MAP_PROFILE,
                "map": [{"node_id": "1:16:0", "floor": 16}],
                "deck": [{"card_id": "proto.silent.strike"}] * 12
                + [{"card_id": "proto.common.restlessness"}],
            }
        else:
            payload = {
                "act": 1,
                "floor": 16,
                "hp": 20 if stage == 1 else (12 if stage == 2 else 0),
                "max_hp": 70,
                "combat": {
                    "turn": stage,
                    "enemies": [{
                        "enemy_id": "proto.enemy.toy_boss",
                        "hp": 100 if stage == 1 else 50,
                        "max_hp": 100,
                        "intent": "attack",
                    }],
                },
            }
            if stage == 3:
                payload["terminal_outcome"] = "defeat"
        serialized = json.dumps(payload, sort_keys=True)
        return Observation(
            policy.policy_id, serialized,
            hashlib.sha256(serialized.encode()).hexdigest(),
        )

    def legal_actions(self, _state: str) -> Sequence[LegalAction]:
        return (A, B)

    def step(self, state: str, action: LegalAction) -> Transition:
        assert action in (A, B)
        return Transition(state, action, self._put(self.states[state] + 1), False)

    def is_terminal(self, state: str) -> bool:
        return self.states[state] == 3

    def release_many(self, states: Sequence[str]) -> int:
        for state in states:
            del self.states[state]
        return len(states)

    def exact_hash(self, _state: str) -> str:
        self.exact_state_calls += 1
        raise AssertionError("Hidden state must never enter diagnostic actor")


def test_stable_temperature_softmax_and_validation() -> None:
    assert temperature_probabilities((1000.0, 1000.0), 1e-8) == (0.5, 0.5)
    assert temperature_probabilities((1000.0, -1000.0), 1e-8) == (1.0, 0.0)
    probabilities = temperature_probabilities((2.0, 1.0), 0.5)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > temperature_probabilities((2.0, 1.0), 1.0)[0]
    for invalid in (-1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            temperature_probabilities((1.0,), invalid)
    with pytest.raises(ValueError, match="finite"):
        temperature_probabilities((float("nan"),), 1.0)
    with pytest.raises(ValueError, match="nonempty"):
        temperature_probabilities((), 1.0)


def test_sampled_agent_records_calibrated_public_action_probabilities() -> None:
    backend = BossToyBackend()
    handle = backend.reset_native_overgrowth("test")
    observation = backend.observe(handle, InformationPolicy("prototype-fair-v0"))
    backend.release_many((handle,))
    agent = NeuralTemperatureAgent(model(), temperature=0.25, rng=random.Random(12))
    choice = agent.choose(observation, (A, B))
    assert choice.action in (A, B)
    stats = json.loads(choice.metadata_json)
    assert stats["chosen_probability"] == 0.5
    assert stats["max_probability"] == 0.5
    assert stats["normalized_entropy"] == pytest.approx(1.0)
    assert stats["legal_action_count"] == 2
    same_seed = NeuralTemperatureAgent(
        model(), temperature=0.25, rng=random.Random(12)
    )
    assert same_seed.choose(observation, (A, B)) == choice
    with pytest.raises(ValueError, match="zero legal"):
        agent.choose(observation, ())


def test_diagnostics_reproduce_and_capture_boss_public_history() -> None:
    backend = BossToyBackend()
    options = (1.0, 0.25, None)
    kwargs = dict(
        seeds=("toy-1", "toy-2"), temperatures=options,
        max_decisions=10, max_boss_actions=1,
    )
    first = diagnose_neural_temperatures(backend, model(), **kwargs)
    second = diagnose_neural_temperatures(backend, model(), **kwargs)
    # Runtime measurements vary; model decisions, trajectories, and all
    # diagnostic content apart from wall-clock values must be deterministic.
    for name in first["temperatures"]:
        first["summary"][name].pop("mean_agent_compute_seconds")
        second["summary"][name].pop("mean_agent_compute_seconds")
    assert first == second
    assert first["temperatures"] == ["t1", "t0.25", "greedy"]
    assert len(first["paired_completed_only"]) == 2
    for name in first["temperatures"]:
        assert first["summary"][name]["completed"] == 2
        assert first["summary"][name]["boss_floor_reaches"] == 2
        assert first["summary"][name]["boss_combat_entries"] == 2
        assert first["summary"][name]["boss_combat_defeats"] == 2
        assert first["summary"][name]["act1_clears"] == 0
        assert first["summary"][name]["observed_policy_decisions"] == 6
        for record in first["runs"][name]:
            assert record["last_observed_hp_before_terminal"] == 12
            assert record["terminal_hp"] == 0
            assert record["boss_combat_decisions"] == 2
            assert record["boss_initial_public_state"]["enemies"][0][
                "enemy_id"
            ] == "proto.enemy.toy_boss"
            assert len(record["boss_first_actions"]) == 1
            assert len(record["boss_last_actions"]) == 1
            assert record["boss_last_actions"][0]["hp"] == 12
    assert first["summary"]["greedy"]["mean_normalized_action_entropy"] == 0.0
    assert first["summary"]["greedy"]["mean_max_action_probability"] == 1.0
    assert backend.exact_state_calls == 0
    assert backend.states == {}


def test_diagnostics_reject_bad_inputs() -> None:
    backend = BossToyBackend()
    for options in ((0.0,), (float("inf"),), (None, None)):
        with pytest.raises(ValueError):
            diagnose_neural_temperatures(
                backend, model(), seeds=("toy",), temperatures=options
            )
    with pytest.raises(ValueError, match="unique"):
        diagnose_neural_temperatures(
            backend, model(), seeds=("toy", "toy")
        )
    assert math.isfinite(temperature_probabilities((0.0,), 1.0)[0])
    assert not backend.states
