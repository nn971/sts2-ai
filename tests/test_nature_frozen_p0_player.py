"""Frozen portable v15 fair-player adapter and checkpoint provenance regressions."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from sts2_ai.emulator.semantic_p0_frozen_player import (
    FrozenP0Player,
    FrozenPlayerSpec,
    FrozenPolicyInferenceError,
)
from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel


@dataclass(frozen=True)
class FakeEstimate:
    action_logits: tuple[float, ...]


class FakePolicy:
    model_id = "test-frozen-v15"

    def __init__(self, logits=(0.0, 2.0, 1.0)):
        self.logits = logits
        self.last_observation = None

    def evaluate(self, observation, legal_actions):
        self.last_observation = observation
        assert observation.policy_id == "prototype-fair-v0"
        assert "semanticDeck" not in observation.payload_json
        assert len(legal_actions) == len(self.logits)
        return FakeEstimate(self.logits)


ACTIONS = [
    {"actionId": "a0", "kind": "play_card", "payloadJson": '{"CardInstanceId":1}'},
    {"actionId": "a1", "kind": "play_card", "payloadJson": '{"CardInstanceId":2}'},
    {"actionId": "end", "kind": "end_turn", "payloadJson": "{}"},
]


def small_portable_model(path: Path) -> PhaseSplitNeuralModel:
    def head(fmt, model_id):
        return NeuralPolicyValueModel.from_dict({
            "format": fmt,
            "model_id": model_id,
            "dimension": 32, "hidden": 8,
            "state_weight": [[0.0] * 32 for _ in range(8)],
            "state_bias": [0.0] * 8,
            "action_weight": [[0.0] * 32 for _ in range(8)],
            "action_bias": [0.0] * 8,
            "policy_weight": [0.0] * 8,
            "policy_bias": 0.0,
            "value_weight": [0.0] * 8,
            "value_bias": 0.0,
        })
    model = PhaseSplitNeuralModel(
        strategy=head(NEURAL_FORMAT, "test-untuned-strategy"),
        combat=head(TACTICAL_FORMAT, "test-untuned-combat"),
        model_id="test-only-zero-weight-v15-shaped",
    )
    model.save(path)
    return model


def test_greedy_player_uses_public_logits_and_original_action_ids():
    model = FakePolicy()
    actor = FrozenP0Player(model, actor_seed=29)
    assert actor.choose({"combat": {"hand": []}}, ACTIONS) == "a1"
    assert actor.decisions == 1
    assert model.last_observation.observation_hash


def test_public_hidden_leak_duplicate_actions_invalid_logits():
    for observation in (
        {"semanticDeck": {"unseen": ["private"]}},
        {"pending": {"intent": "secret"}},
    ):
        with pytest.raises(FrozenPolicyInferenceError, match="Hidden"):
            FrozenP0Player(FakePolicy(), actor_seed=0).choose(observation, ACTIONS)
    with pytest.raises(FrozenPolicyInferenceError, match="Duplicate"):
        FrozenP0Player(FakePolicy(), actor_seed=0).choose(
            {"combat": {}}, [ACTIONS[0], ACTIONS[0]],
        )
    with pytest.raises(FrozenPolicyInferenceError, match="invalid logits"):
        FrozenP0Player(FakePolicy((1.0, math.nan, 0.0)), actor_seed=0).choose(
            {"combat": {}}, ACTIONS,
        )


def test_seeded_sampling_replay_and_greedy_consistency():
    sample1 = FrozenP0Player(FakePolicy((0, 0, 0)), actor_seed=783, greedy=False)
    sample2 = FrozenP0Player(FakePolicy((0, 0, 0)), actor_seed=783, greedy=False)
    choices1 = [sample1.choose({"combat": {}}, ACTIONS) for _ in range(40)]
    choices2 = [sample2.choose({"combat": {}}, ACTIONS) for _ in range(40)]
    assert choices1 == choices2 and len(set(choices1)) > 1
    assert FrozenP0Player(FakePolicy(), actor_seed=0, greedy=True).choose(
        {"combat": {}}, ACTIONS,
    ) == "a1"


def test_portable_phase_split_checkpoint_digest_and_seed_provenance(tmp_path):
    path = tmp_path / "portable.json"
    small_portable_model(path)
    spec = FrozenPlayerSpec.from_file(path, seed=28, temperature=0.9)
    assert spec.load_pinned().model_id == "test-only-zero-weight-v15-shaped"
    assert spec.episode_seed(fixture_seed="run-1", fixture_id="enc-1") == (
        spec.episode_seed(fixture_seed="run-1", fixture_id="enc-1")
    )
    assert spec.episode_seed(fixture_seed="run-1", fixture_id="enc-1") != (
        spec.episode_seed(fixture_seed="run-2", fixture_id="enc-1")
    )
    assert spec.model_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    actor = FrozenP0Player(spec.load_pinned(), actor_seed=0)
    assert actor.choose({"combat": {"hand": []}}, ACTIONS) in {
        "a0", "a1", "end",
    }
    path.write_text(path.read_text() + "\n")
    with pytest.raises(FrozenPolicyInferenceError, match="digest mismatch"):
        spec.load_pinned()
