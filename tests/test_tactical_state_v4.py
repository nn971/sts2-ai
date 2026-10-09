"""Regression tests for end-to-end structured tactical policy rollout training."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy

from sts2_ai.emulator import Observation
from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_FORMAT,
    TACTICAL_STRUCTURED_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.tactical_state import (
    NUMERIC_COORDINATES,
    TACTICAL_STATE_SCHEMA,
    tactical_state_features,
)
from sts2_ai.training.phase_split_selfplay import train_phase_split


def frame(hp: int = 40) -> dict:
    return {
        "hp": hp, "max_hp": 70, "act": 1, "floor": 3, "gold": 80,
        "deck": [{"card_id": "proto.silent.strike", "upgrade_level": 0}],
        "relics": [{"relic_id": "proto.silent_ring"}],
        "potions": [{"slot": 2, "potion_id": "proto.potion.fire"}],
        "combat": {
            "turn": 2, "energy": 2, "player_block": 3,
            "draw_pile_count": 4,
            "hand": [{"card_id": "proto.silent.strike", "instance_id": 18}],
            "discard_pile": [],
            "exhaust_pile": [],
            "enemies": [
                {"enemy_id": "slime", "instance_id": 1, "hp": 11, "block": 0,
                 "move_id": "attack", "powers": []},
            ],
            "player_powers": [],
        },
        "map": [{"node_id": str(i)} for i in range(100)],
    }


def zeros(fmt: str, *, dimension: int = 64) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": fmt, "dimension": dimension, "hidden": 4,
        "model_id": fmt,
        "state_weight": [[0.0] * dimension for _ in range(4)],
        "state_bias": [0.0] * 4,
        "action_weight": [[0.0] * dimension for _ in range(4)],
        "action_bias": [0.0] * 4,
        "policy_weight": [0.0] * 4, "policy_bias": 0.0,
        "value_weight": [0.0] * 4, "value_bias": 0.0,
        "value_head_trained": True,
    })


def test_tactical_state_encoder_preserves_continuous_hp_and_current_hand() -> None:
    state = frame()
    baseline = tactical_state_features(state, 128)
    assert TACTICAL_STATE_SCHEMA == "sts2-tactical-public-state-v4"
    assert NUMERIC_COORDINATES == 32
    assert baseline[1] == pytest.approx(40 / 70)
    less_hp = tactical_state_features(frame(hp=25), 128)
    assert less_hp[1] == pytest.approx(25 / 70)
    assert less_hp != baseline
    irrelevant = frame()
    irrelevant["map"] = [{"node_id": "completely-different"}]
    irrelevant["combat"]["enemies"][0]["instance_id"] = 987654
    irrelevant["combat"]["hand"][0]["instance_id"] = 444
    irrelevant["potions"][0]["slot"] = 9
    assert tactical_state_features(irrelevant, 128) == baseline
    potion = frame()
    potion["potions"][0]["potion_id"] = "proto.potion.different"
    assert tactical_state_features(potion, 128) != baseline
    new_hand = frame()
    new_hand["combat"]["hand"][0]["card_id"] = "proto.silent.defend"
    assert tactical_state_features(new_hand, 128) != baseline
    other_intent = frame()
    other_intent["combat"]["enemies"][0]["move_id"] = "block"
    assert tactical_state_features(other_intent, 128) != baseline
    with pytest.raises(ValueError, match="dimension"):
        tactical_state_features(frame(), 32)
    with pytest.raises(ValueError, match="combat"):
        tactical_state_features({"combat": None}, 128)


def test_v4_inference_uses_reserved_hp_coordinate_and_roundtrips() -> None:
    tactical = zeros(TACTICAL_STRUCTURED_FORMAT)
    tactical.state_weight[0][1] = 1.0
    tactical.value_weight[0] = 1.0
    low = Observation("prototype-fair-v0", json.dumps(frame(20)), "")
    high = Observation("prototype-fair-v0", json.dumps(frame(50)), "")
    assert tactical.evaluate(high, ()).value > tactical.evaluate(low, ()).value
    legacy = zeros(TACTICAL_FORMAT)
    strategy = zeros(NEURAL_FORMAT)
    both = PhaseSplitNeuralModel(strategy, tactical, "structured-test")
    assert PhaseSplitNeuralModel.from_dict(both.to_dict()).to_dict() == both.to_dict()
    assert legacy.format_id == TACTICAL_FORMAT
    assert both.combat.format_id == TACTICAL_STRUCTURED_FORMAT


def test_structured_trainer_warm_start_preserves_strategy_and_action_weights(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    old = PhaseSplitNeuralModel(
        zeros(NEURAL_FORMAT), zeros(TACTICAL_FORMAT), "legacy-warm"
    )
    old.strategy.state_bias[:] = [1.0, 0.0, 0.0, 0.0]
    old.combat.state_weight[0][1] = 999.0
    old.combat.action_bias[:] = [0.3, 0.2, 0.1, 0.0]
    warm = tmp_path / "warm.json"
    old.save(warm)
    backend = TwoPhaseToy()
    model, rows = train_phase_split(
        backend, rounds=1, episodes_per_round=3,
        dimension=64, hidden=4, max_decisions=8,
        seed=21, workers=1, warm_start=warm,
        tactical_state_encoding="structured",
        temperature_start=1.0, temperature_end=1.0,
        checkpoint=tmp_path / "v4-checkpoint.pt",
    )
    assert rows[0].optimization_steps == 2
    assert model.combat.format_id == TACTICAL_STRUCTURED_FORMAT
    assert model.strategy.format_id == NEURAL_FORMAT
    assert model.combat.state_weight[0][1] < 999.0
    assert not backend.states
    resumed, continuation = train_phase_split(
        backend, rounds=2, episodes_per_round=3,
        dimension=64, hidden=4, max_decisions=8, seed=21, workers=1,
        warm_start=warm, tactical_state_encoding="structured",
        temperature_start=1.0, temperature_end=1.0,
        checkpoint=tmp_path / "v4-checkpoint.pt", resume=True,
    )
    assert len(continuation) == 2
    assert resumed.combat.format_id == TACTICAL_STRUCTURED_FORMAT
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            backend, rounds=2, episodes_per_round=3,
            dimension=64, hidden=4, max_decisions=8, seed=21, workers=1,
            warm_start=warm, tactical_state_encoding="legacy",
            temperature_start=1.0, temperature_end=1.0,
            checkpoint=tmp_path / "v4-checkpoint.pt", resume=True,
        )
