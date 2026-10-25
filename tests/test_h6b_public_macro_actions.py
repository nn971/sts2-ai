"""H6-B public macro features: path context and inference/PPO parity."""
import json
from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy, small_model

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.hashed_linear import neural_action_features
from sts2_ai.models.macro_actions import macro_action_features
from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    NEURAL_MACRO_FORMAT,
    TACTICAL_FORMAT,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.phase_split_ppo_batch import encode_decision
from sts2_ai.training.phase_split_selfplay import train_phase_split
from sts2_ai.training.selfplay import PublicDecision


def _map(shift: str = "") -> dict:
    return {
        "hp": 46, "max_hp": 70, "gold": 120, "deck": [],
        "combat": None,
        "map": [
            {"node_id": "1:4:1" + shift, "room_type": 0, "floor": 4,
             "next_node_ids": ["1:5:1" + shift]},
            {"node_id": "1:4:5" + shift, "room_type": 0, "floor": 4,
             "next_node_ids": ["1:5:5" + shift]},
            {"node_id": "1:5:1" + shift, "room_type": 4, "floor": 5,
             "next_node_ids": ["1:6:1" + shift]},
            {"node_id": "1:5:5" + shift, "room_type": 1, "floor": 5,
             "next_node_ids": ["1:6:5" + shift]},
            {"node_id": "1:6:1" + shift, "room_type": 3, "floor": 6,
             "next_node_ids": []},
            {"node_id": "1:6:5" + shift, "room_type": 0, "floor": 6,
             "next_node_ids": []},
        ],
    }


def _action(col: int, suffix: str = "") -> LegalAction:
    return LegalAction(
        f"candidate-{col}", "choose_map_node",
        json.dumps({"NodeId": f"1:4:{col}{suffix}"}),
    )


def test_same_room_type_routes_have_different_future_public_features():
    state = _map()
    a, b = _action(1), _action(5)
    assert neural_action_features(
        state, a.kind, a.payload_json, 512,
    ) == neural_action_features(state, b.kind, b.payload_json, 512)
    assert macro_action_features(
        state, a.kind, a.payload_json, 512,
    ) != macro_action_features(state, b.kind, b.payload_json, 512)


def test_renaming_private_node_keys_does_not_change_macro_features():
    before, after = _map(), _map("-renamed")
    for col in (1, 5):
        assert macro_action_features(
            before, "choose_map_node", _action(col).payload_json, 512,
        ) == macro_action_features(
            after, "choose_map_node", _action(col, "-renamed").payload_json, 512,
        )


def test_reward_duplicate_and_novel_card_uses_public_deck_counts():
    state = {
        "deck": [{"instance_id": 1, "card_id": "silent.strike"},
                 {"instance_id": 2, "card_id": "silent.strike"}],
        "reward": {"card_options": ["silent.strike", "silent.poison_stab"]},
    }
    existing = macro_action_features(
        state, "take_reward_card", '{"Index":0}', 512,
    )
    novel = macro_action_features(
        state, "take_reward_card", '{"Index":1}', 512,
    )
    assert existing != novel
    assert macro_action_features(
        {"deck": []}, "take_reward_card", '{"Index":0}', 512,
    ) == macro_action_features(
        {"deck": []}, "take_reward_card", '{"Index":0}', 512,
    )


def test_portable_model_and_batched_ppo_use_identical_macro_inputs():
    torch = pytest.importorskip("torch")
    state = _map()
    actions = (_action(1), _action(5))
    observation = Observation(
        "prototype-fair-v0", json.dumps(state), "source-hash",
    )
    model = small_model(NEURAL_MACRO_FORMAT)
    model.dimension = 512
    model.state_weight = [[0.0] * 512 for _ in range(4)]
    model.action_weight = [[0.0] * 512 for _ in range(4)]
    model.policy_weight = [1.0, 0.0, 0.0, 0.0]
    model.action_bias = [0.0] * 4
    aa = macro_action_features(state, actions[0].kind, actions[0].payload_json, 512)
    bb = macro_action_features(state, actions[1].kind, actions[1].payload_json, 512)
    chosen = next(i for i in set(aa) | set(bb)
                  if abs(aa.get(i, 0) - bb.get(i, 0)) > 1e-8)
    model.action_weight[0][chosen] = 4.0 if aa.get(chosen, 0) >= bb.get(chosen, 0) else -4.0
    estimate = model.evaluate(observation, actions)
    item = encode_decision(
        PublicDecision(observation, actions, 0, "strategy"),
        512, "enemy_instances", torch, strategy_action_encoding="macro_v1",
    )
    assert item.actions[0].tolist() == pytest.approx(
        [aa.get(i, 0.0) for i in range(512)]
    )
    assert item.actions[1].tolist() == pytest.approx(
        [bb.get(i, 0.0) for i in range(512)]
    )
    assert estimate.action_logits[0] != estimate.action_logits[1]


def test_frozen_macro_ppo_exports_new_strategy_format_only(tmp_path: Path):
    pytest.importorskip("torch")
    backbone = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT),
        small_model(TACTICAL_FORMAT, value_bias=0.333333),
        "h6b-checkpoint",
    )
    path = tmp_path / "h6b-warm.json"
    backbone.save(path)
    model, rows = train_phase_split(
        TwoPhaseToy(), rounds=1, episodes_per_round=12,
        dimension=32, hidden=4, max_decisions=8, workers=1,
        optimizer_method="ppo", ppo_backend="batched",
        hp_monotonic_weight=0, warm_start=path, freeze_combat=True,
        strategy_action_encoding="macro_v1",
        temperature_start=1.0, temperature_end=1.0,
    )
    assert rows[0].strategic_decisions == 12
    assert rows[0].tactical_decisions == 0
    assert model.strategy.format_id == NEURAL_MACRO_FORMAT
    assert model.combat.to_dict() == backbone.combat.to_dict()


def test_h6b_cannot_silently_run_in_unimplemented_optimizer():
    with pytest.raises(ValueError, match="batched PPO"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=1,
            dimension=32, hidden=4, strategy_action_encoding="macro_v1",
        )
