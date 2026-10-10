"""Player-visible enemy/announced-move binding for tactical v5 policies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_RELATIONAL_FORMAT,
    TACTICAL_STRUCTURED_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.tactical_state import (
    RELATIONAL_NUMERIC_COORDINATES,
    RELATIONAL_TACTICAL_STATE_SCHEMA,
    relational_tactical_state_features,
    tactical_state_features,
)
from sts2_ai.training.phase_split_selfplay import train_phase_split


def combat_frame() -> dict:
    return {
        "hp": 50, "max_hp": 70, "floor": 5, "act": 1,
        "deck": [{"card_id": "silent.defend", "instance_id": 100}],
        "potions": [], "relics": [],
        "combat": {
            "turn": 1, "energy": 3, "player_block": 0,
            "draw_pile_count": 4,
            "hand": [{"card_id": "silent.defend", "instance_id": 100}],
            "discard_pile": [], "exhaust_pile": [], "player_powers": [],
            "enemies": [
                {
                    "enemy_id": "mawler", "instance_id": 1,
                    "move_id": "attack", "hp": 40, "block": 0,
                    "statuses": {}, "powers": [],
                },
                {
                    "enemy_id": "slime", "instance_id": 2,
                    "move_id": "defend", "hp": 40, "block": 0,
                    "statuses": {}, "powers": [],
                },
            ],
        },
    }


def swapped_intents(state: dict) -> dict:
    other = copy.deepcopy(state)
    a, b = other["combat"]["enemies"]
    a["move_id"], b["move_id"] = b["move_id"], a["move_id"]
    return other


def zeros(fmt: str, dim: int = 128) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": fmt, "dimension": dim, "hidden": 4,
        "model_id": "intent-v5-test",
        "state_weight": [[0.0] * dim for _ in range(4)],
        "state_bias": [0.0] * 4,
        "action_weight": [[0.0] * dim for _ in range(4)],
        "action_bias": [0.0] * 4,
        "policy_weight": [0.0] * 4, "policy_bias": 0.0,
        "value_weight": [0.0] * 4, "value_bias": 0.0,
    })


def test_same_enemies_swapped_intents_collide_v4_but_not_v5() -> None:
    state = combat_frame()
    swapped = swapped_intents(state)
    # v4 keeps only unbound identity/move counts.
    assert tactical_state_features(state, 128) == tactical_state_features(swapped, 128)
    before = relational_tactical_state_features(state, 128)
    after = relational_tactical_state_features(swapped, 128)
    assert before != after
    assert RELATIONAL_TACTICAL_STATE_SCHEMA.endswith("relational")
    assert RELATIONAL_NUMERIC_COORDINATES == 36
    # No dependence on ephemeral enemy instance IDs or enemy list order.
    reordered = copy.deepcopy(state)
    reordered["combat"]["enemies"].reverse()
    reordered["combat"]["enemies"][0]["instance_id"] = 999
    reordered["combat"]["enemies"][1]["instance_id"] = 888
    assert relational_tactical_state_features(reordered, 128) == before


def test_untargeted_defend_logits_can_distinguish_swapped_enemy_intents() -> None:
    before, after = combat_frame(), swapped_intents(combat_frame())
    f0, f1 = (
        relational_tactical_state_features(before, 128),
        relational_tactical_state_features(after, 128),
    )
    changed_indices = sorted(
        i for i in set(f0) | set(f1) if f0.get(i, 0) != f1.get(i, 0)
    )
    assert changed_indices
    feature_index = changed_indices[0]
    model = zeros(TACTICAL_RELATIONAL_FORMAT)
    model.state_weight[0][feature_index] = 1
    model.policy_weight[0] = 1
    # The same untargeted Defend action (no target-specific enemy features).
    action = LegalAction(
        "defend-100", "play_card", '{"CardInstanceId":100}'
    )
    one = model.evaluate(
        Observation("prototype-fair-v0", json.dumps(before), ""),
        (action,),
    )
    two = model.evaluate(
        Observation("prototype-fair-v0", json.dumps(after), ""),
        (action,),
    )
    assert one.action_logits != two.action_logits
    # Existing v4 checkpoints still load their unmodified format.
    assert zeros(TACTICAL_STRUCTURED_FORMAT).format_id == (
        TACTICAL_STRUCTURED_FORMAT
    )


def test_unknown_intent_stays_unknown_and_public_damage_requires_visible_move() -> None:
    state = combat_frame()
    enemies = state["combat"]["enemies"]
    enemies[0]["move_id"] = None
    no_attack = relational_tactical_state_features(state, 128)
    enemies[0]["intent_damage"] = 999
    enemies[0]["intent_hits"] = 3
    assert relational_tactical_state_features(state, 128) == no_attack
    enemies[0]["move_id"] = "attack"
    enemies[0]["intent_damage"] = 8
    enemies[0]["intent_hits"] = 3
    displayed = relational_tactical_state_features(state, 128)
    assert displayed[32] == pytest.approx(24 / 100)
    assert displayed[33] == pytest.approx(24 / 100)
    assert displayed[34] == pytest.approx(3 / 10)
    assert displayed[35] == pytest.approx(1 / 5)
    assert displayed != no_attack
    # Exclude impossible/invalid numeric intent values.
    enemies[0]["intent_damage"] = float("nan")
    assert 32 not in relational_tactical_state_features(state, 128)


def test_v5_training_warm_start_and_checkpoint_isolation(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    prior = PhaseSplitNeuralModel(
        zeros(NEURAL_FORMAT), zeros(TACTICAL_STRUCTURED_FORMAT), "v4",
    )
    prior.combat.state_weight[0][1] = 999
    warm = tmp_path / "old.json"
    prior.save(warm)
    ckpt = tmp_path / "v5.pt"
    backend = TwoPhaseToy()
    model, rounds = train_phase_split(
        backend, rounds=1, episodes_per_round=4,
        dimension=128, hidden=4, seed=3, max_decisions=8,
        tactical_state_encoding="relational",
        warm_start=warm, checkpoint=ckpt,
        temperature_start=1, temperature_end=1,
    )
    assert model.combat.format_id == TACTICAL_RELATIONAL_FORMAT
    assert model.combat.state_weight[0][1] < 999
    assert rounds[0].optimization_steps == 2
    assert not backend.states
    recovered, next_rounds = train_phase_split(
        backend, rounds=2, episodes_per_round=4,
        dimension=128, hidden=4, seed=3, max_decisions=8,
        tactical_state_encoding="relational", warm_start=warm,
        checkpoint=ckpt, resume=True, temperature_start=1,
        temperature_end=1,
    )
    assert recovered.combat.format_id == TACTICAL_RELATIONAL_FORMAT
    assert next_rounds[0] == rounds[0]
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            backend, rounds=2, episodes_per_round=4,
            dimension=128, hidden=4, seed=3, max_decisions=8,
            tactical_state_encoding="structured", warm_start=warm,
            checkpoint=ckpt, resume=True, temperature_start=1,
            temperature_end=1,
        )
