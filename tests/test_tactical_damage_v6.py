"""Versioned v6: public base damage + modified damage + intent hit counts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy
from test_tactical_intents_v5 import combat_frame, zeros

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_DAMAGE_FORMAT,
    TACTICAL_RELATIONAL_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.tactical_state import (
    DAMAGE_NUMERIC_COORDINATES,
    DAMAGE_TACTICAL_STATE_SCHEMA,
    damage_tactical_state_features,
    relational_tactical_state_features,
    tactical_state_features,
)
from sts2_ai.training.phase_split_selfplay import train_phase_split


def attacking_enemies() -> dict:
    state = combat_frame()
    first, second = state["combat"]["enemies"]
    first.update({
        "move_id": "attack", "intent_base_damage": 8,
        "intent_damage": 12, "intent_hits": 3,
    })
    second.update({
        "move_id": "attack", "intent_base_damage": 5,
        "intent_damage": 5, "intent_hits": 1,
    })
    return state


def test_v6_continuous_base_and_modified_damages_and_v5_stays_identical() -> None:
    state = attacking_enemies()
    new = damage_tactical_state_features(state, 128)
    assert DAMAGE_NUMERIC_COORDINATES == 40
    assert DAMAGE_TACTICAL_STATE_SCHEMA.endswith("damage")
    assert new[32] == pytest.approx(41 / 100)
    assert new[33] == pytest.approx(36 / 100)
    assert new[34] == pytest.approx(4 / 10)
    assert new[35] == pytest.approx(2 / 5)
    assert new[36] == pytest.approx(29 / 100)
    assert new[37] == pytest.approx(24 / 100)
    assert new[38] == pytest.approx(12 / 100)
    assert new[39] == pytest.approx(2 / 5)

    changed_base = copy.deepcopy(state)
    changed_base["combat"]["enemies"][0]["intent_base_damage"] = 6
    assert damage_tactical_state_features(changed_base, 128) != new
    # v5 still encodes exactly the former schema: no base damage fields.
    assert (
        relational_tactical_state_features(changed_base, 128)
        == relational_tactical_state_features(state, 128)
    )
    assert tactical_state_features(state, 128) == tactical_state_features(
        changed_base, 128
    )


def test_v6_binds_attacks_to_their_owner_and_is_order_invariant() -> None:
    state = attacking_enemies()
    old = damage_tactical_state_features(state, 128)
    reordered = copy.deepcopy(state)
    reordered["combat"]["enemies"].reverse()
    reordered["combat"]["enemies"][0]["instance_id"] = 99
    assert damage_tactical_state_features(reordered, 128) == old

    redistributed = copy.deepcopy(state)
    one, two = redistributed["combat"]["enemies"]
    for key in ("intent_base_damage", "intent_damage", "intent_hits"):
        one[key], two[key] = two[key], one[key]
    # Aggregate numeric totals agree, but enemy-bound tokens do not.
    new = damage_tactical_state_features(redistributed, 128)
    assert all(old.get(i) == new.get(i) for i in range(40))
    assert old != new


def test_hidden_or_incomplete_damage_does_not_leak_into_v6_features() -> None:
    state = attacking_enemies()
    state["combat"]["enemies"][0]["move_id"] = None
    baseline = damage_tactical_state_features(state, 128)
    for base in [0, 99999]:
        state["combat"]["enemies"][0]["intent_base_damage"] = base
        assert damage_tactical_state_features(state, 128) == baseline

    enemy = state["combat"]["enemies"][1]
    del enemy["intent_hits"]
    incomplete = damage_tactical_state_features(state, 128)
    assert incomplete.get(32, 0) == 0
    assert incomplete.get(36, 0) == 0
    enemy["intent_hits"] = True
    assert damage_tactical_state_features(state, 128) == incomplete
    enemy["intent_hits"] = 0
    assert damage_tactical_state_features(state, 128) == incomplete
    enemy["intent_hits"] = 2
    enemy["intent_damage"] = float("nan")
    assert damage_tactical_state_features(state, 128) == incomplete

    with pytest.raises(ValueError, match="40"):
        damage_tactical_state_features(state, 40)
    with pytest.raises(ValueError, match="require relational"):
        tactical_state_features(state, 128, damage_aware=True)


def test_v6_untargeted_action_logits_depend_on_public_base_damage() -> None:
    first = attacking_enemies()
    second = copy.deepcopy(first)
    second["combat"]["enemies"][0]["intent_base_damage"] = 5
    model = zeros(TACTICAL_DAMAGE_FORMAT)
    model.state_weight[0][36] = 1
    model.policy_weight[0] = 1
    action = LegalAction("defend-100", "play_card", '{"CardInstanceId":100}')
    a = Observation("prototype-fair-v0", json.dumps(first), "")
    b = Observation("prototype-fair-v0", json.dumps(second), "")
    assert model.evaluate(a, (action,)).action_logits != model.evaluate(
        b, (action,)
    ).action_logits
    # Portable inference works independently from the training library.
    assert NeuralPolicyValueModel.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_v6_warm_start_and_checkpoint_do_not_reuse_v5_state_coordinates(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    prior = PhaseSplitNeuralModel(
        zeros(NEURAL_FORMAT), zeros(TACTICAL_RELATIONAL_FORMAT), "v5",
        combat_value_objective="hp_preservation",
    )
    prior.combat.state_weight[0][1] = 999
    prior.combat.policy_weight[0] = 0.25
    warm = tmp_path / "warm.json"
    prior.save(warm)
    ckpt = tmp_path / "v6.pt"
    backend = TwoPhaseToy()
    model, rows = train_phase_split(
        backend, rounds=1, episodes_per_round=4,
        dimension=128, hidden=4, seed=7, max_decisions=8,
        tactical_state_encoding="relational_damage",
        combat_objective="hp_preservation",
        combat_advantage_baseline="leave_one_run_out",
        warm_start=warm, checkpoint=ckpt,
        temperature_start=1, temperature_end=1,
    )
    assert model.combat.format_id == TACTICAL_DAMAGE_FORMAT
    assert model.combat.state_weight[0][1] < 999
    assert model.combat.policy_weight[0] != 0
    assert rows[0].optimization_steps == 2
    assert PhaseSplitNeuralModel.from_dict(model.to_dict()).to_dict() == model.to_dict()

    _, resumed = train_phase_split(
        backend, rounds=2, episodes_per_round=4,
        dimension=128, hidden=4, seed=7, max_decisions=8,
        tactical_state_encoding="relational_damage",
        combat_objective="hp_preservation",
        combat_advantage_baseline="leave_one_run_out",
        warm_start=warm, checkpoint=ckpt, resume=True,
        temperature_start=1, temperature_end=1,
    )
    assert resumed[0] == rows[0]
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            backend, rounds=2, episodes_per_round=4,
            dimension=128, hidden=4, seed=7, max_decisions=8,
            tactical_state_encoding="relational",
            combat_objective="hp_preservation",
            combat_advantage_baseline="leave_one_run_out",
            warm_start=warm, checkpoint=ckpt, resume=True,
            temperature_start=1, temperature_end=1,
        )
