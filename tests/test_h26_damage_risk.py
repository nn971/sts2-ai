"""H26-D source gross-damage probabilities, public modifiers and parity."""
from __future__ import annotations

import copy
from test_h26_intent_law import _catalog
from test_h26_searchfree import _combat, _models

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet
from sts2_ai.training.h26_damage_risk import conditional_next_gross_damage
from sts2_ai.training.h26_public_intent_law import (
    DAMAGE_SCHEMA,
    validate_public_intent_catalog,
)
from sts2_ai.training.h26_tactical import attach_visible_threat
from tools.train_h26_searchfree_a0 import _h26_ai_source_sha256


def _power(pid, *, enemy_bonus=0, enemy_out=(1, 1),
           enemy_powered=(1, 1), incoming=(1, 1), cap=0):
    return {
        "power_id": pid,
        "enemy_attack_damage_bonus_per_stack": enemy_bonus,
        "enemy_outgoing_damage_numerator": enemy_out[0],
        "enemy_outgoing_damage_denominator": enemy_out[1],
        "enemy_outgoing_powered_attack_numerator": enemy_powered[0],
        "enemy_outgoing_powered_attack_denominator": enemy_powered[1],
        "player_incoming_attack_damage_numerator": incoming[0],
        "player_incoming_attack_damage_denominator": incoming[1],
        "player_incoming_damage_cap": cap,
        "lethal_after_unblocked_powered_attack": False,
        "player_max_hp_loss_on_unblocked_attack_hit_per_stack": 0,
    }


def _damage_effect(*, base=9, reps=2, nonuniform=False, uses=0):
    return {
        "amount": base, "amount_per_act": 0, "repetitions": reps,
        "ascension_deltas": [{"min_ascension": 9, "delta": 2}],
        "repetition_ascension_deltas": [],
        "is_attack": True, "use_stored_enemy_damage": False,
        "extra_amount_per_prior_move_use": uses,
        "extra_amount_from_owner_power_id": None,
        "owner_power_stack_condition_id": None,
        "owner_power_stacks_less_than": None,
        "owner_power_stacks_at_least": None,
        "earlier_non_damage_effect": nonuniform,
    }


def _catalog_v3():
    raw = copy.deepcopy(_catalog())
    raw["schema"] = DAMAGE_SCHEMA
    raw["damage_powers"] = [
        _power("strength", enemy_bonus=2),
        _power("weak", enemy_out=(3, 4)),
        _power("vulnerable", incoming=(3, 2)),
        _power("cap", cap=1),
    ]
    raw["damage_relics"] = []
    raw["enemy_status_power_ids"] = {"proto.status.weak": "weak"}
    raw["tainted_power_id"] = "tainted"
    raw["surrounded_power_id"] = "surrounded"
    for enemy in raw["enemies"]:
        enemy["move_damage"] = [
            {"move_id": mid, "effects": (
                [_damage_effect()] if
                enemy["enemy_id"] == "proto.enemy.random" and mid == "stab"
                else []
            )}
            for mid in enemy["move_ids"]
        ]
    return validate_public_intent_catalog(raw)


def _state():
    state = _combat(hp=20)
    state["act"] = 1
    state["combat"]["enemies"][0]["enemy_id"] = "proto.enemy.random"
    state["combat"]["enemies"][0]["move_id"] = "opener"
    state["combat"]["enemies"][0]["powers"] = []
    state["combat"]["player_powers"] = []
    state["relics"] = []
    return state


def _risk(state, catalog, *, ascension=0, history=()):
    enemy = state["combat"]["enemies"][0]
    return conditional_next_gross_damage(
        state, enemy, catalog, ascension=ascension,
        executed_moves=history,
    )


def test_exact_mixed_buff_attack_gross_damage_variance_and_hp_threshold():
    catalog = _catalog_v3()
    state = _state()
    result = _risk(state, catalog)
    assert result.available
    assert {x.move_id: x.gross_damage for x in result.outcomes} == {
        "stab": 18, "guard": 0,
    }
    assert sum(x.numerator for x in result.outcomes) == result.denominator
    assert result.expected_gross_damage == 9
    assert result.variance_gross_damage == 81
    assert result.probability_gross_at_least_hp == 0
    # Current Block is never subtracted from a FUTURE turn's gross damage.
    state["combat"]["player_block"] = 50
    assert _risk(state, catalog).expected_gross_damage == 9
    # Public post-action HP changes recompute the conditional threshold.
    state["hp"] = 15
    assert _risk(state, catalog).probability_gross_at_least_hp == 0.5


def test_public_strength_weak_incoming_vulnerable_and_damage_cap():
    catalog = _catalog_v3()
    state = _state()
    enemy = state["combat"]["enemies"][0]
    enemy["powers"] = [{"power_id": "strength", "stacks": 2}]
    assert _risk(state, catalog).expected_gross_damage == 13
    enemy["statuses"] = {"proto.status.weak": 1}
    # floor((9+4)*3/4) = 9, times 2 = 18 (weighted expectation 9)
    assert _risk(state, catalog).expected_gross_damage == 9
    state["combat"]["player_powers"] = [
        {"power_id": "vulnerable", "stacks": 1},
    ]
    # floor(floor(13*3/4)*3/2) = 13, two hits.
    assert _risk(state, catalog).expected_gross_damage == 13
    state["combat"]["player_powers"].append(
        {"power_id": "cap", "stacks": 1},
    )
    assert _risk(state, catalog).expected_gross_damage == 1
    assert _risk(state, catalog, ascension=9).expected_gross_damage == 1


def test_conditional_damage_rejects_dynamic_hidden_effects():
    catalog = _catalog_v3()
    state = _state()
    rule = next(r for r in catalog["enemies"]
                if r["enemy_id"] == "proto.enemy.random")
    stab = rule["move_damage"][1]["effects"][0]
    stab["earlier_non_damage_effect"] = True
    assert _risk(state, catalog).reason == "dynamic_or_unobserved_damage_modifier"
    stab["earlier_non_damage_effect"] = False
    stab["use_stored_enemy_damage"] = True
    assert not _risk(state, catalog).available
    stab["use_stored_enemy_damage"] = False
    stab["extra_amount_per_prior_move_use"] = 3
    assert not _risk(state, catalog, history=None).available
    assert _risk(state, catalog, history=("stab",)).expected_gross_damage == 12


def test_tactical_damage_tokens_and_checkpoint_roundtrip(tmp_path):
    catalog = _catalog_v3()
    macro, combat = _models()
    state = _state()
    actions = (LegalAction("wait", "end_turn", "{}"),)
    original = macro.encode_decision(state, actions)
    enhanced = attach_visible_threat(
        original, state, intent_catalog=catalog,
        intent_histories={2: ()}, damage_ascension=0,
    )
    assert len(enhanced.objects) == len(original.objects) + 4
    assert enhanced.actions == original.actions
    assert enhanced.neutral_value == original.neutral_value
    assert len(enhanced.relations) == len(original.relations) + 3
    for obj in enhanced.objects[-3:]:
        assert obj.owner is not None
    combat.h26_intent_catalog = catalog
    combat.h26_damage_ascension = 0
    assert combat.encode_decision(
        state, actions, intent_histories={2: ()},
    ) == enhanced
    file = tmp_path / "risk.pt"
    combat.save(file)
    restored = JointActPilotNet.load(file)
    assert restored.h26_damage_ascension == 0
    assert restored.encode_decision(
        state, actions, intent_histories={2: ()},
    ) == enhanced


def test_source_hash_is_deterministic_and_digest_shaped():
    value = _h26_ai_source_sha256()
    assert len(value) == 64
    assert value == _h26_ai_source_sha256()
