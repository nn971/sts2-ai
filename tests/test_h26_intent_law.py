"""Public-only exact conditional enemy intent law and token contract."""
from __future__ import annotations

import copy

import pytest
from test_h26_searchfree import _combat, _models

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS, ZONES
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet
from sts2_ai.training.h26_public_intent_law import (
    SCHEMA,
    next_intent_law,
    validate_public_intent_catalog,
)
from sts2_ai.training.h26_tactical import attach_visible_threat


def _catalog():
    weights = {
        str(n): [8 // n + (1 if i < (8 % n) else 0) for i in range(n)]
        for n in range(1, 4)
    }
    return validate_public_intent_catalog({
        "schema": SCHEMA,
        "probability_denominator": 8,
        "probability_provenance":
            "source-native-nextint-grid-with-public-conditioning-gate",
        "native_next_int_weights": weights,
        "enemies": [
            {
                "enemy_id": "proto.enemy.loop",
                "move_policy": "SequentialLoop",
                "move_ids": ["a", "b", "c"],
                "max_consecutive_uses": [0, 0, 0],
                "opening_move_indices": [0],
                "random_move_pool_start_index": 0,
                "move_loop_start_index": 1,
            },
            {
                "enemy_id": "proto.enemy.random",
                "move_policy": "UniformRandomAfterOpener",
                "move_ids": ["opener", "stab", "guard"],
                "max_consecutive_uses": [0, 1, 0],
                "opening_move_indices": [0],
                "random_move_pool_start_index": 1,
                "move_loop_start_index": 0,
            },
            {
                "enemy_id": "proto.enemy.ambiguous",
                "move_policy": "UniformRandomAfterOpener",
                "move_ids": ["slash", "stab", "guard"],
                "max_consecutive_uses": [0, 2, 0],
                "opening_move_indices": [1],
                "random_move_pool_start_index": 1,
                "move_loop_start_index": 0,
            },
        ],
    })


def _enemy(eid, mid):
    return {"enemy_id": eid, "move_id": mid, "hp": 20}


def test_source_grid_mass_and_visible_only_uniform_eligibility():
    catalog = _catalog()
    first = next_intent_law(_enemy("proto.enemy.random", "opener"), catalog)
    assert first.available
    assert sum(w for _, w in first.outcomes) == 8
    assert dict(first.outcomes) == {"guard": 4, "stab": 4}
    stab = next_intent_law(_enemy("proto.enemy.random", "stab"), catalog)
    assert dict(stab.outcomes) == {"guard": 8}
    guard = next_intent_law(_enemy("proto.enemy.random", "guard"), catalog)
    assert dict(guard.outcomes) == {"guard": 4, "stab": 4}
    loop = next_intent_law(_enemy("proto.enemy.loop", "c"), catalog)
    assert dict(loop.outcomes) == {"b": 8}


def test_ambiguous_or_private_conditioning_is_explicit_unavailable():
    catalog = _catalog()
    assert not next_intent_law(
        _enemy("proto.enemy.ambiguous", "stab"), catalog,
    ).available
    assert next_intent_law(
        _enemy("proto.enemy.ambiguous", "guard"), catalog,
    ).reason == "consecutive_move_history_needed"
    assert next_intent_law(
        _enemy("proto.enemy.random", "not-a-move"), catalog,
    ).status == "UNAVAILABLE"


def test_unknown_catalog_schema_or_bad_grid_fails_closed():
    raw = copy.deepcopy(_catalog())
    raw["native_next_int_weights"]["2"] = [7, 2]
    with pytest.raises(ValueError, match="sum to one"):
        validate_public_intent_catalog(raw)
    raw = copy.deepcopy(_catalog())
    raw["schema"] = "unsupported"
    with pytest.raises(ValueError, match="schema"):
        validate_public_intent_catalog(raw)


def test_next_intent_tokens_are_enemy_owned_and_do_not_change_value(tmp_path):
    macro, combat = _models()
    catalog = _catalog()
    state = _combat()
    state["combat"]["enemies"][0]["enemy_id"] = "proto.enemy.random"
    state["combat"]["enemies"][0]["move_id"] = "opener"
    actions = (LegalAction("wait", "end_turn", "{}"),)
    original = macro.encode_decision(state, actions)
    enhanced = attach_visible_threat(original, state, intent_catalog=catalog)
    assert enhanced.actions == original.actions
    assert enhanced.neutral_value == original.neutral_value
    assert len(enhanced.objects) == len(original.objects) + 3
    enemy_indices = [
        i for i, obj in enumerate(enhanced.objects[:len(original.objects)])
        if obj.kind == KINDS.index("enemy")
        and obj.zone == ZONES.index("enemy")
    ]
    assert len(enemy_indices) == 1
    forecast = enhanced.objects[-2:]
    assert all(obj.owner == enemy_indices[0] for obj in forecast)
    assert all(edge.source == enemy_indices[0]
               for edge in enhanced.relations[-2:])
    combat.h26_intent_catalog = catalog
    via_model = combat.encode_decision(state, actions)
    assert via_model == enhanced
    path = tmp_path / "combat-with-law.pt"
    combat.save(path)
    restored = JointActPilotNet.load(path)
    assert restored.h26_intent_catalog == catalog
    assert restored.encode_decision(state, actions) == enhanced
    # No next-turn source data enters monotone neutral value encoding.
    assert restored.encode_decision(state, actions).neutral_value == original.neutral_value
