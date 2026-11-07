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
    PublicIntentHistory,
    next_intent_law,
    public_intent_coverage,
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
                "enemy_id": "proto.enemy.max2",
                "move_policy": "UniformRandomAfterOpener",
                "move_ids": ["opener", "stab", "guard"],
                "max_consecutive_uses": [0, 2, 0],
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
        _enemy("proto.enemy.max2", "stab"), catalog,
    ).reason == "consecutive_move_history_needed"
    assert next_intent_law(
        _enemy("proto.enemy.ambiguous", "guard"), catalog,
    ).available
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


def _machine_catalog():
    raw = copy.deepcopy(_catalog())
    raw["enemies"].append({
        "enemy_id": "proto.enemy.machine",
        "move_policy": "StateMachine",
        "move_ids": ["first", "a", "b", "end"],
        "max_consecutive_uses": [0, 0, 0, 0],
        "opening_move_indices": [0],
        "random_move_pool_start_index": 0,
        "move_loop_start_index": 0,
        "ai": {
            "initial_state_id": "first_move",
            "states": [
                {"id": "first_move", "kind": "Move", "move_index": 0,
                 "next_state_id": "lottery", "branches": [],
                 "conditional_branches": []},
                {"id": "lottery", "kind": "Random", "move_index": None,
                 "next_state_id": None, "branches": [
                     {"target_state_id": "a_move", "weight": 1,
                      "repeat_rule": "CanRepeatForever",
                      "max_times": 0, "requires_available_summon": False},
                     {"target_state_id": "b_move", "weight": 2,
                      "repeat_rule": "CannotRepeat",
                      "max_times": 0, "requires_available_summon": False},
                 ], "conditional_branches": []},
                {"id": "a_move", "kind": "Move", "move_index": 1,
                 "next_state_id": "after_a", "branches": [],
                 "conditional_branches": []},
                {"id": "b_move", "kind": "Move", "move_index": 2,
                 "next_state_id": "lottery", "branches": [],
                 "conditional_branches": []},
                {"id": "after_a", "kind": "Conditional", "move_index": None,
                 "next_state_id": None, "branches": [],
                 "conditional_branches": [
                     {"condition": "IsAlone", "target_state_id": "end_move",
                      "value": None},
                 ]},
                {"id": "end_move", "kind": "Move", "move_index": 3,
                 "next_state_id": "lottery", "branches": [],
                 "conditional_branches": []},
            ],
        },
    })
    raw["native_next_int_weights"]["4"] = [2, 2, 2, 2]
    return validate_public_intent_catalog(raw)


def test_public_state_machine_weighted_and_conditional_transitions():
    catalog = _machine_catalog()
    first = _enemy("proto.enemy.machine", "first")
    law = next_intent_law(first, catalog)
    assert law.available
    assert dict(law.outcomes) == {"a": 3, "b": 5}
    assert sum(weight for _, weight in law.outcomes) == 8
    after_b = next_intent_law(
        _enemy("proto.enemy.machine", "b"), catalog,
    )
    assert dict(after_b.outcomes) == {"a": 8}
    after_a = next_intent_law(
        _enemy("proto.enemy.machine", "a"), catalog,
        public_enemies=[_enemy("proto.enemy.machine", "a")],
    )
    assert dict(after_a.outcomes) == {"end": 8}
    without_formation = next_intent_law(
        _enemy("proto.enemy.machine", "a"), catalog,
    )
    assert not without_formation.available
    assert without_formation.reason == "state_machine_condition_history_needed"


def test_public_intent_history_turn_boundary_and_reset():
    tracker = PublicIntentHistory()
    first = _combat()
    first["combat"]["enemies"][0]["move_id"] = "a"
    first["combat"]["enemies"][0]["hp"] = 20
    initial = tracker.observe(first)
    assert initial == {2: ()}
    assert tracker.observe(first) == initial
    second = copy.deepcopy(first)
    second["combat"]["turn"] = 2
    second["combat"]["enemies"][0]["move_id"] = "b"
    assert tracker.observe(second) == {2: ("a",)}
    third = copy.deepcopy(second)
    third["combat"]["turn"] = 3
    third["combat"]["enemies"][0]["move_id"] = "b"
    assert tracker.observe(third) == {2: ("a", "b")}
    event = copy.deepcopy(first)
    event["combat"] = None
    assert tracker.observe(event) == {}


def test_per_enemy_coverage_reports_unavailable_reasons():
    import json

    from sts2_ai.emulator.protocol import Observation
    from sts2_ai.training.h16_joint_act_pilot import PilotDecision, PilotEpisode

    catalog = _machine_catalog()
    state = _combat()
    state["combat"]["enemies"][0]["enemy_id"] = "proto.enemy.machine"
    state["combat"]["enemies"][0]["move_id"] = "first"
    action = (LegalAction("wait", "end_turn", "{}"),)
    decision = PilotDecision(
        Observation("prototype-fair-v0", json.dumps(state), "public"),
        action, 0, (0.0,), h26_public_intent_history={2: ()},
    )
    episode = PilotEpisode(
        "example", (decision,), (state, {**state, "combat": None}),
        "defeat", 0, 0,
    )
    result = public_intent_coverage([episode], catalog)
    assert result["living"] == 1
    assert result["living_available"] == 1
    assert result["by_enemy"]["proto.enemy.machine"]["available"] == 1
    assert result["by_enemy"]["proto.enemy.machine"]["living_availability_rate"] == 1
