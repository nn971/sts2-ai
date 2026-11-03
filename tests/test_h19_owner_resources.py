"""H19 public v1 owner-linked resource acceptance and compatibility."""
from __future__ import annotations

import copy
import math

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import (
    KINDS,
    NUMERIC_FIELDS,
    OWNER_ATTRIBUTE_KIND_V2,
)
from sts2_ai.training.h19_resource_codec import (
    EMULATOR_PUBLIC_SCHEMA_V1,
    H19_OBJECT_SCHEMA,
    encode_h19_resource_decision,
)

ACTIONS = (LegalAction("pass", "skip", "{}"),)


def resource_state() -> dict:
    return {
        "hp": 62, "max_hp": 70, "gold": 35, "act": 1, "floor": 5,
        "deck": [
            {"instance_id": 101, "card_id": "demo.strike",
             "upgrade_level": 0, "state": {"uses": 1},
             "enchantment": {"kind": "Nimble", "amount": 2, "disabled": False}},
            {"instance_id": 102, "card_id": "demo.strike",
             "upgrade_level": 0, "state": {"uses": 2},
             "enchantment": {"kind": "Nimble", "amount": 2, "disabled": False}},
        ],
        "relics": [
            {"relic_id": "demo.relic", "state": {"uses": 1},
             "is_wax": False, "is_melted": True},
        ],
        "potions": [{"slot": 1, "potion_id": "demo.potion",
                     "state": {"uses": 3}}],
        "potion_slot_capacity": 3,
        "potion_slots_occupied": 1,
        "potion_slots_empty": 2,
        "combat": None, "map": [],
    }


def encode(state: dict, actions=ACTIONS):
    return encode_h19_resource_decision(
        state, actions, public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1)


def owned(decision, index: int):
    return [obj for obj in decision.objects
            if obj.kind == OWNER_ATTRIBUTE_KIND_V2 and obj.owner == index]


def test_owner_linked_attributes_are_typed_and_reuse_field_identity():
    decision = encode(resource_state())
    assert decision.schema == H19_OBJECT_SCHEMA
    assert decision.phase == "macro"
    assert len(decision.actions) == 1
    assert len([x for x in decision.objects if x.kind == KINDS.index("card")]) == 2
    deck_indices = [i for i, item in enumerate(decision.objects)
                    if item.kind == KINDS.index("card")]
    relic_idx = next(i for i, x in enumerate(decision.objects)
                     if x.kind == KINDS.index("relic"))
    potion_idx = next(i for i, x in enumerate(decision.objects)
                      if x.kind == KINDS.index("potion"))
    stacks = NUMERIC_FIELDS.index("stacks")
    deck_a, deck_b = (owned(decision, i) for i in deck_indices)
    # Uses=1 and uses=2 share the same semantic field but have different
    # numeric values and owner pointers.
    a = [x for x in deck_a if x.numeric[stacks] == pytest.approx(math.tanh(1 / 10))]
    b = [x for x in deck_b if x.numeric[stacks] == pytest.approx(math.tanh(2 / 10))]
    assert len(a) > 0 and len(b) > 0
    assert a[0].identity in {x.identity for x in b}
    # The same state.uses concept on a relic is attached to the relic rather
    # than represented as a detached combat status.
    assert {x.identity for x in deck_a} & {x.identity for x in owned(decision, relic_idx)}
    assert owned(decision, potion_idx)
    assert owned(decision, 0)  # capacity and occupancy are global attributes
    assert all(x.owner >= 0 for x in decision.objects if x.kind == OWNER_ATTRIBUTE_KIND_V2)


def test_public_modifiers_afflictions_and_boolean_presence_change_encoding():
    original = resource_state()
    a = encode(original)
    modified = copy.deepcopy(original)
    modified["deck"][0]["enchantment"]["disabled"] = True
    assert encode(modified) != a
    modified = copy.deepcopy(original)
    modified["deck"][0]["enchantment"]["amount"] = 3
    assert encode(modified) != a
    modified = copy.deepcopy(original)
    modified["deck"][0]["affliction"] = {"kind": "Bound", "amount": 1}
    assert encode(modified) != a
    modified = copy.deepcopy(original)
    modified["relics"][0].pop("is_wax")
    assert encode(modified) != a
    modified = copy.deepcopy(original)
    modified["relics"][0]["is_wax"] = True
    assert encode(modified) != a
    modified = copy.deepcopy(original)
    modified["potion_slot_capacity"] = 4
    modified["potion_slots_empty"] = 3
    assert encode(modified) != a


def test_thousand_exact_clones_are_a_multiset_and_public_variants_survive():
    base = resource_state()
    first = base["deck"][0]
    base["deck"] = [{**copy.deepcopy(first), "instance_id": i} for i in range(1000)]
    one = encode(base)
    cards = [x for x in one.objects if x.kind == KINDS.index("card")]
    assert len(cards) == 1
    assert cards[0].numeric[NUMERIC_FIELDS.index("count")] == pytest.approx(
        math.tanh(math.log1p(1000) / 10))
    base["deck"][-1]["state"] = {"uses": 99}
    two = encode(base)
    assert len([x for x in two.objects if x.kind == KINDS.index("card")]) == 2


def test_instance_renaming_does_not_change_resource_tokens():
    before = resource_state()
    after = copy.deepcopy(before)
    after["deck"][0]["instance_id"] = 5001
    after["deck"][1]["instance_id"] = 5002
    assert encode(before) == encode(after)


def test_combat_card_and_history_owners_are_explicit():
    state = resource_state()
    state["combat"] = {
        "turn": 2, "energy": 2, "player_block": 0,
        "hand": [{
            "instance_id": 7, "card_id": "demo.strike", "state": {},
            "enchantment": {"kind": "Sharp", "amount": 1},
            "effective_energy_cost": 0,
            "cost_kind": "Fixed", "is_temporary": False,
            "keyword_overrides": [{"keyword": "Retain", "enabled": True}],
        }],
        "draw_pile_count": 5,
        "discard_pile": [], "exhaust_pile": [],
        "enemies": [{"instance_id": 23, "enemy_id": "demo.enemy",
                     "hp": 18, "block": 0, "powers": []}],
        "player_powers": [],
        "history": {"cards_played_this_turn": 3,
                    "attacks_played_last_turn": 2},
    }
    actions = (LegalAction(
        "hit", "play_card", '{"CardInstanceId":7,"TargetEnemyId":23}'),)
    decision = encode(state, actions)
    player = next(i for i, x in enumerate(decision.objects)
                  if x.kind == KINDS.index("player"))
    hand = next(i for i, x in enumerate(decision.objects)
                if x.kind == KINDS.index("card") and x.zone == 1)
    assert owned(decision, player)  # history on player
    assert owned(decision, hand)  # modifiers on hand card
    assert decision.actions[0].source == hand
    assert decision.actions[0].target >= 0
    assert decision.objects[decision.actions[0].target].kind == KINDS.index("enemy")


def test_v1_field_contract_and_legacy_frame_fail_closed():
    state = resource_state()
    with pytest.raises(ValueError, match="prototype-ai-v1"):
        encode_h19_resource_decision(state, ACTIONS, public_schema_id="prototype-ai-v0")
    missing = copy.deepcopy(state)
    del missing["potion_slot_capacity"]
    with pytest.raises(ValueError, match="potion inventory"):
        encode(missing)
    legacy = copy.deepcopy(state)
    legacy["deck"][0]["persistent_state"] = legacy["deck"][0].pop("state")
    with pytest.raises(ValueError, match="resource.state"):
        encode(legacy)
    invalid = copy.deepcopy(state)
    invalid["potion_slots_occupied"] = 99
    with pytest.raises(ValueError, match="Inconsistent"):
        encode(invalid)


def test_new_public_fields_and_reference_ids_fail_closed():
    state = resource_state()
    state["deck"][0]["new_public_modifier"] = 1
    with pytest.raises(ValueError, match="Unclassified public resource"):
        encode(state)

    state = resource_state()
    state["deck"][0]["affliction"] = {
        "kind": "Bound", "amount": 1, "source_enemy_instance_id": 37,
    }
    with pytest.raises(ValueError, match="explicit relation"):
        encode(state)

    state = resource_state()
    state["potions"][0]["slot"] = 3
    with pytest.raises(ValueError, match="slot references"):
        encode(state)

    state = resource_state()
    state["potion_slots_occupied"] = 0
    state["potion_slots_empty"] = 3
    with pytest.raises(ValueError, match="Occupied potion slot"):
        encode(state)


def test_opaque_macro_actions_still_keep_h16_menu_collision_checks():
    state = resource_state()
    actions = (
        LegalAction("a", "event_choice", '{"ChoiceId":"gold"}'),
        LegalAction("b", "event_choice", '{"ChoiceId":"heal"}'),
    )
    decision = encode(state, actions)
    assert decision.actions[0] != decision.actions[1]
    assert all(a.source == -1 for a in decision.actions)
