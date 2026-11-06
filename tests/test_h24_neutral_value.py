"""H24 hard-value context is invariant on certified resource coordinates."""
from copy import deepcopy

import pytest

from sts2_ai.models.shared_game_objects import KINDS
from sts2_ai.training.h24_applicability import ResourceConstraintSelection
from sts2_ai.training.h24_neutral_value import (
    assert_h24_pair_neutral_parity,
    encode_h24_neutral_value,
    neutralize_h24_state,
)
from sts2_ai.training.h24_resource_monotone import SafeResourceVocabulary


def _state():
    return {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [{
            "instance_id": 201, "card_id": "silent.strike",
            "upgrade_level": 0, "state": {},
        }],
        "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [],
        "current_map_node_id": None, "combat": None,
        "event": {
            "event_id": "native.ancient_temple",
            "offered_choice_ids": ["gold", "heal"],
            "visible_choice_values": [
                {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6},
                {"choice_id": "heal", "heal": 15},
            ],
        },
    }


def test_h24_hp_and_gold_same_encoded_context():
    lower, higher = _state(), _state()
    lower["hp"], lower["gold"] = 40, 60
    higher["hp"], higher["gold"] = 70, 150
    assert_h24_pair_neutral_parity(lower, higher)
    low = encode_h24_neutral_value(lower)
    high = encode_h24_neutral_value(higher)
    assert low.encoded == high.encoded
    assert (low.hp, low.gold) == (40, 60)
    assert (high.hp, high.gold) == (70, 150)
    assert low.encoded.schema == high.encoded.schema
    assert len(low.encoded.actions) == 1


def test_h24_audited_potion_identity_can_be_neutralized():
    lower, higher = _state(), _state()
    higher["potions"] = [{
        "slot": 1, "potion_id": "proto.potion.block", "state": {},
    }]
    higher["potion_slots_occupied"] = 1
    higher["potion_slots_empty"] = 2
    vocabulary = SafeResourceVocabulary(
        potion_ids=("proto.potion.block",),
    )
    assert_h24_pair_neutral_parity(lower, higher, vocabulary=vocabulary)
    high = encode_h24_neutral_value(higher, vocabulary=vocabulary)
    assert high.potions == (1.0,)
    low = encode_h24_neutral_value(lower, vocabulary=vocabulary)
    assert low.potions == (0.0,)


def test_h24_uncertified_potions_remain_visible_in_value_context():
    low, high = _state(), _state()
    high["potions"] = [{"slot": 0, "potion_id": "proto.potion.block", "state": {}}]
    high["potion_slots_occupied"], high["potion_slots_empty"] = 1, 2
    with pytest.raises(ValueError, match="neutral H21"):
        assert_h24_pair_neutral_parity(low, high)
    # Even with other certified potions, unknown inventory is preserved.
    projected, *_ = neutralize_h24_state(high,
        vocabulary=SafeResourceVocabulary(potion_ids=("proto.potion.fire",)))
    assert len(projected["potions"]) == 1


def test_h24_relics_stay_visible_by_default():
    low, high = _state(), _state()
    high["relics"] = [{"relic_id": "proto.relic.paper_krane", "state": {}}]
    with pytest.raises(ValueError, match="neutral H21"):
        assert_h24_pair_neutral_parity(low, high)
    # A relic vocabulary requires an explicit switch and does not certify
    # beneficial mechanics by itself.
    vocab = SafeResourceVocabulary(relic_ids=("proto.relic.paper_krane",))
    with pytest.raises(ValueError, match="disabled"):
        encode_h24_neutral_value(high, vocabulary=vocab)
    assert_h24_pair_neutral_parity(
        low, high, selection=ResourceConstraintSelection(relic=True),
        vocabulary=vocab,
    )


def test_h24_potion_bookkeeping_and_input_not_mutated():
    state = _state()
    state["potions"] = [{"slot": 2, "potion_id": "proto.potion.block", "state": {}}]
    state["potion_slots_occupied"], state["potion_slots_empty"] = 1, 2
    original = deepcopy(state)
    projected, hp, gold, potions, relics = neutralize_h24_state(
        state, vocabulary=SafeResourceVocabulary(potion_ids=("proto.potion.block",)))
    assert state == original
    assert (hp, gold, potions, relics) == (55.0, 90.0, (1.0,), ())
    assert projected["hp"] == 1 and projected["gold"] == 0
    assert projected["potions"] == []
    assert projected["potion_slots_occupied"] == 0
    assert projected["potion_slots_empty"] == 3


def test_h24_rejects_invalid_potion_slot_bookkeeping():
    state = _state()
    state["potions"] = [{"slot": 2, "potion_id": "proto.potion.block", "state": {}}]
    with pytest.raises(ValueError, match="potion slot"):
        neutralize_h24_state(state)
