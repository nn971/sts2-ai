"""H21 grounded reward/shop/event action and visibility acceptance tests."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS, ZONES
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h21_macro_offer_codec import (
    H21_OBJECT_SCHEMA,
    OPTION_OF,
    encode_h21_macro_decision,
)


def state():
    return {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [
            {"instance_id": 201, "card_id": "silent.strike",
             "upgrade_level": 0, "state": {}},
            {"instance_id": 202, "card_id": "silent.strike",
             "upgrade_level": 0, "state": {}},
            {"instance_id": 203, "card_id": "silent.strike",
             "upgrade_level": 1, "state": {}},
        ],
        "relics": [], "potions": [
            {"slot": 1, "potion_id": "potion.weak", "state": {}},
        ],
        "potion_slot_capacity": 3, "potion_slots_occupied": 1,
        "potion_slots_empty": 2,
        "map": [], "current_map_node_id": None,
        "combat": None,
    }


def act(kind, **fields):
    return LegalAction(
        kind + ":" + json.dumps(fields, sort_keys=True), kind,
        json.dumps(fields, separators=(",", ":"), sort_keys=True))


def encode(obs, actions):
    return encode_h21_macro_decision(
        obs, actions, public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1)


def test_reward_options_are_grounded_with_card_upgrades_and_extra_groups():
    frame = state()
    frame["reward"] = {
        "card_options": ["silent.strike", "silent.defend"],
        "card_resolved": False,
        "pending_card_groups": [
            {"group_index": 0, "card_options": ["silent.strike", "silent.defend"],
             "option_upgrade_flags": [False, True]},
            {"group_index": 1, "card_options": ["silent.strike"],
             "option_upgrade_flags": [True],
             "card_previews": [
                 {"card_id": "silent.strike", "upgrade_level": 2,
                  "enchantment": {"kind": "Nimble", "amount": 3}}],
             },
        ],
        "potion_option": "potion.acid",
        "relic_options": ["relic.rare"],
        "pending_extra_relic_rewards": [
            {"group_index": 4, "relic_id": "relic.extra", "is_wax": True},
        ],
        "pending_extra_gold_rewards": [
            {"group_index": 5, "amount": 150},
        ],
        "pending_extra_potion_rewards": [
            {"group_index": 6, "potion_id": "potion.extra"},
        ],
    }
    actions = [
        act("take_reward_card", Index=0),
        act("take_reward_card", Index=1),
        act("take_reward_card_group", GroupIndex=1, Index=0),
        act("skip_reward_card_group", GroupIndex=1, Index=-1),
        act("reroll_reward_card", GroupIndex=0),
        act("replace_reward_potion", Slot=1),
        act("take_reward_relic", Index=0),
        act("take_reward_extra_relic", Index=4),
        act("take_reward_extra_gold", Index=5),
        act("take_reward_extra_potion", GroupIndex=6),
    ]
    decision = encode(frame, actions)
    assert decision.schema == H21_OBJECT_SCHEMA
    assert len(set(decision.actions)) == len(actions)
    assert all(token.target >= 0 for token in decision.actions)
    assert decision.actions[0].semantic == decision.actions[1].semantic
    assert decision.actions[0].target != decision.actions[1].target
    assert decision.actions[0].target != decision.actions[2].target
    upgraded = decision.objects[decision.actions[2].target]
    assert upgraded.kind == KINDS.index("card")
    assert upgraded.zone == ZONES.index("reward")
    assert upgraded.owner >= 0
    assert decision.actions[3].target == decision.objects[
        decision.actions[2].target].owner
    assert decision.actions[5].source >= 0
    assert decision.objects[decision.actions[5].source].identity != 0
    assert any(edge.relation_type == OPTION_OF for edge in decision.relations)


def test_shop_prices_are_attached_to_real_offers_and_replacements():
    frame = state()
    frame["shop"] = {
        "card_offers": [
            {"offer_id": 17, "item_id": "silent.strike", "price": 65,
             "base_price": 85, "on_sale": True, "sold": False},
            {"offer_id": 18, "item_id": "silent.strike", "price": 125,
             "base_price": 125, "on_sale": False, "sold": False},
        ],
        "potion_offers": [
            {"offer_id": 25, "item_id": "potion.acid", "price": 45, "sold": False},
        ],
        "relic_offers": [
            {"offer_id": 35, "item_id": "relic.mirror", "price": 75, "sold": False},
        ],
        "removal_price": 75, "removal_used": False,
    }
    actions = [
        act("buy_card", OfferId=17),
        act("buy_card", OfferId=18),
        act("buy_relic", OfferId=35),
        act("replace_shop_potion", OfferId=25, Slot=1),
        act("remove_card", CardInstanceId=201),
        act("remove_card", CardInstanceId=202),
        act("remove_card", CardInstanceId=203),
        act("leave_shop"),
    ]
    decision = encode(frame, actions)
    assert decision.actions[0].semantic == decision.actions[1].semantic
    a, b = (decision.objects[t.target] for t in decision.actions[:2])
    assert a.identity == b.identity
    assert a.numeric != b.numeric  # public prices distinguish offers
    assert decision.actions[0].target != decision.actions[1].target
    assert decision.actions[3].source >= 0
    assert decision.actions[3].target >= 0
    # Identical deck card copies remain one selectable class.
    assert decision.actions[4] == decision.actions[5]
    assert decision.actions[6] != decision.actions[5]


def test_event_choice_outcomes_and_guaranteed_items_are_owner_local():
    frame = state()
    frame["event"] = {
        "event_id": "native.ancient_temple", "offered_choice_ids": ["gold", "heal"],
        "visible_choice_values": [
            {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6,
             "guaranteed_card_id": "silent.rare"},
            {"choice_id": "heal", "heal": 15},
        ],
    }
    actions = [act("event_choice", ChoiceId="gold"),
               act("event_choice", ChoiceId="heal")]
    decision = encode(frame, actions)
    a, b = (decision.objects[t.target] for t in decision.actions)
    assert a.kind == b.kind == KINDS.index("event_choice")
    assert a.identity != b.identity
    assert a.owner == b.owner
    assert any(obj.owner == decision.actions[0].target
               for obj in decision.objects)
    assert decision.actions[0].target != decision.actions[1].target
    changed = copy.deepcopy(frame)
    changed["event"]["visible_choice_values"][0]["gold_gain"] = 50
    assert encode(changed, actions) != decision


def test_unbound_offers_and_unknown_payloads_fail_closed():
    frame = state()
    frame["shop"] = {"card_offers": [
        {"offer_id": 1, "item_id": "silent.strike", "price": 50, "sold": False},
    ]}
    with pytest.raises(ValueError, match="no public match"):
        encode(frame, [act("buy_card", OfferId=3)])
    with pytest.raises(ValueError, match="Unaudited nonempty macro action"):
        encode(frame, [act("unseen_action", Id=50)])
    frame = state()
    frame["event"] = {"event_id": "native.event",
                      "offered_choice_ids": ["good"]}
    with pytest.raises(ValueError, match="published offer list"):
        encode(frame, [act("event_choice", ChoiceId="bad")])


def test_opaque_instance_renaming_does_not_change_owned_card_action():
    before = state()
    after = copy.deepcopy(before)
    for i, item in enumerate(after["deck"]):
        item["instance_id"] = 9000 + i
    lhs = encode(before, [act("remove_card", CardInstanceId=201)])
    rhs = encode(after, [act("remove_card", CardInstanceId=9000)])
    assert lhs == rhs


def test_grounded_macro_model_and_v20_checkpoint_separation(tmp_path):
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_macro_transformer_v2 import (
        H21GroundedMacroTransformer, collate_h21,
    )
    from sts2_ai.models.shared_map_transformer_v2 import H20MapGraphTransformer
    from sts2_ai.models.shared_transformer import H7Config

    frame = state()
    frame["event"] = {
        "event_id": "native.event", "offered_choice_ids": ["a", "b"],
        "visible_choice_values": [
            {"choice_id": "a", "gold_gain": 70},
            {"choice_id": "b", "heal": 10},
        ],
    }
    decision = encode(frame, [act("event_choice", ChoiceId="a"),
                              act("event_choice", ChoiceId="b")])
    torch.manual_seed(4)
    model = H21GroundedMacroTransformer(
        H7Config(width=32, heads=4, layers=1, feedforward=64)).eval()
    batch = collate_h21([decision])
    with torch.no_grad():
        logits, values = model(batch)
    assert logits.shape == (1, 2)
    assert values.shape == (1,)
    assert torch.isfinite(logits).all()
    path = tmp_path / "h21.pt"
    model.save_checkpoint(path)
    loaded = H21GroundedMacroTransformer.load_checkpoint(path).eval()
    with torch.no_grad():
        got, value = loaded(batch)
    torch.testing.assert_close(logits, got)
    torch.testing.assert_close(values, value)
    with pytest.raises(ValueError, match="Unsupported H20"):
        H20MapGraphTransformer.load_checkpoint(path)
