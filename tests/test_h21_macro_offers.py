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
        act("take_reward_extra_gold", GroupIndex=5),
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


def test_card_bundle_actions_expose_every_card_in_selected_bundle():
    frame = state()
    frame["reward"] = {
        "card_resolved": False,
        "card_bundles": [
            ["silent.strike", "silent.defend"],
            ["silent.strike", "silent.neutralize"],
        ],
    }
    actions = [act("take_reward_bundle", Index=0),
               act("take_reward_bundle", Index=1)]
    decision = encode(frame, actions)
    first, second = (a.target for a in decision.actions)
    assert first != second
    assert decision.actions[0].semantic == decision.actions[1].semantic
    cards_a = {
        decision.objects[e.target].identity for e in decision.relations
        if e.source == first and e.target != first
        and decision.objects[e.target].kind == KINDS.index("card")
    }
    cards_b = {
        decision.objects[e.target].identity for e in decision.relations
        if e.source == second and e.target != second
        and decision.objects[e.target].kind == KINDS.index("card")
    }
    assert len(cards_a) == len(cards_b) == 2
    assert cards_a != cards_b


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


def test_event_potion_uses_one_offer_and_explicit_destination_slot():
    frame = state()
    frame["event"] = {
        "event_id": "event.potions",
        "pending_potion_replacement": {
            "choice_id": "replace", "potion_id": "potion.acid",
            "candidate_slots": [0, 1, 2],
        },
    }
    actions = [
        act("take_event_potion", Slot=0),
        act("take_event_potion", Slot=2),
        act("replace_event_potion", Slot=1),
    ]
    encoded = encode(frame, actions)
    first, second, third = encoded.actions
    assert first.target == second.target == third.target
    assert len({first.source, second.source, third.source}) == 3
    assert all(item.source >= 0 for item in encoded.actions)
    assert encoded.objects[first.target].kind == KINDS.index("potion")
    assert encoded.objects[third.source].kind == KINDS.index("potion")

    wrong = copy.deepcopy(frame)
    wrong["event"]["pending_potion_replacement"]["candidate_slots"] = [1]
    with pytest.raises(ValueError, match="visible candidate"):
        encode(wrong, [act("take_event_potion", Slot=0)])


def test_macro_potion_choice_points_to_its_visible_inventory_slot():
    frame = state()
    actions = [
        act("use_potion", Slot=1, TargetEnemyId=None),
        act("leave_reward"),
    ]
    decision = encode(frame, actions)
    source = decision.actions[0].source
    assert source >= 0
    assert decision.objects[source].kind == KINDS.index("potion")
    assert decision.objects[source].zone == ZONES.index("inventory")
    assert decision.actions[0].target == -1
    assert decision.actions[0].kind != decision.actions[1].kind
    with pytest.raises(ValueError, match="ungrounded enemy target"):
        encode(frame, [act("use_potion", Slot=1, TargetEnemyId=12)])


def test_known_silent_card_hash_collision_is_resolved_in_reward_options():
    frame = state()
    frame["reward"] = {
        "card_options": [
            "proto.silent.piercing_wail", "proto.silent.dagger_spray",
        ],
    }
    actions = [
        act("take_reward_card", Index=0),
        act("take_reward_card", Index=1),
    ]
    encoded = encode(frame, actions)
    ids = [encoded.objects[token.target].identity for token in encoded.actions]
    assert len(set(ids)) == 2
    assert encoded.actions[0].kind == encoded.actions[1].kind
    assert encoded.actions[0].target != encoded.actions[1].target


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


def test_action_payload_guards_reject_unrepresented_choice_dimensions():
    frame = state()
    frame["reward"] = {
        "card_options": ["silent.strike"],
        "pending_card_groups": [
            {"group_index": 0, "card_options": ["silent.strike"]},
            {"group_index": 1, "card_options": ["silent.defend"]},
        ],
        "pending_extra_potion_rewards": [
            {"group_index": 2, "potion_id": "potion.acid"},
        ],
    }
    with pytest.raises(ValueError, match="Index=-1"):
        encode(frame, [act(
            "skip_reward_card_group", GroupIndex=1, Index=0)])
    with pytest.raises(ValueError, match="Unexpected replacement slot"):
        encode(frame, [act(
            "take_reward_extra_potion", GroupIndex=2, Slot=1)])
    with pytest.raises(ValueError, match="Unaudited macro action payload fields"):
        encode(frame, [act(
            "take_reward_card", Index=0, BonusDamage=3)])
    with pytest.raises(ValueError, match="two distinct"):
        encode(frame, [act(
            "rest_cook", CardInstanceIds=[201, 201])])


def test_opaque_instance_renaming_does_not_change_owned_card_action():
    before = state()
    after = copy.deepcopy(before)
    for i, item in enumerate(after["deck"]):
        item["instance_id"] = 9000 + i
    lhs = encode(before, [act("remove_card", CardInstanceId=201)])
    rhs = encode(after, [act("remove_card", CardInstanceId=9000)])
    assert lhs == rhs


def test_hash_collisions_raise_diagnostics_instead_of_aliasing(monkeypatch):
    import sts2_ai.training.h21_stable_identity as identities

    frame = state()
    frame["shop"] = {
        "card_offers": [
            {"offer_id": 1, "item_id": "silent.strike",
             "price": 40, "sold": False},
            {"offer_id": 2, "item_id": "silent.defend",
             "price": 45, "sold": False},
        ],
    }
    original = identities.stable_id

    def collide(name):
        return 7 if name in {"silent.strike", "silent.defend"} else original(name)

    monkeypatch.setattr(identities, "stable_id", collide)
    with pytest.raises(ValueError, match="stable full identity collision"):
        encode(frame, [act("buy_card", OfferId=1)])


def test_action_kind_collision_fails_explicitly(monkeypatch):
    import sts2_ai.training.h21_stable_identity as identities

    original = identities.stable_id

    def collide(name):
        return 7 if name.startswith("action_kind:") else original(name)

    monkeypatch.setattr(identities, "stable_id", collide)
    with pytest.raises(ValueError, match="stable full identity collision"):
        encode(state(), [act("start_run"), act("leave_reward")])


def test_completed_room_history_is_ordered_and_node_ids_are_opaque():
    from sts2_ai.training.h21_macro_offer_codec import PREVIOUS_ROOM

    frame = state()
    frame["completed_rooms"] = [
        {"act": 1, "floor": 1, "node_id": "arbitrary-a", "room_type": "Combat"},
        {"act": 1, "floor": 2, "node_id": "arbitrary-b", "room_type": "Shop"},
        {"act": 1, "floor": 3, "node_id": "arbitrary-c", "room_type": "Elite"},
    ]
    actions = [act("leave_reward")]
    original = encode(frame, actions)
    predecessor = [edge for edge in original.relations
                   if edge.relation_type == PREVIOUS_ROOM]
    assert len(predecessor) == 2
    assert predecessor[0].target != predecessor[1].target

    renamed = copy.deepcopy(frame)
    for index, visit in enumerate(renamed["completed_rooms"]):
        visit["node_id"] = f"renamed-{index + 100}"
    assert encode(renamed, actions) == original

    reordered = copy.deepcopy(frame)
    reordered["completed_rooms"] = [
        reordered["completed_rooms"][i] for i in (0, 2, 1)]
    changed = encode(reordered, actions)
    assert changed != original
    # History does not use raw NodeId values as neural features.
    assert len([r for r in changed.relations
                if r.relation_type == PREVIOUS_ROOM]) == 2


def test_combat_play_pile_and_enemy_move_are_owner_local():
    frame = state()
    frame["combat"] = {
        "turn": 3, "energy": 2, "player_block": 3,
        "hand": [], "draw_pile_count": 4, "discard_pile": [],
        "exhaust_pile": [],
        "play_pile": [{
            "instance_id": 700, "card_id": "silent.strike",
            "upgrade_level": 1, "state": {},
            "is_temporary": False,
            "enchantment": {"kind": "Sharp", "amount": 3},
            "effective_energy_cost": 1, "cost_kind": "Fixed",
        }],
        "enemies": [{
            "instance_id": 44, "enemy_id": "enemy.boss", "hp": 52, "block": 0,
            "move_id": "sweep", "is_minion": False, "powers": [],
            "statuses": {},
        }],
        "player_powers": [], "history": {"cards_played_this_turn": 1},
        "pending_choice_id": "opaque-dialog-123",
    }
    actions = [act("end_turn")]
    first = encode(frame, actions)
    played = [i for i, obj in enumerate(first.objects)
              if obj.kind == KINDS.index("card")
              and obj.zone == ZONES.index("event")
              and obj.owner >= 0]
    assert len(played) == 1
    assert sum(obj.owner == played[0] for obj in first.objects) >= 3
    changed = copy.deepcopy(frame)
    changed["combat"]["play_pile"][0]["enchantment"]["amount"] = 4
    assert encode(changed, actions) != first
    changed = copy.deepcopy(frame)
    changed["combat"]["enemies"][0]["move_id"] = "defend"
    assert encode(changed, actions) != first
    changed = copy.deepcopy(frame)
    changed["combat"]["enemies"][0]["is_minion"] = True
    assert encode(changed, actions) != first
    # The exact pending-choice reference must be opaque; only its presence
    # affects policy features until a dedicated choice relation exists.
    changed = copy.deepcopy(frame)
    changed["combat"]["pending_choice_id"] = "opaque-dialog-999"
    assert encode(changed, actions) == first
    changed["combat"]["pending_choice_id"] = None
    assert encode(changed, actions) != first


def test_grounded_macro_model_and_v20_checkpoint_separation(tmp_path):
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_macro_transformer_v2 import (
        H21GroundedMacroTransformer,
        collate_h21,
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
