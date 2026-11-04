"""Extended public-map probe varies routes without changing the short smoke."""
from __future__ import annotations

import json
from collections import Counter

from tools.probe_h21_live_public import choose, missing_executed_action_groups


def _action(node: str) -> dict:
    return {
        "actionId": "node:" + node,
        "kind": "choose_map_node",
        "payloadJson": json.dumps({"NodeId": node}),
    }


def test_short_smoke_action_order_is_stable():
    actions = [_action("z"), _action("a")]
    assert choose(actions)["actionId"] == "node:a"


def test_extended_probe_targets_revealed_shop_and_elite_rooms():
    observation = {
        "map": [
            {"node_id": "combat", "room_type": 0},
            {"node_id": "elite", "room_type": 1},
            {"node_id": "shop", "room_type": 3},
        ],
    }
    actions = [_action("combat"), _action("elite"), _action("shop")]
    assert choose(actions, public_state=observation, seed="s", step=1,
                  route_variant=0)["actionId"] == "node:shop"
    assert choose(actions, public_state=observation, seed="s", step=1,
                  route_variant=1)["actionId"] == "node:elite"


def test_opaque_node_renaming_changes_no_public_room_priority():
    source = {
        "map": [
            {"node_id": "first", "room_type": "Shop"},
            {"node_id": "second", "room_type": "Elite"},
        ],
    }
    new = {
        "map": [
            {"node_id": "x-unknown", "room_type": "Shop"},
            {"node_id": "y-random", "room_type": "Elite"},
        ],
    }
    assert choose([_action("first"), _action("second")],
                  public_state=source, route_variant=0)["actionId"] == "node:first"
    assert choose([_action("y-random"), _action("x-unknown")],
                  public_state=new, route_variant=0)["actionId"] == "node:x-unknown"


def test_continuation_rollout_actually_buys_before_leaving_shop():
    public = {"map": []}
    actions = [
        {"actionId": "leave", "kind": "leave_shop", "payloadJson": "{}"},
        {"actionId": "buy", "kind": "buy_relic",
         "payloadJson": '{"OfferId":7}'},
    ]
    assert choose(actions, public_state=public, seed="shop", step=2,
                  route_variant=0)["actionId"] == "leave"
    assert choose(actions, public_state=public, seed="shop", step=2,
                  route_variant=0, continuation_coverage=True
                  )["actionId"] == "buy"


def test_continuation_policy_exercises_both_deck_pick_and_optional_finish():
    actions = [
        {"actionId": "finish", "kind": "finish_relic_deck_choice",
         "payloadJson": "{}"},
        {"actionId": "pick", "kind": "choose_relic_deck_card",
         "payloadJson": '{"CardInstanceId":12}'},
    ]
    assert choose(actions, public_state={"map": []},
                  route_variant=0, continuation_coverage=True
                  )["actionId"] == "pick"
    assert choose(actions, public_state={"map": []},
                  route_variant=1, continuation_coverage=True
                  )["actionId"] == "finish"


def test_continuation_acceptance_requires_actual_purchase_not_just_legal_menu():
    purchase = "buy_card,buy_potion,buy_relic"
    # A purchase that was merely offered does not satisfy the live-rollout gate.
    assert missing_executed_action_groups(
        Counter({"leave_shop": 3, "event_choice": 2}), [purchase],
    ) == [purchase]
    assert missing_executed_action_groups(
        Counter({"leave_shop": 2, "buy_potion": 1}), [purchase],
    ) == []
    assert missing_executed_action_groups(
        Counter({"buy_card": 1}), [purchase, "event_choice,continue_event"],
    ) == ["event_choice,continue_event"]


def test_debuff_coverage_prefers_only_visible_legal_debuff_cards():
    public = {
        "combat": {"hand": [
            {"instance_id": 12, "card_id": "proto.silent.strike"},
            {"instance_id": 13, "card_id": "proto.silent.neutralize"},
        ]},
    }
    actions = [
        {"actionId": "strike", "kind": "play_card",
         "payloadJson": json.dumps({"CardInstanceId": 12, "TargetEnemyId": 1})},
        {"actionId": "neutralize", "kind": "play_card",
         "payloadJson": json.dumps({"CardInstanceId": 13, "TargetEnemyId": 1})},
    ]
    assert choose(actions, public_state=public,
                  prefer_enemy_debuff_cards=True)["actionId"] == "neutralize"
    # No invented action: the preference only reorders genuinely legal moves.
    assert choose(actions[:1], public_state=public,
                  prefer_enemy_debuff_cards=True)["actionId"] == "strike"


def test_observed_enemy_effects_require_positive_public_power_stacks():
    from tools.probe_h21_live_public import observed_canonical_enemy_effects

    frame = {"combat": {"enemies": [
        {"powers": [
            {"power_id": "proto.power.poison", "stacks": 3},
            {"power_id": "proto.power.weak", "stacks": 0},
            {"power_id": "proto.power.strength", "stacks": 12},
        ]},
    ]}}
    assert observed_canonical_enemy_effects(frame) == {"proto.power.poison"}
    assert observed_canonical_enemy_effects({"combat": None}) == set()
