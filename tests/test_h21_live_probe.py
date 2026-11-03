"""Extended public-map probe varies routes without changing the short smoke."""
from __future__ import annotations

import json

from tools.probe_h21_live_public import choose


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
