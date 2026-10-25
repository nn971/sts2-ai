"""Public-only, candidate-conditioned strategic action features (H6-B).

These features are *learned inputs*, not heuristic action scores. Node IDs are
used solely to join the currently visible map DAG; hidden encounter/reward
realizations and private RNG state never enter the encoder.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from sts2_ai.models.hashed_linear import (
    _hash_features,
    neural_action_features,
)

MACRO_ACTION_SCHEMA = "sts2-h6b-public-macro-action-v1"

_ROOM_TYPE_NAMES = {
    0: "combat", 1: "elite", 2: "event", 3: "shop",
    4: "rest", 5: "boss", 6: "unknown", 7: "treasure",
}


def _room_type(room: object) -> str:
    if type(room) is int:
        return _ROOM_TYPE_NAMES.get(room, "other")
    if isinstance(room, str):
        name = room.lower().replace("_", "-")
        return name if name in _ROOM_TYPE_NAMES.values() else "other"
    return "other"


def _nodes(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = state.get("map")
    if not isinstance(raw, list):
        return {}
    return {
        node["node_id"]: node
        for node in raw
        if isinstance(node, dict) and isinstance(node.get("node_id"), str)
    }


def _children(node: dict[str, Any]) -> tuple[str, ...]:
    # Native public projection of MapNodeState.NextNodeIds.
    links = node.get("next_node_ids")
    if not isinstance(links, list):
        return ()
    return tuple(x for x in links if isinstance(x, str))


def _future_room_tokens(
    selected_id: str, graph: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    node = graph.get(selected_id)
    if node is None:
        return []
    result: list[tuple[str, float]] = []
    room = _room_type(node.get("room_type"))
    result.append((f"macro:map:room={room}", 1.0))
    floor = node.get("floor")
    if type(floor) is int:
        result.append(("macro:map:floor", min(1.0, max(0.0, floor / 16.0))))
    visited = {selected_id}
    layer = {selected_id}
    for depth in range(1, 5):
        next_layer: set[str] = set()
        for parent in layer:
            for child in _children(graph[parent]):
                if child in graph and child not in visited:
                    next_layer.add(child)
        if depth == 1:
            result.append(("macro:map:branches", min(1.0, len(next_layer) / 4.0)))
        if not next_layer:
            break
        visited.update(next_layer)
        tally = Counter(_room_type(graph[x].get("room_type"))
                        for x in next_layer)
        scale = max(1, len(next_layer))
        for room_name in sorted(tally):
            # The number and fraction of *visible possible destinations*,
            # not a probability distribution over future routes.
            result.append((
                f"macro:map:depth={depth}:room={room_name}",
                tally[room_name] / scale,
            ))
        layer = next_layer
    return result


def _public_card_counts(state: dict[str, Any]) -> Counter[str]:
    cards = state.get("deck")
    result: Counter[str] = Counter()
    if isinstance(cards, list):
        for card in cards:
            if isinstance(card, dict) and isinstance(card.get("card_id"), str):
                result[card["card_id"]] += 1
    return result


def _card_choice_tokens(
    state: dict[str, Any], action_kind: str, payload: dict[str, Any],
) -> list[tuple[str, float]]:
    counts = _public_card_counts(state)
    total = sum(counts.values())
    chosen_card = None
    if action_kind == "take_reward_card":
        reward = state.get("reward")
        index = payload.get("Index", payload.get("index"))
        if (isinstance(reward, dict)
                and isinstance(reward.get("card_options"), list)
                and type(index) is int
                and 0 <= index < len(reward["card_options"])):
            option = reward["card_options"][index]
            if isinstance(option, str):
                chosen_card = option
    elif action_kind in {"rest_upgrade", "remove_card"}:
        instance = payload.get("CardInstanceId", payload.get("card_instance_id"))
        if type(instance) is int and isinstance(state.get("deck"), list):
            for card in state["deck"]:
                if (isinstance(card, dict)
                        and card.get("instance_id") == instance
                        and isinstance(card.get("card_id"), str)):
                    chosen_card = card["card_id"]
                    break
    tokens: list[tuple[str, float]] = [
        ("macro:deck:size", min(1.0, total / 40.0)),
        ("macro:deck:unique", min(1.0, len(counts) / 30.0)),
    ]
    if chosen_card is not None:
        tokens.extend((
            (f"macro:chosen-card={chosen_card}", 1.0),
            ("macro:chosen-card:copies", min(1.0, counts[chosen_card] / 5.0)),
            ("macro:chosen-card:already-owned", float(counts[chosen_card] > 0)),
        ))
    return tokens


def macro_action_features(
    state: dict[str, Any],
    action_kind: str,
    action_payload_json: str,
    dimension: int,
) -> dict[int, float]:
    """Add explicit *public* macro context to the old semantic action vector.

    A missing/unsupported source field contributes no invented information.
    Existing action tokens are retained so warm-start is possible, but new
    extra features change behavior and require an explicit model format.
    """
    baseline = neural_action_features(
        state, action_kind, action_payload_json, dimension,
    )
    try:
        raw = json.loads(action_payload_json)
    except (TypeError, ValueError):
        raw = {}
    payload = raw if isinstance(raw, dict) else {}
    tokens: list[tuple[str, float]] = []
    if action_kind == "choose_map_node":
        selected_id = payload.get("NodeId", payload.get("node_id"))
        if isinstance(selected_id, str):
            tokens.extend(_future_room_tokens(selected_id, _nodes(state)))
    if action_kind in {"take_reward_card", "rest_upgrade", "remove_card"}:
        tokens.extend(_card_choice_tokens(state, action_kind, payload))
    hp, max_hp = state.get("hp"), state.get("max_hp")
    if (isinstance(hp, int | float) and not isinstance(hp, bool)
            and isinstance(max_hp, int | float) and not isinstance(max_hp, bool)
            and max_hp > 0):
        tokens.append(("macro:hp:fraction", min(1.0, max(0.0, hp / max_hp))))
    gold = state.get("gold")
    if isinstance(gold, int | float) and not isinstance(gold, bool):
        tokens.append(("macro:gold", math.tanh(max(0.0, gold) / 200.0)))
    if not tokens:
        return baseline
    # Sum separate normalized components; never encode arbitrary map node IDs
    # or instance IDs as tokens. Hash collisions are intrinsic to the v2 model.
    for idx, value in _hash_features(tokens, dimension).items():
        baseline[idx] = max(-8.0, min(8.0, baseline.get(idx, 0.0) + value))
    return baseline
