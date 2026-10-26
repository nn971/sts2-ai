"""Public-only typed objects for H6-C macro attention.

Raw identifiers join a legal action to a *visible* object. They never enter
feature tokens. No future encounters, RNG values, hidden map information or
card-instance addresses are inferred. Unsupported public fields are omitted.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from sts2_ai.emulator import LegalAction
from sts2_ai.models.hashed_linear import _hash_features, neural_action_features

MACRO_ATTENTION_FORMAT = "sts2-neural-policy-value-v17-macro-object-attention"


def _id(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        entry = value.get(key)
        if isinstance(entry, str):
            return entry
    return None


def _payload(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: object, scale: float) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return math.tanh(number / scale)
    return None


def _nodes(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = state.get("map")
    if not isinstance(raw, list):
        return {}
    return {
        node["node_id"]: node for node in raw
        if isinstance(node, dict) and isinstance(node.get("node_id"), str)
    }


def _reachable_tokens(node_id: str, nodes: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    """Four publicly visible graph layers, without guessed route probabilities."""
    if node_id not in nodes:
        return []
    tokens: list[tuple[str, float]] = []
    current = {node_id}
    visited = {node_id}
    for depth in range(1, 5):
        children: set[str] = set()
        for parent in current:
            outgoing = nodes[parent].get("next_node_ids")
            if isinstance(outgoing, list):
                children.update(
                    item for item in outgoing
                    if isinstance(item, str) and item in nodes and item not in visited
                )
        if not children:
            break
        if depth == 1:
            tokens.append(("map:outdegree", min(len(children) / 4, 2.0)))
        counts = Counter(str(nodes[item].get("room_type", "unknown")) for item in children)
        for name, count in sorted(counts.items()):
            tokens.append((f"map:depth{depth}:room:{name}", count / len(children)))
        visited.update(children)
        current = children
    return tokens


def _typed_tokens(kind: str, identity: str) -> list[tuple[str, float]]:
    return [("macro:object:bias", 1.0),
            (f"macro:object:kind:{kind}", 1.0),
            (f"macro:object:{kind}:{identity}", 1.0)]


def _cards(state: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    raw = state.get("deck")
    if not isinstance(raw, list):
        return {}, {}
    groups: dict[str, dict[str, Any]] = {}
    instances: dict[int, str] = {}
    for card in raw:
        if not isinstance(card, dict):
            continue
        name = _id(card, "card_id")
        if name is None:
            continue
        upgrades = card.get("upgrades", card.get("upgrade_level", 0))
        upgrades = upgrades if type(upgrades) is int else 0
        key = f"deck:{name}:u{upgrades}"
        if key not in groups:
            groups[key] = {"card_id": name, "upgrades": upgrades, "count": 0}
        groups[key]["count"] += 1
        instance = card.get("instance_id")
        if type(instance) is int:
            instances[instance] = key
    return groups, instances


def macro_object_vectors(
    state: dict[str, Any], actions: Sequence[LegalAction], dimension: int,
) -> dict[str, dict[int, float]]:
    """Stable keyed typed-object set; cap unrelated objects, never legal targets."""
    vectors: dict[str, dict[int, float]] = {}
    cards, _ = _cards(state)
    for key, card in sorted(cards.items())[:48]:
        tokens = _typed_tokens("card", card["card_id"])
        tokens += [("card:upgrades", min(card["upgrades"] / 4, 2.0)),
                   ("card:copies", min(card["count"] / 5, 2.0))]
        vectors[key] = _hash_features(tokens, dimension)

    for kind, plural, names, limit in (
        ("relic", "relics", ("relic_id", "id", "model_id"), 32),
        ("potion", "potions", ("potion_id", "id", "model_id"), 8),
    ):
        raw = state.get(plural)
        if not isinstance(raw, list):
            continue
        counts: Counter[str] = Counter()
        for item in raw:
            name = item if isinstance(item, str) else (
                _id(item, *names) if isinstance(item, dict) else None
            )
            if name is not None:
                counts[name] += 1
        for name in sorted(counts)[:limit]:
            tokens = _typed_tokens(kind, name)
            tokens.append((f"{kind}:copies", min(counts[name] / 5.0, 2.0)))
            vectors[f"{kind}:{name}"] = _hash_features(tokens, dimension)

    nodes = _nodes(state)
    # Encode *candidate* nodes rather than all map nodes. Their public future
    # connectivity is summarized without leaking hidden RNG or card outcomes.
    selected_ids = sorted({
        node_id
        for a in actions if a.kind == "choose_map_node"
        if (node_id := _id(_payload(a.payload_json), "NodeId", "node_id")) is not None
    })
    for node_id in selected_ids:
        node = nodes.get(node_id)
        if node is None:
            continue
        room = str(node.get("room_type", "unknown"))
        tokens = _typed_tokens("room", room)
        for field in ("floor", "row", "column", "x", "y"):
            numeric = _number(node.get(field), 16.0)
            if numeric is not None:
                tokens.append((f"map:{field}", numeric))
        tokens += _reachable_tokens(node_id, nodes)
        vectors[f"map:{node_id}"] = _hash_features(tokens, dimension)

    reward = state.get("reward")
    if isinstance(reward, dict) and isinstance(reward.get("card_options"), list):
        for index, name in enumerate(reward["card_options"]):
            if isinstance(name, str):
                # Index is a lookup key only; card identity is the feature.
                vectors[f"reward:{index}"] = _hash_features(
                    _typed_tokens("reward-card", name), dimension
                )
    return vectors


def macro_action_target(
    state: dict[str, Any], action: LegalAction,
    objects: dict[str, dict[int, float]],
) -> str | None:
    """Resolve a selected public object, or None for genuinely untargeted actions."""
    payload = _payload(action.payload_json)
    key: str | None = None
    if action.kind == "choose_map_node":
        node_id = _id(payload, "NodeId", "node_id")
        key = f"map:{node_id}" if node_id is not None else None
    elif action.kind == "take_reward_card":
        index = payload.get("Index", payload.get("index"))
        key = f"reward:{index}" if type(index) is int else None
    elif action.kind in ("rest_upgrade", "remove_card"):
        instance = payload.get("CardInstanceId", payload.get("card_instance_id"))
        _, instances = _cards(state)
        key = instances.get(instance) if type(instance) is int else None
    return key if key in objects else None


def macro_attention_action_features(
    state: dict[str, Any], action_kind: str, payload_json: str, dimension: int,
) -> dict[int, float]:
    """Action type and semantic content, never temporary instance/node IDs."""
    # The legacy action representation helps preserve an interpretable
    # warm-start, while object pointers carry graph/card distinctions.
    return neural_action_features(state, action_kind, payload_json, dimension)
