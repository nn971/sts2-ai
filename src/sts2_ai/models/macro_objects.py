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
from sts2_ai.models.hashed_linear import (
    _hash_features,
    neural_action_features,
    state_features,
)

MACRO_ATTENTION_FORMAT = "sts2-neural-policy-value-v17-macro-object-attention"
MACRO_SEMANTIC_FORMAT = "sts2-neural-policy-value-v18-macro-semantic-actions"


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


def _offer_from_action(
    state: dict[str, Any], kind: str, payload: dict[str, Any],
) -> dict[str, Any] | None:
    shop = state.get("shop")
    if not isinstance(shop, dict):
        return None
    key = (
        "card_offers" if kind == "buy_card" else
        "relic_offers" if kind == "buy_relic" else
        "potion_offers" if kind in ("buy_potion", "replace_shop_potion") else None
    )
    if key is None:
        return None
    offer_id = payload.get("OfferId", payload.get("offer_id"))
    listed = shop.get(key)
    for offer in listed if isinstance(listed, list) else []:
        if (isinstance(offer, dict) and offer.get("offer_id") == offer_id):
            return offer
    # Native observations expose both plural arrays and legacy singular
    # potion/relic aliases; card purchases have no singular fallback.
    singular_key = {
        "potion_offers": "potion_offer",
        "relic_offers": "relic_offer",
    }.get(key)
    singular = shop.get(singular_key) if singular_key else None
    if isinstance(singular, dict) and singular.get("offer_id") == offer_id:
        return singular
    return None


def _potion_in_slot(state: dict[str, Any], slot: object) -> str | None:
    """Native public potions contains only occupied slots, not slot-indexed entries."""
    if type(slot) is not int:
        return None
    potions = state.get("potions")
    if isinstance(potions, list):
        for potion in potions:
            if isinstance(potion, dict) and potion.get("slot") == slot:
                return _id(potion, "potion_id")
    return None


def _reward_action_tokens(
    state: dict[str, Any], kind: str, payload: dict[str, Any],
) -> list[tuple[str, float]]:
    reward = state.get("reward")
    if not isinstance(reward, dict):
        return []
    tokens: list[tuple[str, float]] = []
    index = payload.get("Index", payload.get("index"))
    group = payload.get("GroupIndex", payload.get("group_index"))
    if kind in ("take_reward_card", "take_reward_card_group",
                "skip_reward_card_group", "sacrifice_reward_card_group",
                "reroll_reward_card"):
        cards = reward.get("card_options")
        if type(group) is int:
            groups = reward.get("pending_card_groups")
            matched = next((
                item for item in groups
                if isinstance(item, dict) and item.get("group_index") == group
            ), None) if isinstance(groups, list) else None
            cards = matched.get("card_options") if isinstance(matched, dict) else None
        if isinstance(cards, list):
            if kind in ("take_reward_card", "take_reward_card_group"):
                if type(index) is int and 0 <= index < len(cards):
                    tokens.append((f"macro:reward:card:{cards[index]}", 1.0))
            else:
                # The offered set is visible even for a skip/reroll decision.
                for card_id in sorted(name for name in cards if isinstance(name, str)):
                    tokens.append((f"macro:reward:group-card:{card_id}", 1.0))
    elif kind in ("take_reward_relic", "take_reward_extra_relic",
                  "skip_reward_extra_relic"):
        if kind == "take_reward_relic":
            options = reward.get("relic_options")
            if isinstance(options, list) and type(index) is int and 0 <= index < len(options):
                relic_id = options[index]
            else:
                relic_id = reward.get("relic_option")
        else:
            relic_id = None
            pending = reward.get("pending_extra_relic_rewards")
            if isinstance(pending, list) and type(index) is int:
                for item in pending:
                    if isinstance(item, dict) and item.get("group_index") == index:
                        relic_id = _id(item, "relic_id")
                        if type(item.get("is_wax")) is bool:
                            tokens.append(("macro:reward:relic:is_wax", float(item["is_wax"])))
                        break
        if isinstance(relic_id, str):
            tokens.append((f"macro:reward:relic:{relic_id}", 1.0))
    elif kind in ("take_reward_extra_potion", "replace_reward_extra_potion",
                  "skip_reward_extra_potion"):
        pending = reward.get("pending_extra_potion_rewards")
        if type(group) is int and isinstance(pending, list):
            for item in pending:
                if isinstance(item, dict) and item.get("group_index") == group:
                    name = _id(item, "potion_id")
                    if name is not None:
                        tokens.append((f"macro:reward:potion:{name}", 1.0))
                    break
    elif kind == "take_reward_extra_gold":
        pending = reward.get("pending_extra_gold_rewards")
        if type(group) is int and isinstance(pending, list):
            for item in pending:
                if isinstance(item, dict) and item.get("group_index") == group:
                    amount = _number(item.get("amount"), 100.0)
                    if amount is not None:
                        tokens.append(("macro:reward:gold", amount))
                    break
    elif kind == "take_reward_bundle":
        # The pinned public reward schema does not publish bundle contents.
        # Index is an explicit player-visible selector, never hidden content.
        if type(index) is int:
            tokens.append((f"macro:reward:bundle-index:{index}", 1.0))
    if kind in ("replace_reward_potion", "replace_reward_extra_potion"):
        slot = payload.get("Slot", payload.get("slot"))
        outgoing = _potion_in_slot(state, slot)
        if outgoing is not None:
            tokens.append((f"macro:reward:replace-potion:{outgoing}", 1.0))
    return tokens


def macro_semantic_action_features(
    state: dict[str, Any], action_kind: str, payload_json: str, dimension: int,
) -> dict[int, float]:
    """V18: identify *publicly visible* action choices instead of offer slots.

    V17 checkpoints always use macro_attention_action_features unchanged.
    A malformed/unresolvable offer receives an explicit unknown identity,
    never a raw ephemeral OfferId masquerading as an intrinsic item feature.
    """
    tokens: list[tuple[str, float]] = [
        ("neural-action:bias", 1.0),
        (f"neural-action:kind={action_kind}", 1.0),
    ]
    base = neural_action_features(state, action_kind, payload_json, dimension)
    payload = _payload(payload_json)
    if action_kind in ("buy_card", "buy_potion", "buy_relic", "replace_shop_potion"):
        offer = _offer_from_action(state, action_kind, payload)
        if offer is None:
            tokens.append(("macro:offer:unknown", 1.0))
        else:
            name = _id(offer, "item_id", "content_id", "card_id", "relic_id", "potion_id")
            tokens.append((f"macro:offer:item:{name or 'unknown'}", 1.0))
            for price_field in ("price", "base_price"):
                price = _number(offer.get(price_field), 100.0)
                if price is not None:
                    tokens.append((f"macro:offer:{price_field}", price))
            if action_kind == "buy_card":
                shop = state.get("shop")
                previews = shop.get("card_offer_previews") if isinstance(shop, dict) else None
                offer_id = payload.get("OfferId", payload.get("offer_id"))
                for preview in previews if isinstance(previews, list) else []:
                    if not isinstance(preview, dict) or preview.get("offer_id") != offer_id:
                        continue
                    card = preview.get("card")
                    if isinstance(card, dict):
                        card_id = _id(card, "card_id")
                        if card_id is not None:
                            tokens.append((f"macro:offer:preview:{card_id}", 1.0))
                        level = _number(card.get("upgrade_level"), 4.0)
                        if level is not None:
                            tokens.append(("macro:offer:preview:upgrade_level", level))
                        enchantment = card.get("enchantment")
                        if isinstance(enchantment, dict):
                            for field, value in sorted(enchantment.items()):
                                if isinstance(value, str):
                                    tokens.append((
                                        f"macro:offer:preview:enchant:{field}:{value}", 1.0,
                                    ))
                                else:
                                    amount = _number(value, 10.0)
                                    if amount is not None:
                                        tokens.append((
                                            f"macro:offer:preview:enchant:{field}", amount,
                                        ))
                    break
            for field in ("on_sale", "sold"):
                if type(offer.get(field)) is bool:
                    tokens.append((f"macro:offer:{field}", float(offer[field])))
        if action_kind == "replace_shop_potion":
            tokens.append(("macro:potion:replace", 1.0))
            slot = payload.get("Slot", payload.get("SlotIndex", payload.get("slot")))
            # A slot is an actual visible potion resource, unlike an ephemeral offer ID.
            potion_id = _potion_in_slot(state, slot)
            if potion_id is not None:
                tokens.append((f"macro:potion:replace:{potion_id}", 1.0))
    elif action_kind == "event_choice":
        event = state.get("event")
        choice = payload.get("ChoiceId", payload.get("choice_id"))
        event_id = state.get("event_id")
        if isinstance(event, dict):
            event_id = event.get("event_id", event_id)
        tokens.append((f"macro:event:{event_id}:choice:{choice}", 1.0))
        if isinstance(event, dict):
            values = event.get("visible_choice_values")
            for item in values if isinstance(values, list) else []:
                if isinstance(item, dict) and item.get("choice_id") == choice:
                    for field, value in sorted(item.items()):
                        if field == "choice_id" or value is None:
                            continue
                        numeric = _number(value, 100.0)
                        if numeric is not None:
                            tokens.append((f"macro:event:value:{field}", numeric))
                        elif isinstance(value, str):
                            tokens.append((f"macro:event:value:{field}:{value}", 1.0))
                    break
    elif action_kind == "replace_event_potion":
        event = state.get("event")
        replacement = (
            event.get("pending_potion_replacement") if isinstance(event, dict) else None
        )
        if isinstance(replacement, dict):
            incoming = _id(replacement, "potion_id")
            if incoming is not None:
                tokens.append((f"macro:event:incoming-potion:{incoming}", 1.0))
        slot = payload.get("Slot", payload.get("slot"))
        outgoing = _potion_in_slot(state, slot)
        if outgoing is not None:
            tokens.append((f"macro:event:outgoing-potion:{outgoing}", 1.0))
    elif action_kind in (
        "choose_event_deck_card", "choose_shop_relic_deck_card",
        "choose_relic_deck_card",
    ):
        instance = payload.get("CardInstanceId", payload.get("card_instance_id"))
        _, ids = _cards(state)
        key = ids.get(instance) if type(instance) is int else None
        tokens.append((f"macro:deck-choice:{key or 'unknown'}", 1.0))
    elif action_kind in (
        "take_reward_card", "take_reward_card_group", "skip_reward_card_group",
        "sacrifice_reward_card_group", "reroll_reward_card",
        "take_reward_relic", "take_reward_extra_relic", "skip_reward_extra_relic",
        "replace_reward_potion", "take_reward_extra_potion",
        "replace_reward_extra_potion", "skip_reward_extra_potion",
        "take_reward_extra_gold", "take_reward_bundle",
    ):
        tokens.extend(_reward_action_tokens(state, action_kind, payload))
    # Preserve V17 signal without modifying any V17 representation or its checkpoints.
    enriched = _hash_features(tokens, dimension)
    # Hashed feature merging deliberately sums collisions; identity tokens
    # should change the representation even for same-kind alternatives.
    return {key: base.get(key, 0.0) + enriched.get(key, 0.0)
            for key in base.keys() | enriched.keys()}



def macro_global_features(state: dict[str, Any], dimension: int) -> dict[int, float]:
    """Global public summary without raw graph identity/edge-label tokens.

    Map topology is represented in candidate-node embeddings, not flattened
    next_node_ids (which are ephemeral source IDs and can spuriously overfit).
    """
    return state_features({key: value for key, value in state.items()
                           if key != "map"}, dimension)
