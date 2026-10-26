"""Versioned public semantic identities for H6-C macro legal actions.

The v17 encoder maps all shop cards, shop potions and event choices of one
kind to identical vectors. This v18 replacement resolves OFFER and CHOICE
identities using only the player-facing observation. Temporary offer IDs
and card-instance IDs are used only as lookup keys, never model tokens.
"""
from __future__ import annotations

import json
import math
from typing import Any

from sts2_ai.models.hashed_linear import _hash_features, semantic_action_label

MACRO_CHOICE_FORMAT = "sts2-neural-policy-value-v18-macro-public-choice-identity"


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _payload(text: str) -> dict[str, Any]:
    try:
        return _dict(json.loads(text))
    except (ValueError, TypeError):
        return {}


def _field(obj: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in obj:
            return obj[name]
    return None


def _shop_offer(state: dict[str, Any], kind: str,
                payload: dict[str, Any]) -> dict[str, Any] | None:
    offer_id = _field(payload, "OfferId", "offer_id")
    if type(offer_id) is not int:
        return None
    shop = _dict(state.get("shop"))
    names = {
        "buy_card": ("card_offers",),
        "buy_potion": ("potion_offers", "potion_offer"),
        "buy_relic": ("relic_offers", "relic_offer"),
    }.get(kind, ())
    for name in names:
        raw = shop.get(name)
        offers = raw if isinstance(raw, list) else [raw]
        for item in offers:
            if not isinstance(item, dict):
                continue
            if _field(item, "offer_id", "OfferId") == offer_id:
                return item
    return None


def _visible_choice(state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    choice_id = _field(payload, "ChoiceId", "choice_id")
    event = _dict(state.get("event"))
    raw = event.get("visible_choice_values")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and _field(item, "choice_id", "ChoiceId") == choice_id:
                return item
    return None


def choice_tokens(state: dict[str, Any], kind: str,
                  payload_json: str) -> list[tuple[str, float]]:
    """Stable public choice identity and known numeric consequences."""
    payload = _payload(payload_json)
    tokens: list[tuple[str, float]] = [
        ("macro-action:bias", 1.0),
        (f"macro-action:kind:{kind}", 1.0),
        (f"macro-action:old:{semantic_action_label(state, kind, payload_json)}", 1.0),
    ]
    if kind in ("buy_card", "buy_potion", "buy_relic"):
        offer = _shop_offer(state, kind, payload)
        if offer is None:
            tokens.append(("macro-action:shop:offer-unavailable", 1.0))
        else:
            item = _field(offer, "item_id", "ItemId")
            if isinstance(item, str):
                tokens.append((f"macro-action:shop:{kind}:item:{item}", 1.0))
            price = _field(offer, "price", "Price")
            if type(price) in (int, float) and math.isfinite(price):
                tokens.append(("macro-action:shop:price", math.tanh(price / 150.0)))
                tokens.append((f"macro-action:shop:price-band:{int(price // 25)}", 1.0))
            # A sold/unsupported offer is never assumed to be available.
            sale = _field(offer, "on_sale", "OnSale")
            if isinstance(sale, bool):
                tokens.append(("macro-action:shop:on-sale", float(sale)))
    elif kind == "event_choice":
        event = _dict(state.get("event"))
        event_id = _field(event, "event_id", "EventId", "event_id")
        choice = _field(payload, "ChoiceId", "choice_id")
        if not isinstance(event_id, str):
            event_id = state.get("event_id")
        if isinstance(event_id, str):
            tokens.append((f"macro-action:event:{event_id}", 1.0))
        if isinstance(choice, str):
            tokens.append((f"macro-action:event:choice:{choice}", 1.0))
        else:
            tokens.append(("macro-action:event:unknown-choice", 1.0))
        visible = _visible_choice(state, payload)
        if visible is not None:
            for key in ("gold_gain", "gold_cost", "hp_loss", "heal", "max_hp_delta"):
                value = visible.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    tokens.append((f"macro-action:event:known:{key}",
                                   math.tanh(float(value) / 50.0)))
            for key in ("guaranteed_card_id", "guaranteed_relic_id",
                        "guaranteed_potion_id"):
                value = visible.get(key)
                if isinstance(value, str):
                    tokens.append((f"macro-action:event:known:{key}:{value}", 1.0))
    elif kind in ("take_reward_relic", "take_reward_potion",
                  "take_reward_extra_relic", "take_reward_extra_potion"):
        reward = _dict(state.get("reward"))
        names = {"take_reward_relic": "relic_option",
                 "take_reward_potion": "potion_option"}
        name = reward.get(names[kind]) if kind in names else None
        if isinstance(name, str):
            tokens.append((f"macro-action:reward:{kind}:{name}", 1.0))
        for source in ("Index", "index", "GroupIndex", "group_index"):
            if source in payload and type(payload[source]) is int:
                value = payload[source]
                # Only used to join a visible option, never a physical instance.
                tokens.append((f"macro-action:reward:{kind}:group:{value}", 1.0))
    return tokens


def macro_choice_action_features(state: dict[str, Any], kind: str,
                                 payload_json: str, dimension: int) -> dict[int, float]:
    return _hash_features(choice_tokens(state, kind, payload_json), dimension)


def public_choice_label(state: dict[str, Any], kind: str, payload_json: str) -> str:
    payload = _payload(payload_json)
    if kind in ("buy_card", "buy_potion", "buy_relic"):
        offer = _shop_offer(state, kind, payload)
        if offer is not None:
            name = _field(offer, "item_id", "ItemId")
            price = _field(offer, "price", "Price")
            return f"{kind.replace('_', ' ')}: {name} ({price} gold)"
    if kind == "event_choice":
        event = _dict(state.get("event"))
        event_id = _field(event, "event_id", "EventId") or state.get("event_id")
        choice = _field(payload, "ChoiceId", "choice_id")
        known = _visible_choice(state, payload)
        effects = []
        for key in ("gold_gain", "gold_cost", "hp_loss", "heal", "max_hp_delta"):
            if isinstance(known, dict) and known.get(key) is not None:
                effects.append(f"{key}={known[key]}")
        effect = f" [{', '.join(effects)}]" if effects else ""
        return f"Event {event_id}: {choice}{effect}"
    if kind.startswith("skip_") or kind in ("leave_shop", "leave_event"):
        return kind.replace("_", " ")
    return semantic_action_label(state, kind, payload_json).replace("_", " ")
