"""Verify that H21 action-token aliases represent identical public choices.

Some legal menus distinguish opaque instance IDs for observationally identical
deck cards. Such copies may deliberately share a neural action. Other aliases
are errors: an item, price, target, modifier, or choice must not vanish.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sts2_ai.models.shared_game_objects import ActionToken, EncodedDecision

_SINGLE_CARD_KINDS = frozenset({
    "choose_event_deck_card", "choose_relic_deck_card",
    "choose_shop_relic_deck_card", "remove_card", "rest_upgrade",
})


@dataclass(frozen=True)
class ActionAliasAudit:
    total_actions: int
    canonical_choices: int
    equivalent_aliases: int


def _reference_value(payload: dict[str, Any], name: str) -> Any:
    snake = "".join(
        ("_" + c.lower() if c.isupper() else c)
        for c in name
    ).lstrip("_")
    if name in payload:
        return payload[name]
    if snake in payload:
        return payload[snake]
    raise ValueError("Missing expected public action reference " + name)


def _card_signature(card: dict[str, Any]) -> str:
    return json.dumps(
        {key: value for key, value in card.items() if key != "instance_id"},
        sort_keys=True, separators=(",", ":"),
    )


def _equivalence_key(state: dict[str, Any], action: Any) -> str:
    """Canonicalize only verified same-class deck-card choices."""
    try:
        payload = json.loads(action.payload_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed aliased legal action payload") from exc
    if not isinstance(payload, dict):
        raise ValueError("Aliased legal action must have an object payload")
    combat = state.get("combat")
    if isinstance(combat, dict) and action.kind != "select_cards":
        raise ValueError("H21 aliased a non-selection combat action")
    sources = [state.get("deck")]
    if isinstance(combat, dict):
        sources = [
            combat.get(zone) or []
            for zone in ("hand", "discard_pile", "exhaust_pile",
                         "selection_candidates")
        ]
    by_id = {}
    for cards in sources:
        if not isinstance(cards, list):
            raise ValueError("Malformed public card pool for alias audit")
        for card in cards:
            if not isinstance(card, dict) or type(card.get("instance_id")) is not int:
                raise ValueError("Malformed public card for alias audit")
            signature = _card_signature(card)
            old = by_id.get(card["instance_id"])
            if old is not None and old != signature:
                raise ValueError("Conflicting public card reference in choice pools")
            by_id[card["instance_id"]] = signature

    if action.kind in _SINGLE_CARD_KINDS:
        if set(payload) not in ({"CardInstanceId"}, {"card_instance_id"}):
            raise ValueError("Unaudited extra payload field on aliased card action")
        key = _reference_value(payload, "CardInstanceId")
        if type(key) is not int or key not in by_id:
            raise ValueError("Aliased action refers to unknown public deck card")
        return action.kind + ":" + by_id[key]

    if set(payload) not in ({"CardInstanceIds"}, {"card_instance_ids"}):
        raise ValueError(
            "Distinct public actions have identical tokens without "
            "an audited card-class equivalence")
    ids = _reference_value(payload, "CardInstanceIds")
    if (not isinstance(ids, list) or any(type(i) is not int for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Malformed aliased public card selection")
    if any(i not in by_id for i in ids):
        raise ValueError("Aliased selection references unknown public card")
    signatures = sorted(by_id[i] for i in ids)
    return json.dumps([action.kind, signatures], separators=(",", ":"))


def audit_h21_action_aliases(
    state: dict[str, Any],
    actions: Sequence[Any],
    encoded: EncodedDecision,
) -> ActionAliasAudit:
    if len(actions) != len(encoded.actions):
        raise ValueError("H21 changed legal action count")
    grouped: dict[ActionToken, list[Any]] = defaultdict(list)
    for action, token in zip(actions, encoded.actions, strict=True):
        grouped[token].append(action)
    aliases = 0
    for token, group in grouped.items():
        if len(group) == 1:
            continue
        expected = _equivalence_key(state, group[0])
        for item in group[1:]:
            if _equivalence_key(state, item) != expected:
                raise ValueError(
                    "Distinct public legal choices share H21 tokens")
        aliases += len(group) - 1
    return ActionAliasAudit(
        total_actions=len(actions),
        canonical_choices=len(grouped),
        equivalent_aliases=aliases,
    )
