"""Frozen H9 combat-only codec preserving the exact H7-era input representation.

This codec is deliberately independent from the old hashed v8/v13 and H6
encodings. Opaque instance IDs are used for within-frame joins, never embeddings.
An absent numeric feature has its own mask; a present zero has mask=1.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SHARED_OBJECT_SCHEMA = "sts2-h7-public-shared-objects-v1"
IDENTITY_BUCKETS = 4096
ACTION_BUCKETS = 2048
NUMERIC_FIELDS = (
    "hp", "max_hp", "block", "energy", "gold", "cost", "upgrade_level",
    "stacks", "intent_damage", "intent_hits", "intent_base_damage",
    "draw_pile_count", "turn", "floor", "count", "formation_position",
)
NUMERIC_SCALES = (100, 100, 50, 5, 500, 5, 4, 10, 50, 10, 50, 40, 10, 16, 10, 10)
KINDS = (
    "state", "player", "card", "enemy", "relic", "potion", "power", "status",
    "room", "event", "shop_offer", "event_choice",
)
ZONES = (
    "global", "hand", "deck", "discard", "exhaust", "enemy", "player",
    "inventory", "map", "shop", "reward", "event",
)
MAX_OBJECTS = 256
# Action menus are variable length. The model and collator support arbitrary
# legal-menu sizes; limiting them here silently excludes reachable decisions.


def _bucket(name: str, mod: int) -> int:
    if not name:
        return 0
    return 1 + int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") % (
        mod - 1
    )


def _entry_id(record: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _records(obj: object) -> list[dict[str, Any]]:
    return [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []


def _numbers(record: dict[str, Any]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    values: list[float] = []
    masks: list[int] = []
    for name, scale in zip(NUMERIC_FIELDS, NUMERIC_SCALES, strict=True):
        raw = record.get(name)
        if isinstance(raw, (float, int)) and not isinstance(raw, bool) and math.isfinite(raw):
            values.append(math.tanh(float(raw) / scale))
            masks.append(1)
        else:
            values.append(0.0)
            masks.append(0)
    return tuple(values), tuple(masks)


@dataclass(frozen=True)
class ObjectToken:
    kind: int
    identity: int
    zone: int
    numeric: tuple[float, ...]
    observed: tuple[int, ...]
    owner: int = -1


@dataclass(frozen=True)
class ActionToken:
    kind: int
    semantic: int
    source: int = -1
    target: int = -1


@dataclass(frozen=True)
class EncodedDecision:
    schema: str
    phase: str
    objects: tuple[ObjectToken, ...]
    actions: tuple[ActionToken, ...]


def encode_public_decision(
    state: dict[str, Any], legal_actions: Sequence[Any],
) -> EncodedDecision:
    """Encode only published game fields and currently legal action payloads.

    Missing public target pointers fail explicitly. This prevents an invisible
    target or alias from silently getting the same action logit.
    """
    if not isinstance(state, dict) or not legal_actions:
        raise ValueError("Need a public state and a nonempty legal action menu")
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Frozen H9 codec only accepts observable combat frames")
    phase = "combat"
    objects: list[ObjectToken] = []
    lookup: dict[tuple[str, object], int] = {}

    def add(kind: str, identity: str, zone: str, record: dict[str, Any],
            *, owner: int = -1, ref: tuple[str, object] | None = None) -> int:
        if len(objects) >= MAX_OBJECTS:
            raise ValueError("Public object cap exceeded: do not drop referenced targets")
        numeric, observed = _numbers(record)
        index = len(objects)
        objects.append(ObjectToken(
            KINDS.index(kind), _bucket(identity, IDENTITY_BUCKETS),
            ZONES.index(zone), numeric, observed, owner,
        ))
        if ref is not None:
            if ref in lookup:
                raise ValueError("Duplicate public object reference")
            lookup[ref] = index
        return index

    # The first position is the global state/CLS token (no position embeddings).
    add("state", "global", "global", state)
    if isinstance(combat, dict):
        player = dict(state)
        for key in ("energy", "turn", "draw_pile_count"):
            if key in combat:
                player[key] = combat[key]
        if "player_block" in combat:
            player["block"] = combat["player_block"]
        player_idx = add("player", "player", "player", player)
        for zone, field in (
            ("hand", "hand"), ("discard", "discard_pile"),
            ("exhaust", "exhaust_pile"),
        ):
            for card in _records(combat.get(field)):
                ref_id = card.get("instance_id")
                add("card", _entry_id(card, "card_id"), zone, card,
                    ref=("hand_card", ref_id) if zone == "hand" and
                    type(ref_id) is int else None)
        for enemy in _records(combat.get("enemies")):
            ref_id = enemy.get("instance_id")
            idx = add("enemy", _entry_id(enemy, "enemy_id"), "enemy", enemy,
                      ref=("enemy", ref_id) if type(ref_id) is int else None)
            for power in _records(enemy.get("powers")):
                add("power", _entry_id(power, "power_id"), "enemy", power, owner=idx)
            statuses = enemy.get("statuses")
            if isinstance(statuses, dict):
                for name, stacks in sorted(statuses.items()):
                    if isinstance(name, str):
                        add("status", name, "enemy", {"stacks": stacks}, owner=idx)
        for power in _records(combat.get("player_powers")):
            add("power", _entry_id(power, "power_id"), "player", power, owner=player_idx)

    for card in _records(state.get("deck")):
        add("card", _entry_id(card, "card_id"), "deck", card)
    for relic in state.get("relics", []) if isinstance(state.get("relics"), list) else []:
        record = {"relic_id": relic} if isinstance(relic, str) else relic
        if isinstance(record, dict):
            add("relic", _entry_id(record, "relic_id", "id"), "inventory", record)
    for potion in state.get("potions", []) if isinstance(state.get("potions"), list) else []:
        record = {"potion_id": potion} if isinstance(potion, str) else potion
        if isinstance(record, dict):
            slot = record.get("slot")
            add("potion", _entry_id(record, "potion_id", "id"), "inventory", record,
                ref=("potion_slot", slot) if type(slot) is int else None)

    # Room type, not raw map node ID, is the embedded identity. Map topology
    # and candidate connectivity require a later versioned relational schema.
    for room in _records(state.get("map")):
        add("room", _entry_id(room, "room_type"), "map", room)

    action_tokens: list[ActionToken] = []
    for action in legal_actions:
        kind = action.kind
        try:
            payload = json.loads(action.payload_json)
        except (ValueError, TypeError) as exc:
            raise ValueError("Malformed public action payload") from exc
        if not isinstance(kind, str) or not isinstance(payload, dict):
            raise ValueError("Malformed public legal action")
        source, target = -1, -1
        raw_card = payload.get("CardInstanceId", payload.get("card_instance_id"))
        raw_enemy = payload.get("TargetEnemyId", payload.get("target_enemy_id"))
        if raw_card is not None:
            source = lookup.get(("hand_card", raw_card), -1)
            if source < 0:
                raise ValueError("Legal card action has no visible hand target")
        if raw_enemy is not None:
            target = lookup.get(("enemy", raw_enemy), -1)
            if target < 0:
                raise ValueError("Legal action references an invisible enemy")
        if kind == "use_potion":
            slot = payload.get("Slot", payload.get("slot"))
            if slot is not None:
                source = lookup.get(("potion_slot", slot), -1)
                if source < 0:
                    raise ValueError("Legal potion slot has no visible potion")
        semantic = objects[source].identity if source >= 0 else 0
        action_tokens.append(ActionToken(_bucket(kind, ACTION_BUCKETS), semantic, source, target))
    return EncodedDecision(SHARED_OBJECT_SCHEMA, phase, tuple(objects), tuple(action_tokens))
