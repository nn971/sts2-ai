"""H7 opt-in, public-only, cross-phase object/action codec (v1).

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
MAX_ACTIONS = 128
# New v2 codec kind; the legacy KINDS vocabulary and checkpoints stay unchanged.
OWNER_ATTRIBUTE_KIND_V2 = len(KINDS)


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
    # Compositional subset selection; indexed into the public object tokens.
    members: tuple[int, ...] = ()


@dataclass(frozen=True)
class RelationToken:
    relation_type: int
    source: int
    target: int


@dataclass(frozen=True)
class EncodedDecision:
    schema: str
    phase: str
    objects: tuple[ObjectToken, ...]
    actions: tuple[ActionToken, ...]
    # Opt-in H20 typed graph edges; older codecs keep the empty default.
    relations: tuple[RelationToken, ...] = ()


def encode_public_decision(
    state: dict[str, Any], legal_actions: Sequence[Any], *,
    max_actions: int | None = MAX_ACTIONS,
    encode_card_subsets: bool = False,
    compress_deck: bool = False,
    owned_attributes: bool = False,
    object_limit: int = MAX_OBJECTS,
    schema_id: str = SHARED_OBJECT_SCHEMA,
    omit_affliction_source_reference: bool = False,
) -> EncodedDecision:
    """Encode only published game fields and currently legal action payloads.

    Missing public target pointers fail explicitly. This prevents an invisible
    target or alias from silently getting the same action logit.
    The historical H7 menu guard is retained by default; H16 may pass
    max_actions=None because its action tensors are dynamically sized.
    """
    if not isinstance(state, dict) or not legal_actions:
        raise ValueError("Need a public state and a nonempty legal action menu")
    combat = state.get("combat")
    phase = "combat" if isinstance(combat, dict) else "macro"
    objects: list[ObjectToken] = []
    lookup: dict[tuple[str, object], int] = {}
    # Keep the legacy object positions and action pointers stable. In v2,
    # append owner-linked attributes only after every base object is built.
    attribute_sources: list[tuple[int, str, str, dict[str, Any]]] = []

    def add(kind: str, identity: str, zone: str, record: dict[str, Any],
            *, owner: int = -1, ref: tuple[str, object] | None = None) -> int:
        if len(objects) >= object_limit:
            raise ValueError("Public object cap exceeded: do not drop referenced targets")
        numeric, observed = _numbers(record)
        index = len(objects)
        objects.append(ObjectToken(
            KINDS.index(kind), _bucket(identity, IDENTITY_BUCKETS),
            ZONES.index(zone), numeric, observed, owner,
        ))
        if owned_attributes and kind in ("state", "player", "card", "relic", "potion"):
            attribute_sources.append((index, kind, zone, record))
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
        if owned_attributes and isinstance(combat.get("history"), dict):
            player["history"] = combat["history"]
        player_idx = add("player", "player", "player", player)
        def selection_signature(card: dict[str, Any]) -> str:
            return json.dumps(
                {key: value for key, value in card.items()
                 if key != "instance_id"},
                sort_keys=True, separators=(",", ":"),
            )

        for zone, field in (
            ("hand", "hand"), ("discard", "discard_pile"),
            ("exhaust", "exhaust_pile"),
        ):
            pile = _records(combat.get(field))
            grouped: dict[str, tuple[dict[str, Any], list[int]]] = {}
            # Hand instances are always distinct: play-card actions refer to
            # their exact public instances, and their count is at most 10.
            if compress_deck and zone != "hand":
                for card in pile:
                    ref_id = card.get("instance_id")
                    if type(ref_id) is not int:
                        raise ValueError("Malformed public combat pile card")
                    signature = selection_signature(card)
                    if signature not in grouped:
                        grouped[signature] = (card, [])
                    grouped[signature][1].append(ref_id)
                represented = list(grouped.values())
            else:
                represented = [(card, [card["instance_id"]]
                                if type(card.get("instance_id")) is int else [])
                               for card in pile]
            for card, ids in represented:
                ref_id = card.get("instance_id")
                record = dict(card)
                if compress_deck and zone != "hand":
                    record["count"] = math.log1p(len(ids))
                idx = add("card", _entry_id(card, "card_id"), zone, record,
                          ref=("hand_card", ref_id) if zone == "hand" and
                          type(ref_id) is int else None)
                if encode_card_subsets:
                    for selected_id in ids:
                        key = ("selection_card", selected_id)
                        if key in lookup:
                            raise ValueError("Duplicate selectable public card instance")
                        lookup[key] = idx
        # Selection candidates can also originate from visible temporary
        # choice pools and draw-pile selection dialogs. Use one token per
        # identical *public* card class while preserving a pointer from every
        # candidate ID; no hidden draw order is exposed.
        if encode_card_subsets:
            unknown: dict[str, tuple[dict[str, Any], list[int]]] = {}
            for candidate in _records(combat.get("selection_candidates")):
                ref_id = candidate.get("instance_id")
                if type(ref_id) is not int:
                    raise ValueError("Malformed public choice candidate")
                key = ("selection_card", ref_id)
                if key in lookup:
                    continue
                signature = selection_signature(candidate) if compress_deck else str(ref_id)
                if signature not in unknown:
                    unknown[signature] = (candidate, [])
                unknown[signature][1].append(ref_id)
            for candidate, ids in unknown.values():
                record = dict(candidate)
                if compress_deck:
                    record["count"] = math.log1p(len(ids))
                idx = add(
                    "card", _entry_id(candidate, "card_id"), "reward", record,
                )
                for ref_id in ids:
                    lookup[("selection_card", ref_id)] = idx
        # Partial AI-only selection steps are public decisions. The already
        # selected cards remain observable and must enter the context, or the
        # policy would forget which card classes it has accumulated.
        if encode_card_subsets:
            previous = _records(combat.get("selection_selected_cards"))
            selected_groups: dict[str, tuple[dict[str, Any], int]] = {}
            for chosen in previous:
                sig = selection_signature(chosen) if compress_deck else str(
                    chosen.get("instance_id"))
                if sig in selected_groups:
                    card, count = selected_groups[sig]
                    selected_groups[sig] = (card, count + 1)
                else:
                    selected_groups[sig] = (chosen, 1)
            for chosen, count in selected_groups.values():
                record = dict(chosen)
                if compress_deck:
                    record["count"] = math.log1p(count)
                add("card", _entry_id(chosen, "card_id"), "event", record)
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

    # H16 may treat identical inventory copies as one multiset class.
    # Equality compares the entire public card record except the opaque
    # instance reference: afflictions, enchantments, upgrades, persistent
    # effects, and costs remain part of the signature. Preserve order of
    # first appearance. H7's original per-instance encoding is unchanged.
    deck = _records(state.get("deck"))
    if compress_deck:
        grouped: dict[str, tuple[dict[str, Any], int]] = {}
        for card in deck:
            public_card = {key: value for key, value in card.items()
                           if key != "instance_id"}
            signature = json.dumps(public_card, sort_keys=True, separators=(",", ":"))
            if signature in grouped:
                representative, n = grouped[signature]
                grouped[signature] = (representative, n + 1)
            else:
                grouped[signature] = (public_card, 1)
        for card, multiplicity in grouped.values():
            # log1p keeps the count feature informative even for 1,000+
            # clones rather than saturating tanh(count / 10).
            record = dict(card)
            record["count"] = math.log1p(multiplicity)
            add("card", _entry_id(card, "card_id"), "deck", record)
    else:
        for card in deck:
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
    if max_actions is not None and len(legal_actions) > max_actions:
        raise ValueError("Legal action cap exceeded: all legal actions must remain present")
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
        members: tuple[int, ...] = ()
        if encode_card_subsets and kind == "select_cards":
            raw_members = payload.get("CardInstanceIds", payload.get("card_instance_ids"))
            if (not isinstance(raw_members, list)
                    or any(type(item) is not int for item in raw_members)
                    or len(raw_members) != len(set(raw_members))):
                raise ValueError("Malformed select_cards CardInstanceIds")
            selected: list[int] = []
            for item in raw_members:
                pointer = lookup.get(("selection_card", item))
                if pointer is None:
                    raise ValueError(
                        "select_cards candidate missing from public hand/discard/exhaust: "
                        + str(item)
                    )
                selected.append(pointer)
            members = tuple(sorted(selected))
            # Stable across instance IDs/runs; the selected card embeddings
            # encode content while cardinality distinguishes empty vs 1 vs 2.
            semantic = _bucket(f"select_cards:count={len(members)}", IDENTITY_BUCKETS)
        action_tokens.append(ActionToken(
            _bucket(kind, ACTION_BUCKETS), semantic, source, target, members,
        ))
    if owned_attributes:
        _append_owned_public_attributes(
            objects, attribute_sources, object_limit=object_limit,
            omit_affliction_source_reference=omit_affliction_source_reference,
        )
    return EncodedDecision(schema_id, phase, tuple(objects), tuple(action_tokens))


# Typed H19 resource feature paths. We deliberately never hash an instance ID
# into a learned identity: owner is a within-observation object index only.
# The same attribute name is reused across different resource identities.
_V2_RESOURCE_FIELDS = {
    "state": ("potion_slot_capacity", "potion_slots_occupied", "potion_slots_empty"),
    "player": ("history",),
    "card": (
        "state", "enchantment", "affliction", "is_temporary",
        "combat_energy_cost_delta", "temporary_energy_cost",
        "keyword_overrides", "enchantment_triggered_this_combat",
        "cost_kind", "effective_energy_cost",
    ),
    "relic": ("state", "is_wax", "is_melted"),
    "potion": ("state",),
}


def _public_attribute_leaves(value: Any, path: str):
    """Walk already-public values with deterministic key/list ordering.

    Presence is represented by the existence of an attribute token. Type
    suffixes separate false/0/empty string/null and numeric/string values.
    """
    if isinstance(value, dict):
        if not value:
            yield path + ":empty_object", 0.0
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("Public resource state has non-string key")
            yield from _public_attribute_leaves(value[key], path + "." + key)
    elif isinstance(value, list):
        if not value:
            yield path + ":empty_list", 0.0
        for i, item in enumerate(value):
            yield from _public_attribute_leaves(item, path + f"[{i}]")
    elif value is None:
        yield path + ":null", 0.0
    elif type(value) is bool:
        yield path + ":bool", float(value)
    elif type(value) is int:
        yield path + ":integer", float(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite public resource attribute")
        yield path + ":float", value
    elif type(value) is str:
        yield path + ":string:" + value, 1.0
    else:
        raise ValueError(f"Unsupported public resource value type at {path}")


def _append_owned_public_attributes(
    objects: list[ObjectToken],
    sources: list[tuple[int, str, str, dict[str, Any]]],
    *, object_limit: int,
    omit_affliction_source_reference: bool = False,
) -> None:
    for owner, kind, zone, record in sources:
        for field in _V2_RESOURCE_FIELDS[kind]:
            # Missing stays distinct from an explicit zero/false/null.
            if field not in record:
                continue
            field_value = record[field]
            if (omit_affliction_source_reference and field == "affliction"
                    and isinstance(field_value, dict)):
                # An enemy instance ID is a within-frame reference only.
                # Keep it for card-class grouping and explicit graph joins,
                # while excluding it from all learned numeric/identity fields.
                field_value = {
                    key: value for key, value in field_value.items()
                    if key != "source_enemy_instance_id"
                }
            for identity, value in _public_attribute_leaves(
                field_value, field,
            ):
                if len(objects) >= object_limit:
                    raise ValueError(
                        "Public owner-linked attribute cap exceeded: "
                        "no resource or attribute may be silently dropped")
                numeric, observed = _numbers({"stacks": value})
                objects.append(ObjectToken(
                    kind=OWNER_ATTRIBUTE_KIND_V2,
                    identity=_bucket("attribute:" + identity, IDENTITY_BUCKETS),
                    zone=ZONES.index(zone),
                    numeric=numeric,
                    observed=observed,
                    owner=owner,
                ))
