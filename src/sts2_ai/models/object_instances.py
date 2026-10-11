"""v15a: public-only, instance-preserving combat object extraction.

A record's ref is a lookup key, never a neural feature. The codec never
reads exact emulator states or private draw ordering. It deliberately does
not merge duplicate cards, potions or same-species enemies.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

OBJECT_SCHEMA = "sts2-public-combat-object-instances-v15a"
CARD_ZONES = ("hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile")
# This list is deliberately auditable. Do not recursively tokenize the full observation.
CARD_DYNAMIC_FIELDS = (
    "state", "affliction", "enchantment", "temporary_energy_cost",
    "enchantment_triggered_this_combat", "cost", "energy_cost",
)


@dataclass(frozen=True, slots=True)
class ObjectRef:
    kind: str
    key: int | str | tuple[str, str]


@dataclass(frozen=True, slots=True)
class ObjectToken:
    kind: str
    type_id: str
    role: str
    ref: ObjectRef
    owner: ObjectRef | None
    numeric: tuple[tuple[str, float], ...]
    categorical: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObjectSet:
    schema: str
    tokens: tuple[ObjectToken, ...]

    def index(self) -> dict[ObjectRef, int]:
        result: dict[ObjectRef, int] = {}
        for i, token in enumerate(self.tokens):
            if token.ref in result:
                raise ValueError(f"Duplicate object reference: {token.ref}")
            result[token.ref] = i
        return result


def _records(obj: object, path: str) -> list[dict[str, Any]]:
    if not isinstance(obj, list) or any(not isinstance(x, dict) for x in obj):
        raise ValueError(f"Expected array of records at {path}")
    return obj


def _identifier(record: dict[str, Any], field: str, path: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {path}.{field}")
    return value


def _instance(record: dict[str, Any], path: str) -> int:
    value = record.get("instance_id")
    if type(value) is not int or value <= 0:
        raise ValueError(f"Invalid {path}.instance_id")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric public field: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite public field: {field}")
    return result


def _flatten(
    prefix: str, value: object, numerical: dict[str, float], categorical: list[str],
    *, depth: int = 0,
) -> None:
    """Flatten only allowlisted public item metadata, without ephemeral IDs."""
    if depth > 8:
        raise ValueError(f"Public object metadata exceeds nesting limit: {prefix}")
    if value is None:
        categorical.append(prefix + "=null")
    elif isinstance(value, bool):
        categorical.append(f"{prefix}={str(value).lower()}")
    elif isinstance(value, str):
        categorical.append(f"{prefix}={value}")
    elif isinstance(value, (int, float)):
        numerical[prefix] = _number(value, prefix)
    elif isinstance(value, dict):
        for k, v in sorted(value.items()):
            if not isinstance(k, str):
                raise ValueError(f"Non-string object metadata key: {prefix}")
            # Never represent arbitrary identity counters as learned values.
            if k in ("instance_id", "persistent_card_instance_id"):
                continue
            _flatten(f"{prefix}.{k}", v, numerical, categorical, depth=depth + 1)
    elif isinstance(value, list):
        numerical[prefix + ".length"] = float(len(value))
        # An observed list can encode ordered trigger counters or unordered tags.
        for j, v in enumerate(value):
            _flatten(f"{prefix}[{j}]", v, numerical, categorical, depth=depth + 1)
    else:
        raise ValueError(f"Unsupported public object metadata at {prefix}")


def _token(
    kind: str, type_id: str, role: str, ref: ObjectRef,
    numeric: dict[str, float], categorical: list[str],
    owner: ObjectRef | None = None,
) -> ObjectToken:
    return ObjectToken(
        kind, type_id, role, ref, owner,
        tuple(sorted(numeric.items())), tuple(sorted(categorical)),
    )


def extract_combat_objects(state: dict[str, Any]) -> ObjectSet:
    """Extract the visible object set; never enumerate private combat.Cards.

    Persistent `deck` is intentionally not a second copy of live combat cards.
    Combat-local card IDs are unique across exposed zones. The exact order of
    `draw_pile` is never used as an input feature.
    """
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("v15 object encoder requires a public combat observation")
    objects: list[ObjectToken] = []
    global_num: dict[str, float] = {}
    global_cat: list[str] = []
    for k in ("hp", "max_hp", "act", "floor", "gold"):
        if k in state and state[k] is not None:
            global_num[k] = _number(state[k], k)
    for k in ("turn", "energy", "player_block", "draw_pile_count"):
        if k in combat and combat[k] is not None:
            global_num[k] = _number(combat[k], k)
    if isinstance(state.get("character_id"), str):
        global_cat.append("character=" + state["character_id"])
    objects.append(_token(
        "global", "player", "combat", ObjectRef("global", "player"),
        global_num, global_cat,
    ))

    seen_cards: set[int] = set()
    for zone in CARD_ZONES:
        cards = _records(combat.get(zone), "combat." + zone)
        for card in cards:
            instance = _instance(card, "card")
            if instance in seen_cards:
                raise ValueError(f"Card instance {instance} appears in multiple combat zones")
            seen_cards.add(instance)
            ident = _identifier(card, "card_id", "card")
            num: dict[str, float] = {}
            cat: list[str] = []
            for key in ("upgrade_level", "is_temporary"):
                if key in card:
                    _flatten(key, card[key], num, cat)
            for key in CARD_DYNAMIC_FIELDS:
                if key in card:
                    _flatten(key, card[key], num, cat)
                # Absence is NOT equivalent to an observed null enchantment.
            objects.append(_token("card", ident, zone, ObjectRef("card", instance), num, cat))
    if "draw_pile_count" in combat:
        count = combat["draw_pile_count"]
        if type(count) is not int or count != len(combat["draw_pile"]):
            raise ValueError("Public draw_pile count does not match visible card instances")

    relic_counters: dict[str, object] = {}
    for row in _records(combat.get("relic_counters"), "combat.relic_counters"):
        key = _identifier(row, "relic_id", "relic_counter")
        if key in relic_counters:
            raise ValueError("Duplicate relic counter")
        relic_counters[key] = row.get("trigger_counts")
    for relic in _records(state.get("relics"), "relics"):
        ident = _identifier(relic, "relic_id", "relic")
        num: dict[str, float] = {}
        cat: list[str] = []
        if "state" in relic:
            _flatten("state", relic["state"], num, cat)
        if ident in relic_counters:
            _flatten("trigger_counts", relic_counters.pop(ident), num, cat)
        objects.append(_token("relic", ident, "owned", ObjectRef("relic", ident), num, cat))
    if relic_counters:
        raise ValueError("Relic counter references an unowned relic")

    for potion in _records(state.get("potions"), "potions"):
        slot = potion.get("slot")
        if type(slot) is not int or slot < 0:
            raise ValueError("Potion has invalid public slot")
        ident = _identifier(potion, "potion_id", "potion")
        num: dict[str, float] = {}
        cat: list[str] = []
        if "state" in potion:
            _flatten("state", potion["state"], num, cat)
        objects.append(_token("potion", ident, "held", ObjectRef("potion", slot), num, cat))

    player = ObjectRef("global", "player")
    def append_effects(owner: ObjectRef, powers: object, statuses: object = None) -> None:
        for power in _records(powers, "powers"):
            ident = _identifier(power, "power_id", "power")
            num = {"stacks": _number(power.get("stacks"), "power.stacks")}
            objects.append(_token(
                "power", ident, "active", ObjectRef("power", (str(owner), ident)),
                num, [], owner,
            ))
        if statuses is not None:
            if not isinstance(statuses, dict):
                raise ValueError("Enemy statuses must be a mapping")
            for ident, value in sorted(statuses.items()):
                if not isinstance(ident, str) or not ident:
                    raise ValueError("Invalid status identity")
                objects.append(_token(
                    "status", ident, "active", ObjectRef("status", (str(owner), ident)),
                    {"stacks": _number(value, "status.stacks")}, [], owner,
                ))
    append_effects(player, combat.get("player_powers"))

    for enemy in _records(combat.get("enemies"), "combat.enemies"):
        instance = _instance(enemy, "enemy")
        ident = _identifier(enemy, "enemy_id", "enemy")
        num: dict[str, float] = {}
        cat: list[str] = []
        for k in (
            "hp", "block", "formation_position",
            "intent_damage", "intent_hits", "intent_base_damage",
        ):
            if k in enemy and enemy[k] is not None:
                num[k] = _number(enemy[k], "enemy." + k)
        for k in ("move_id", "last_move_id"):
            if k in enemy:
                _flatten(k, enemy[k], num, cat)
        ref = ObjectRef("enemy", instance)
        objects.append(_token("enemy", ident, "living", ref, num, cat))
        append_effects(ref, enemy.get("powers"), enemy.get("statuses"))

    return ObjectSet(OBJECT_SCHEMA, tuple(objects))


def legal_action_object_refs(
    action_kind: str, payload_json: str, objects: ObjectSet,
) -> tuple[ObjectRef, ...]:
    """Resolve object pointers for legal actions, never numeric ID embeddings."""
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed legal action JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Malformed legal action payload")
    refs: list[ObjectRef] = []
    fields = (
        ("CardInstanceId", "card_instance_id", "card"),
        ("TargetEnemyId", "target_enemy_id", "enemy"),
    )
    for a, b, kind in fields:
        value = payload.get(a, payload.get(b))
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid action pointer {a}")
            refs.append(ObjectRef(kind, value))
    if action_kind == "use_potion":
        slot = payload.get("Slot", payload.get("slot"))
        if type(slot) is not int or slot < 0:
            raise ValueError("Invalid potion slot")
        refs.insert(0, ObjectRef("potion", slot))
    index = objects.index()
    for ref in refs:
        if ref not in index:
            raise ValueError(f"Legal action references an absent object: {ref}")
    return tuple(refs)
