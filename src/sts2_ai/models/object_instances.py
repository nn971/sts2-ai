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
CARD_ZONES = (
    "hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile",
    "choice_pool",
)
# The pinned emulator does not yet project its ChoicePool, although its
# select_cards menu may refer to cards inside it. Placeholder attributes are
# explicitly unknown, never inferred from an engine card ID.
UNOBSERVED_CHOICE_TYPE = "<unobserved-choice>"
# This list is deliberately auditable. Do not recursively tokenize the full observation.
CARD_DYNAMIC_FIELDS = (
    "state", "affliction", "enchantment", "temporary_energy_cost",
    "enchantment_triggered_this_combat", "cost", "energy_cost",
)


@dataclass(frozen=True, slots=True)
class ObjectRef:
    kind: str
    key: int | str | tuple[str, str] | tuple[str, str, int]


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
            # Instance identifiers are pointers, never scalar neural input.
            # Includes nested affliction.source_enemy_instance_id and arrays
            # of instance IDs; changing arbitrary labels cannot change logits.
            if v is not None and (
                k == "instance_id"
                or k.endswith(("_instance_id", "_instance_ids"))
                or k.endswith(("InstanceId", "InstanceIds"))
            ):
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
        cards = _records(
            combat.get(zone, []) if zone == "choice_pool" else combat.get(zone),
            "combat." + zone,
        )
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
        # Powers can be instanced: a single owner may legitimately have two
        # records with the same power_id (e.g. proto.power.strangle). Each
        # occurrence must remain distinct; combining stacks would change the
        # number of objects and potentially the modeled game mechanics.
        # Occurrence numbers are transient lookup keys only, never features.
        occurrences: dict[str, int] = {}
        for power in _records(powers, "powers"):
            ident = _identifier(power, "power_id", "power")
            ordinal = occurrences.get(ident, 0)
            occurrences[ident] = ordinal + 1
            num = {"stacks": _number(power.get("stacks"), "power.stacks")}
            objects.append(_token(
                "power", ident, "active",
                ObjectRef("power", (str(owner), ident, ordinal)),
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
    # "select_cards" is a combat decision in the emulator, not a macro
    # reward choice. Each option contains a combination of card instances.
    # Without these edges, every combination scores the same residual.
    if action_kind == "select_cards":
        choices = payload.get("CardInstanceIds", payload.get("card_instance_ids"))
        if not isinstance(choices, list):
            raise ValueError("Missing CardInstanceIds in select_cards action")
        if any(type(value) is not int or value <= 0 for value in choices):
            raise ValueError("Invalid card selection instance pointer")
        if len(set(choices)) != len(choices):
            raise ValueError("Duplicate card in select_cards action")
        refs.extend(ObjectRef("card", value) for value in choices)
    else:
        card = payload.get("CardInstanceId", payload.get("card_instance_id"))
        # The native emulator always supplies CardInstanceId for play_card.
        # Legacy/test emulators may expose only a kind and an opaque action
        # ID: preserve the empty-ref fallback instead of rejecting the run.
        if card is not None:
            if type(card) is not int or card <= 0:
                raise ValueError("Invalid action pointer CardInstanceId")
            refs.append(ObjectRef("card", card))
        target = payload.get("TargetEnemyId", payload.get("target_enemy_id"))
        if target is not None:
            if type(target) is not int or target <= 0:
                raise ValueError("Invalid action pointer TargetEnemyId")
            refs.append(ObjectRef("enemy", target))
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


@dataclass(frozen=True, slots=True)
class ActionBinding:
    """A visible action-object edge with an explicit *semantic* role.

    No instance integer is ever an input feature. Role preserves the
    difference between a card source, potion source, enemy target, and an
    individual card chosen by a pending combat selection.
    """
    role: str
    ref: ObjectRef


def legal_action_object_bindings(
    action_kind: str, payload_json: str, objects: ObjectSet,
) -> tuple[ActionBinding, ...]:
    """Return typed action references without changing the v15c pointer API."""
    refs = legal_action_object_refs(action_kind, payload_json, objects)
    if action_kind == "select_cards":
        return tuple(ActionBinding("selected_card", ref) for ref in refs)
    roles = {
        "card": "source_card",
        "potion": "source_potion",
        "enemy": "target_enemy",
    }
    return tuple(ActionBinding(roles[ref.kind], ref) for ref in refs)


def extract_actionable_combat_objects(
    state: dict[str, Any],
    legal_actions: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> ObjectSet:
    """Resolve *public-menu* selection pointers omitted from older observations.

    The emulator's generated-card ChoicePool is legal-action-visible but is
    missing from the pinned public observation schema. When and ONLY when the
    pending choice is select_cards, represent menu-referenced absent cards
    with explicitly unobserved placeholders. Never fabricate their type,
    stats or effects, and never treat instance IDs as numeric features.

    Once the emulator exposes combat.choice_pool with actual public card
    records, normal extraction supplies the real tokens and no placeholder
    is created. Non-choice actions retain strict pointer validation.
    """
    objects = extract_combat_objects(state)
    combat = state["combat"]
    assert isinstance(combat, dict)
    if combat.get("pending_choice_id") != "select_cards":
        return objects
    known = objects.index()
    absent: set[int] = set()
    for kind, raw in legal_actions:
        if kind != "select_cards":
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed legal action JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Malformed legal action payload")
        choices = payload.get("CardInstanceIds", payload.get("card_instance_ids"))
        if not isinstance(choices, list):
            raise ValueError("Missing CardInstanceIds in select_cards action")
        if any(type(value) is not int or value <= 0 for value in choices):
            raise ValueError("Invalid card selection instance pointer")
        if len(set(choices)) != len(choices):
            raise ValueError("Duplicate card in select_cards action")
        for value in choices:
            if ObjectRef("card", value) not in known:
                absent.add(value)
    if not absent:
        return objects
    # Sorting provides action-menu-order invariance. IDs are routing keys
    # only: the neural inputs of every unavailable card are identical.
    extras = tuple(
        ObjectToken(
            "card", UNOBSERVED_CHOICE_TYPE, "unobserved_choice",
            ObjectRef("card", value), None, (),
            ("attributes_unavailable",),
        )
        for value in sorted(absent)
    )
    return ObjectSet(objects.schema, objects.tokens + extras)
