"""H24 canonical public-state projection for a resource-neutral value encoder.

The ordinary H21 policy encoder continues to see the entire public state and
legal-action menu. The H24 value branch uses an *independent*, deterministic
single-action observation to eliminate action-menu leakage, then removes only
the resource coordinates to which the certified monotone head applies.

This is a structural invariant, not a certification of game mechanics.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h21_macro_offer_codec import encode_h21_macro_decision
from sts2_ai.training.h24_applicability import ResourceConstraintSelection
from sts2_ai.training.h24_resource_monotone import SafeResourceVocabulary

H24_NEUTRAL_ENCODER = "sts2-h24-neutral-value-h21-v1"
H24_CONTEXT_ANCHOR = LegalAction(
    "h24_value_context_only", "h24_value_context_only", "{}",
)


@dataclass(frozen=True)
class H24NeutralValue:
    encoded: EncodedDecision
    hp: float | None
    gold: float | None
    potions: tuple[float, ...]
    relics: tuple[float, ...]


def _public_resources(state: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    values = state.get(kind)
    if not isinstance(values, list) or any(not isinstance(x, dict) for x in values):
        raise ValueError(f"H24 missing/invalid public {kind}")
    return values


def neutralize_h24_state(
    state: Mapping[str, Any], *, selection: ResourceConstraintSelection | None = None,
    vocabulary: SafeResourceVocabulary | None = None,
) -> tuple[dict[str, Any], float | None, float | None,
           tuple[float, ...], tuple[float, ...]]:
    """Canonicalize only selected resource coordinates, preserving everything else.

    Resource item *identities* without an approved coefficient remain visible
    inside H21's value context. A matching pair must yield an identical
    canonical encoding, checked at the actual H21 token boundary.
    """
    chosen = selection if selection is not None else ResourceConstraintSelection()
    words = vocabulary if vocabulary is not None else SafeResourceVocabulary()
    if not isinstance(chosen, ResourceConstraintSelection):
        raise TypeError("Expected H24 resource selection")
    if not isinstance(words, SafeResourceVocabulary):
        raise TypeError("Expected H24 resource vocabulary")
    if words.relic_ids and not chosen.relic:
        raise ValueError("Relic resource order is disabled")
    if words.potion_ids and not chosen.potion:
        raise ValueError("Potion resource order is disabled")
    if not isinstance(state, Mapping):
        raise TypeError("Expected public state mapping")
    projected: dict[str, Any] = deepcopy(dict(state))
    hp: float | None = None
    gold: float | None = None
    if chosen.hp:
        value = projected.get("hp")
        maxhp = projected.get("max_hp")
        if (type(value) is not int or type(maxhp) is not int
                or not 0 < value <= maxhp):
            raise ValueError("Invalid public H24 HP")
        hp = float(value)
        projected["hp"] = 1  # canonical positive HP, preserves max HP
    if chosen.gold:
        value = projected.get("gold")
        if type(value) is not int or value < 0:
            raise ValueError("Invalid public H24 gold")
        gold = float(value)
        projected["gold"] = 0
    potion_counts: dict[str, float] = {item: 0.0 for item in words.potion_ids}
    if chosen.potion:
        capacity = projected.get("potion_slot_capacity")
        occupied = projected.get("potion_slots_occupied")
        empty = projected.get("potion_slots_empty")
        items = _public_resources(projected, "potions")
        if (type(capacity) is not int or type(occupied) is not int
                or type(empty) is not int or capacity < 0
                or occupied < 0 or empty < 0
                or occupied + empty != capacity or occupied != len(items)):
            raise ValueError("Invalid H24 potion slot inventory")
        slots: set[int] = set()
        remaining: list[dict[str, Any]] = []
        for item in items:
            slot = item.get("slot")
            name = item.get("potion_id")
            if (type(slot) is not int or not 0 <= slot < capacity
                    or slot in slots or not isinstance(name, str)
                    or not name or not isinstance(item.get("state"), dict)):
                raise ValueError("Invalid H24 public potion")
            slots.add(slot)
            if name in potion_counts:
                potion_counts[name] += 1
            else:
                remaining.append(item)
        projected["potions"] = remaining
        projected["potion_slots_occupied"] = len(remaining)
        projected["potion_slots_empty"] = capacity - len(remaining)
    relic_counts: dict[str, float] = {item: 0.0 for item in words.relic_ids}
    if chosen.relic:
        remaining_relics: list[dict[str, Any]] = []
        for item in _public_resources(projected, "relics"):
            name = item.get("relic_id")
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid H24 public relic")
            if name in relic_counts:
                relic_counts[name] += 1
            else:
                remaining_relics.append(item)
        projected["relics"] = remaining_relics
    return (projected, hp, gold, tuple(potion_counts.values()),
            tuple(relic_counts.values()))


def encode_h24_neutral_value(
    state: Mapping[str, Any], *, selection: ResourceConstraintSelection | None = None,
    vocabulary: SafeResourceVocabulary | None = None,
) -> H24NeutralValue:
    projected, hp, gold, potions, relics = neutralize_h24_state(
        state, selection=selection, vocabulary=vocabulary)
    # Explicitly fixed menu: avoids card, potion and shop action leakage into
    # the value embedding, including macro action-grounding side effects.
    encoded = encode_h21_macro_decision(
        projected, (H24_CONTEXT_ANCHOR,),
        public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1,
    )
    if len(encoded.actions) != 1:
        raise AssertionError("Value context must use exactly one fixed anchor")
    return H24NeutralValue(encoded, hp, gold, potions, relics)


def assert_h24_pair_neutral_parity(
    lower: Mapping[str, Any], higher: Mapping[str, Any], *,
    selection: ResourceConstraintSelection | None = None,
    vocabulary: SafeResourceVocabulary | None = None,
) -> None:
    """Fail closed unless the complete H21 value tokens/graph/actions match."""
    a = encode_h24_neutral_value(lower, selection=selection, vocabulary=vocabulary)
    b = encode_h24_neutral_value(higher, selection=selection, vocabulary=vocabulary)
    if a.encoded != b.encoded:
        raise ValueError("H24 counterfactual changes the neutral H21 value context")
