"""Versioned public-observation v1 owner-linked resource codec.

This is a separate opt-in representation. Legacy H7/H16/H18 checkpoints
retain their original vocabulary and schema; a fresh model must be trained
for the new owner-linked attribute kind.
"""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from sts2_ai.models.shared_game_objects import (
    EncodedDecision,
    encode_public_decision,
)
from sts2_ai.training.h16_public_codec import _h16_actions

EMULATOR_PUBLIC_SCHEMA_V1 = "prototype-ai-v1"
H19_RESOURCE_CODEC = "h19-public-owned-resources-v1"
H19_OBJECT_SCHEMA = "sts2-h19-public-owned-resources-v1"
H19_OBJECT_LIMIT = 1024

# Machine-readable first-slice contract. New emulator fields must be
# classified before they can enter this versioned training representation.
# Source references are local keys, never learned scalar identities.
PUBLIC_RESOURCE_FIELDS_V1 = {
    "deck": frozenset({
        "instance_id", "card_id", "upgrade_level", "is_temporary",
        "state", "affliction", "enchantment", "combat_energy_cost_delta",
        "temporary_energy_cost", "keyword_overrides",
        "enchantment_triggered_this_combat", "cost_kind",
        "effective_energy_cost",
    }),
    "relics": frozenset({"relic_id", "state", "is_wax", "is_melted"}),
    "potions": frozenset({"slot", "potion_id", "state"}),
}
PUBLIC_CARD_FIELDS_V1 = PUBLIC_RESOURCE_FIELDS_V1["deck"]


def _check_public_resource_fields(
    resource: dict[str, Any], kind: str, *,
    allow_affliction_source_relations: bool = False,
) -> None:
    extra = set(resource) - PUBLIC_RESOURCE_FIELDS_V1[kind]
    if extra:
        raise ValueError(
            "Unclassified public resource fields in " + kind + ": "
            + ", ".join(sorted(extra)))
    if kind == "deck":
        for field, allowed in (
            ("enchantment", {"kind", "amount", "disabled"}),
            ("temporary_energy_cost", {"cost", "expiry"}),
            ("affliction", {"kind", "amount", "source_enemy_instance_id"}),
        ):
            child = resource.get(field)
            if isinstance(child, dict):
                unknown = set(child) - allowed
                if unknown:
                    raise ValueError(
                        "Unclassified public " + field + " fields: "
                        + str(sorted(unknown)))
        for override in resource.get("keyword_overrides") or ():
            if not isinstance(override, dict) or set(override) - {
                "keyword", "enabled", "expiry",
            }:
                raise ValueError("Unclassified public keyword override")
        affliction = resource.get("affliction")
        if isinstance(affliction, dict):
            unknown = set(affliction) - {
                "kind", "amount", "source_enemy_instance_id",
            }
            if unknown:
                raise ValueError("Unclassified affliction fields: " + str(sorted(unknown)))
            # The source is an enemy relation, never a learned ordinal.
            # Fail closed until the cross-entity edge is implemented.
            source = affliction.get("source_enemy_instance_id")
            if source is not None:
                if not allow_affliction_source_relations:
                    raise ValueError(
                        "Affliction source_enemy_instance_id requires an explicit relation")
                if type(source) is not int:
                    raise ValueError("Invalid affliction enemy reference")


def encode_h19_resource_decision(
    state: dict[str, Any],
    actions: Sequence[Any],
    *,
    public_schema_id: str,
    allow_affliction_source_relations: bool = False,
) -> EncodedDecision:
    """Encode only explicitly versioned public data, preserving H16 action IDs.

    Schema is a mandatory caller-supplied frame property; the standalone
    observation JSON does not carry its framing schema ID. This prevents
    a silently changed H16/H18 input from sharing H19 checkpoint semantics.
    """
    if public_schema_id != EMULATOR_PUBLIC_SCHEMA_V1:
        raise ValueError(
            "H19 requires emulator prototype-ai-v1 observation frame")
    if not isinstance(state, dict):
        raise ValueError("H19 requires an object-valued public observation")
    for key in ("potion_slot_capacity", "potion_slots_occupied",
                "potion_slots_empty"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError("H19 requires public potion inventory counts: " + key)
    capacity = state["potion_slot_capacity"]
    occupied = state["potion_slots_occupied"]
    empty = state["potion_slots_empty"]
    if occupied + empty != capacity:
        raise ValueError("Inconsistent public potion inventory counts")
    for key in ("deck", "relics", "potions"):
        if not isinstance(state.get(key), list):
            raise ValueError("H19 missing public resource list: " + key)
        for resource in state[key]:
            if not isinstance(resource, dict) or not isinstance(resource.get("state"), dict):
                raise ValueError("H19 expects canonical public resource.state: " + key)
            if "persistent_state" in resource:
                raise ValueError("Legacy persistent_state is not a public v1 field")
            _check_public_resource_fields(
                resource, key,
                allow_affliction_source_relations=allow_affliction_source_relations,
            )
    if len(state["potions"]) != occupied:
        raise ValueError("Occupied potion slot count disagrees with public potions")
    slots = [potion.get("slot") for potion in state["potions"]]
    if (any(type(slot) is not int or slot < 0 or slot >= capacity for slot in slots)
            or len(set(slots)) != len(slots)):
        raise ValueError("Public potion slot references are invalid")

    combat = state.get("combat")
    if isinstance(combat, dict):
        if not isinstance(combat.get("history"), dict):
            raise ValueError("H19 requires public combat.history")
        for field in ("hand", "discard_pile", "exhaust_pile", "play_pile",
                      "selection_candidates", "selection_selected_cards"):
            cards = combat.get(field)
            if cards is None:
                continue
            if not isinstance(cards, list):
                raise ValueError("Malformed public combat card collection: " + field)
            for card in cards:
                if not isinstance(card, dict):
                    raise ValueError("Malformed public combat card")
                _check_public_resource_fields(
                    card, "deck",
                    allow_affliction_source_relations=allow_affliction_source_relations,
                )

    # The existing H16 macro-kind+payload collision guard and combat target
    # validation still apply. Only v2 resource attributes are appended, so
    # original source/target/member indices remain stable and exact.
    macro = not isinstance(combat, dict)
    codec_actions = (
        [SimpleNamespace(kind=a.kind, payload_json="{}") for a in actions]
        if macro else actions
    )
    base = encode_public_decision(
        state, codec_actions,
        max_actions=None, compress_deck=True,
        encode_card_subsets=not macro,
        owned_attributes=True, object_limit=H19_OBJECT_LIMIT,
        schema_id=H19_OBJECT_SCHEMA,
        omit_affliction_source_reference=allow_affliction_source_relations,
    )
    encoded_actions = _h16_actions(base, state, actions)
    return EncodedDecision(
        H19_OBJECT_SCHEMA, base.phase, base.objects, encoded_actions)
