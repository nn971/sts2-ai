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


def encode_h19_resource_decision(
    state: dict[str, Any],
    actions: Sequence[Any],
    *,
    public_schema_id: str,
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
    combat = state.get("combat")
    if isinstance(combat, dict) and not isinstance(combat.get("history"), dict):
        raise ValueError("H19 requires public combat.history")

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
    )
    encoded_actions = _h16_actions(base, state, actions)
    return EncodedDecision(
        H19_OBJECT_SCHEMA, base.phase, base.objects, encoded_actions)
