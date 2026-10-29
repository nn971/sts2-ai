"""H15 combat encoding: H7-compatible weights with source-visible enemy moves.

Do not mutate the H9 codec or reinterpret a H15 checkpoint as an H9 checkpoint.
The new token has its enemy as owner and carries only a public move identity.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from sts2_ai.models.shared_game_objects import (
    IDENTITY_BUCKETS,
    KINDS,
    MAX_OBJECTS,
    NUMERIC_FIELDS,
    SHARED_OBJECT_SCHEMA,
    ZONES,
    EncodedDecision,
    ObjectToken,
    _bucket,
    encode_public_decision,
)
if TYPE_CHECKING:
    from sts2_ai.models.shared_transformer import H7Batch

H15_COMBAT_SCHEMA = "sts2-h15-combat-moves-v1"


def encode_h15_combat(state: dict[str, Any], legal_actions: Sequence[Any]) -> EncodedDecision:
    """Retain H7 legal-action pointers and append enemy-owned, visible move tokens.

    This is additive, so H9 embeddings/encoder/action-head tensors remain shape
    compatible; H15 checkpoints must always use this codec during inference.
    """
    encoded = encode_public_decision(state, legal_actions)
    combat = state.get("combat")
    if encoded.phase != "combat" or not isinstance(combat, dict):
        raise ValueError("H15 requires a public combat decision")
    enemies = combat.get("enemies")
    if not isinstance(enemies, list) or any(not isinstance(e, dict) for e in enemies):
        raise ValueError("H15 requires public enemy records")
    enemy_kind = KINDS.index("enemy")
    enemy_zone = ZONES.index("enemy")
    owners = [i for i, obj in enumerate(encoded.objects)
              if obj.kind == enemy_kind and obj.zone == enemy_zone]
    if len(owners) != len(enemies):
        raise ValueError("H15 enemy tokens disagree with visible enemy records")
    objects = list(encoded.objects)
    zeros = (0.0,) * len(NUMERIC_FIELDS)
    absent = (0,) * len(NUMERIC_FIELDS)
    for enemy, owner in zip(enemies, owners, strict=True):
        move = enemy.get("move_id")
        if isinstance(move, str) and move:
            if len(objects) >= MAX_OBJECTS:
                raise ValueError("H15 move tokens exceed public object cap")
            objects.append(ObjectToken(
                KINDS.index("power"),
                _bucket("intent-move:" + move, IDENTITY_BUCKETS),
                enemy_zone, zeros, absent, owner,
            ))
    return replace(encoded, schema=H15_COMBAT_SCHEMA, objects=tuple(objects))


def collate_h15(decisions: Sequence[EncodedDecision]) -> H7Batch:
    """The H7 tensor layout is unchanged; the externally checked schema is not."""
    if not decisions or any(d.schema != H15_COMBAT_SCHEMA or d.phase != "combat"
                            for d in decisions):
        raise ValueError("H15 batch requires versioned combat observations")
    from sts2_ai.models.shared_transformer import collate_h7

    return collate_h7([replace(d, schema=SHARED_OBJECT_SCHEMA) for d in decisions])


def encode_h15_observation(observation_json: str, legal_actions: Sequence[Any]) -> EncodedDecision:
    raw = json.loads(observation_json)
    if not isinstance(raw, dict):
        raise ValueError("Public observation must be an object")
    return encode_h15_combat(raw, legal_actions)
