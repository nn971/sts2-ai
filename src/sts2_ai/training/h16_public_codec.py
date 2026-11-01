"""Pilot-only extension of the H7 public codec for persistent resources.

Keeps H7's shared Transformer tensor layout and the combat hand-card codec,
while differentiating macro choices and exposing observable nested persistent
state. No resource-specific scalar prices or private RNG information enter.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from sts2_ai.models.shared_game_objects import (
    IDENTITY_BUCKETS, KINDS, MAX_OBJECTS, ZONES, ActionToken,
    EncodedDecision, ObjectToken, _bucket, _numbers, encode_public_decision,
)

H16_CODEC = "h16-public-persistent-macro-actions-v1"
MAX_EXTRA_TOKENS = 64


def _leaves(value: object, path: str):
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            if isinstance(key, str):
                yield from _leaves(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, f"{path}[{index}]")
    elif type(value) in (int, float, bool, str):
        yield path, value


def encode_h16_decision(
    state: dict[str, Any], actions: Sequence[Any],
) -> EncodedDecision:
    # H7's CardInstanceId pointer validator assumes every selected card is
    # in the combat hand. Macro rewards/shops have different public card
    # references; their full offered choice is encoded in the action semantic
    # below instead. Keep the original strict combat pointer validation.
    macro = not isinstance(state.get("combat"), dict)
    codec_actions = (
        [SimpleNamespace(kind=a.kind, payload_json="{}") for a in actions]
        if macro else actions
    )
    base = encode_public_decision(state, codec_actions)
    objects = list(base.objects)
    extra = 0
    for field, name_fields in (
        ("relics", ("relic_id", "id")),
        ("potions", ("potion_id", "id")),
        ("deck", ("card_id", "id")),
    ):
        resources = state.get(field)
        if not isinstance(resources, list):
            continue
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            name = next(
                (resource[key] for key in name_fields
                 if isinstance(resource.get(key), str)), "unknown",
            )
            persistent = resource.get("persistent_state")
            if not isinstance(persistent, (dict, list)):
                continue
            for path, value in _leaves(persistent, field + ":" + name):
                extra += 1
                if extra > MAX_EXTRA_TOKENS or len(objects) >= MAX_OBJECTS:
                    raise ValueError("H16 observable persistent resource token cap exceeded")
                number = float(value) if type(value) in (int, float, bool) else 1.0
                # Identity encodes field semantics; numeric value is supplied
                # through the existing masked stacks channel.
                identity = path if type(value) is not str else path + ":" + value
                numeric, observed = _numbers({"stacks": number})
                objects.append(ObjectToken(
                    KINDS.index("status"), _bucket(identity, IDENTITY_BUCKETS),
                    ZONES.index("inventory"), numeric, observed,
                ))
    encoded_actions: list[ActionToken] = []
    for old, action in zip(base.actions, actions, strict=True):
        if base.phase == "combat" or old.source >= 0:
            encoded_actions.append(old)  # preserve H9 combat token behavior
            continue
        payload = json.loads(action.payload_json)
        # Public chosen option identity, rather than a flat "event_choice"
        # kind token shared by every legal macro option.
        label = json.dumps(
            {"kind": action.kind, "payload": payload},
            sort_keys=True, separators=(",", ":"),
        )
        encoded_actions.append(ActionToken(
            old.kind, _bucket(label, IDENTITY_BUCKETS),
            old.source, old.target,
        ))
    return EncodedDecision(
        base.schema, base.phase, tuple(objects), tuple(encoded_actions),
    )
