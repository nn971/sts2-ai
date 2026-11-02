"""Pilot-only extension of the H7 public codec for persistent resources.

Keeps H7's shared Transformer tensor layout and the combat hand-card codec,
while differentiating macro choices and exposing observable nested persistent
state. No resource-specific scalar prices or private RNG information enter.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from sts2_ai.models.shared_game_objects import (
    IDENTITY_BUCKETS,
    KINDS,
    MAX_OBJECTS,
    ZONES,
    ActionToken,
    EncodedDecision,
    ObjectToken,
    _bucket,
    _numbers,
    encode_public_decision,
)

H16_CODEC = "h16-public-persistent-subset-multiset-v3"
PREVIOUS_H16_CODEC = "h16-public-persistent-subset-actions-v2"
LEGACY_H16_CODEC = "h16-public-persistent-macro-actions-v1"
MAX_EXTRA_TOKENS = 64
# Once per process: expose unusual real/search menus without dumping states.
_LARGE_MENU_REPORTED = False
_CURRENT_EPISODE_SEED: str | None = None
_SELECT_ALIAS_REPORTED = False


def set_h16_rollout_seed(seed: str | None) -> None:
    global _CURRENT_EPISODE_SEED
    _CURRENT_EPISODE_SEED = seed


def _diagnose_large_menu(state: dict[str, Any], actions: Sequence[Any]) -> None:
    global _LARGE_MENU_REPORTED
    if len(actions) <= 128 or _LARGE_MENU_REPORTED:
        return
    _LARGE_MENU_REPORTED = True
    examples: dict[str, list[dict[str, str]]] = {}
    for action in actions:
        sample = examples.setdefault(action.kind, [])
        if len(sample) < 3:
            sample.append({
                "action_id": str(action.action_id),
                "payload": str(action.payload_json)[:240],
            })
    combat = state.get("combat")
    detail = {
        "seed": _CURRENT_EPISODE_SEED,
        "floor": state.get("floor"),
        "phase": "combat" if isinstance(combat, dict) else "macro",
        "turn": combat.get("turn") if isinstance(combat, dict) else None,
        "num_legal_actions": len(actions),
        "action_kinds": dict(sorted(Counter(a.kind for a in actions).items())),
        "samples": examples,
    }
    print("[h16-action-overflow] " + json.dumps(detail, sort_keys=True),
          file=sys.stderr, flush=True)


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
    # H16/PUCT must preserve every legal action, including combinatorial
    # target/choice menus exceeding H7's historical 128-action guard.
    # collate_h7 and the shared actor already allocate by actual menu length.
    _diagnose_large_menu(state, actions)
    base = encode_public_decision(
        state, codec_actions, max_actions=None,
        encode_card_subsets=not macro,
        compress_deck=True,
    )
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
        seen_deck_resources: set[str] = set()
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            if field == "deck":
                # Public deck cards are already encoded as multiset classes.
                # Emit persistent-state leaves once per identical class,
                # keeping the 64-token auxiliary cap independent of copies.
                signature = json.dumps(
                    {key: value for key, value in resource.items()
                     if key != "instance_id"},
                    sort_keys=True, separators=(",", ":"),
                )
                if signature in seen_deck_resources:
                    continue
                seen_deck_resources.add(signature)
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
    # Hash-bucket collisions are possible in 4,096 dimensions. Resolve
    # collisions among different macro choices within the current menu.
    used_macro_semantics: dict[int, str] = {}
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
        semantic = _bucket(label, IDENTITY_BUCKETS)
        salt = 0
        while (previous := used_macro_semantics.get(semantic)) is not None and previous != label:
            salt += 1
            if salt >= IDENTITY_BUCKETS:
                raise ValueError("H16 macro menu exceeds unique semantic bucket capacity")
            semantic = _bucket(f"{label}:collision={salt}", IDENTITY_BUCKETS)
        used_macro_semantics[semantic] = label
        encoded_actions.append(ActionToken(
            old.kind, semantic, old.source, old.target,
        ))
    # The emulator groups only truly interchangeable internal cards.
    # Distinct internal card variants can have the same *public* projection:
    # an H16 multiset card token then represents several legal selections.
    # Preserve both actions without putting arbitrary instance IDs in learned
    # embeddings. Only in this exceptional case, give the alternatives a
    # stable ordinal within their public action class. Other collisions are
    # still errors, so new missing representations remain detectable.
    seen: dict[ActionToken, tuple[int, Any]] = {}
    aliases: dict[ActionToken, list[int]] = {}
    for index, (action, token) in enumerate(
        zip(actions, encoded_actions, strict=True)
    ):
        previous = seen.get(token)
        if previous is None:
            seen[token] = (index, action)
            continue
        prev_index, prior = previous
        if prior.kind == action.kind and (
            json.loads(prior.payload_json) == json.loads(action.payload_json)
        ):
            continue
        if base.phase != "combat" or action.kind != "select_cards" or (
            prior.kind != "select_cards"
        ):
            raise ValueError(
                "H16 action representation collision: "
                f"phase={base.phase}, floor={state.get('floor')}, "
                f"kind={action.kind}, payloads="
                f"{str(prior.payload_json)[:200]} and "
                f"{str(action.payload_json)[:200]}"
            )
        aliases.setdefault(token, [prev_index]).append(index)

    if aliases:
        global _SELECT_ALIAS_REPORTED
        if not _SELECT_ALIAS_REPORTED:
            _SELECT_ALIAS_REPORTED = True
            combat = state.get("combat")
            print("[h16-selection-alias] " + json.dumps({
                "seed": _CURRENT_EPISODE_SEED,
                "floor": state.get("floor"),
                "turn": combat.get("turn") if isinstance(combat, dict) else None,
                "groups": len(aliases),
                "alternatives": sum(len(x) for x in aliases.values()),
                "example": [actions[i].payload_json for i in
                            next(iter(aliases.values()))[:4]],
            }, sort_keys=True), file=sys.stderr, flush=True)

        used = set(encoded_actions)
        for token, indices in aliases.items():
            # Canonicalizing the selection payload ensures stable ordering
            # even if the emulator emits the legal menu in another order.
            variants: dict[tuple[int, ...], list[int]] = {}
            for index in indices:
                payload = json.loads(actions[index].payload_json)
                members = payload.get(
                    "CardInstanceIds", payload.get("card_instance_ids"),
                )
                if not isinstance(members, list) or any(
                    type(item) is not int for item in members
                ):
                    raise ValueError("Malformed public select_cards alias")
                variants.setdefault(tuple(sorted(members)), []).append(index)
            for ordinal, (_, positions) in enumerate(
                sorted(variants.items())
            ):
                if ordinal == 0:
                    continue  # Retain the original class token for one choice.
                salt = 0
                while True:
                    semantic = _bucket(
                        f"h16-select-cards-alias:{token.semantic}:{ordinal}:{salt}",
                        IDENTITY_BUCKETS,
                    )
                    replacement = ActionToken(
                        token.kind, semantic, token.source, token.target,
                        token.members,
                    )
                    if replacement not in used:
                        break
                    salt += 1
                    if salt >= IDENTITY_BUCKETS:
                        raise ValueError("H16 select_cards alias capacity exceeded")
                used.add(replacement)
                for position in positions:
                    encoded_actions[position] = replacement

        # The usual path stays O(actions) without extra JSON parsing.
        # Audit the rewritten menu only when a collision actually occurred.
        seen_payloads: dict[ActionToken, tuple[str, object]] = {}
        for action, token in zip(actions, encoded_actions, strict=True):
            payload = json.loads(action.payload_json)
            current = (action.kind, payload)
            prior = seen_payloads.get(token)
            if prior is not None and prior != current:
                raise ValueError(
                    "H16 action representation collision after disambiguation: "
                    f"phase={base.phase}, floor={state.get('floor')}, "
                    f"kind={action.kind}"
                )
            seen_payloads[token] = current
    return EncodedDecision(
        base.schema, base.phase, tuple(objects), tuple(encoded_actions),
    )
