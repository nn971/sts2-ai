"""Structured, public-only tactical state features for phase-split v4 models.

Numeric combat quantities occupy reserved coordinates. Named cards, enemy
moves, statuses and potions hash only into the remaining coordinates. This
does not inspect emulator handles, RNG seeds, unrevealed draws or future states.
The schema must change whenever these feature semantics change.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

TACTICAL_STATE_SCHEMA = "sts2-tactical-public-state-v4"
NUMERIC_COORDINATES = 32


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else 0.0
    return 0.0


def _objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


def _add_tokens(
    result: Counter[str], prefix: str, items: list[dict[str, Any]], key: str,
) -> None:
    for item in items:
        name = item.get(key)
        if isinstance(name, str) and name:
            result[f"{prefix}:{name}"] += 1
            upgraded = item.get("upgrade_level")
            if prefix in ("deck", "hand") and _number(upgraded) > 0:
                result[f"{prefix}-upgraded:{name}"] += 1


def tactical_state_features(
    state: dict[str, Any], dimension: int,
) -> dict[int, float]:
    """Encode a visible tactical frame without map-node/instance noise.

    Critical HP, attack pressure, block, energy, hand and consumable counts
    have collision-free numeric coordinates. Names are stable-hashed separately.
    """
    if dimension <= NUMERIC_COORDINATES:
        raise ValueError("Tactical state dimension must exceed 32")
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Tactical state encoder requires a public combat frame")

    enemies = _objects(combat.get("enemies"))
    hand = _objects(combat.get("hand"))
    deck = _objects(state.get("deck"))
    potions = _objects(state.get("potions"))
    relics = _objects(state.get("relics"))
    draw = _objects(combat.get("discard_pile"))
    exhausted = _objects(combat.get("exhaust_pile"))
    player_powers = _objects(combat.get("player_powers"))

    hp = _number(state.get("hp"))
    max_hp = max(1.0, _number(state.get("max_hp")))
    enemy_hp = [_number(e.get("hp")) for e in enemies]
    enemy_block = [_number(e.get("block")) for e in enemies]
    # Continuous features remain continuous, even for different HP values
    # within the same five-HP bin.
    values = [
        1.0,
        hp / max_hp,
        hp / 100.0,
        max_hp / 100.0,
        _number(state.get("floor")) / 16.0,
        _number(state.get("act")) / 3.0,
        len(enemies) / 5.0,
        sum(enemy_hp) / 200.0,
        max(enemy_hp, default=0.0) / 200.0,
        min(enemy_hp, default=0.0) / 100.0,
        sum(enemy_block) / 100.0,
        _number(combat.get("player_block")) / 50.0,
        _number(combat.get("energy")) / 5.0,
        _number(combat.get("turn")) / 10.0,
        len(hand) / 10.0,
        _number(combat.get("draw_pile_count")) / 20.0,
        len(draw) / 20.0,
        len(exhausted) / 20.0,
        len(deck) / 40.0,
        len(potions) / 4.0,
        len(relics) / 15.0,
        _number(state.get("gold")) / 500.0,
        len(player_powers) / 10.0,
        sum(enemy_hp) / max(1.0, hp) / 10.0,
        float(hp <= 20),
        float(hp <= 40),
        float(len(potions) == 0),
        float(_number(combat.get("energy")) <= 0),
        float(len(enemies) > 1),
        float(max(enemy_hp, default=0) >= 150),
        float(len(hand) == 0),
        float(combat.get("pending_choice_id") is not None),
    ]
    features = {
        i: min(8.0, max(-8.0, value))
        for i, value in enumerate(values) if value != 0.0
    }

    tokens: Counter[str] = Counter()
    _add_tokens(tokens, "enemy", enemies, "enemy_id")
    _add_tokens(tokens, "hand", hand, "card_id")
    _add_tokens(tokens, "deck", deck, "card_id")
    _add_tokens(tokens, "potion", potions, "potion_id")
    _add_tokens(tokens, "relic", relics, "relic_id")
    _add_tokens(tokens, "discard", draw, "card_id")
    _add_tokens(tokens, "exhaust", exhausted, "card_id")
    _add_tokens(tokens, "power", player_powers, "power_id")
    for enemy in enemies:
        for name in ("move_id", "intent", "intent_id"):
            move = enemy.get(name)
            if isinstance(move, str) and move:
                tokens[f"enemy-{name}:{move}"] += 1
        powers = _objects(enemy.get("powers"))
        _add_tokens(tokens, "enemy-power", powers, "power_id")

    boss = state.get("act_one_boss_encounter_id")
    if isinstance(boss, str) and boss:
        tokens[f"boss:{boss}"] += 1
    for token, count in tokens.items():
        digest = hashlib.sha256(
            ("tactical-state-v4:" + token).encode("utf-8")
        ).digest()
        index = NUMERIC_COORDINATES + (
            int.from_bytes(digest[:8], "big") % (dimension - NUMERIC_COORDINATES)
        )
        features[index] = min(
            8.0, features.get(index, 0.0) + min(count, 8) / 8.0
        )
    return features
