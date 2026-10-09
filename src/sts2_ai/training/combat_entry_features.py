"""Player-visible structured combat-entry encoder (stable v2 schema).

Use reserved numeric coordinates for combat quantities and stable-hashed
categorical IDs. The full map's many arbitrary node identifiers must not
collide with player/enemy HP or potion features as in generic state hashing.
Only current public observation is used: no hidden RNG, exact state or
post-combat outcomes.

This is deliberately simple and interpretable, not a learned entity/attention
encoder. Future encoders should use new version identifiers.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

FEATURE_SCHEMA = "sts2-combat-entry-structured-v2"
NUMERIC_FEATURES = 24


def _number(value: object) -> float:
    if type(value) in (int, float):
        result = float(value)
        if math.isfinite(result):
            return result
    return 0.0


def _objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]


def combat_entry_features(state: dict[str, Any], dimension: int) -> dict[int, float]:
    """Sparse, fixed-dimension features with 24 explicit numeric coordinates.

    Fixed categorical hashing depends on names, never instance IDs or map
    coordinates. Numeric values are scaled to stable approximate ranges.
    The exact feature schema is versioned with every persisted model.
    """
    if dimension <= NUMERIC_FEATURES:
        raise ValueError("Structured combat dimension must exceed 24")
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Expected public combat-entry observation")
    enemies = _objects(combat.get("enemies"))
    deck = _objects(state.get("deck"))
    potions = _objects(state.get("potions"))
    relics = _objects(state.get("relics"))
    hp = _number(state.get("hp"))
    max_hp = max(1.0, _number(state.get("max_hp")))
    enemy_hp = [_number(enemy.get("hp")) for enemy in enemies]
    enemy_block = [_number(enemy.get("block")) for enemy in enemies]
    total_enemy_hp = sum(enemy_hp)
    features = [
        1.0,
        hp / max_hp,
        hp / 100.0,
        max_hp / 100.0,
        _number(state.get("floor")) / 16.0,
        len(enemies) / 5.0,
        total_enemy_hp / 200.0,
        max(enemy_hp, default=0.0) / 200.0,
        sum(enemy_block) / 100.0,
        len(potions) / 4.0,
        len(deck) / 40.0,
        len(relics) / 15.0,
        _number(combat.get("energy")) / 5.0,
        _number(combat.get("turn")) / 10.0,
        _number(combat.get("player_block")) / 50.0,
        len(_objects(combat.get("hand"))) / 10.0,
        _number(combat.get("draw_pile_count")) / 20.0,
        total_enemy_hp / max(1.0, hp) / 10.0,
        _number(state.get("gold")) / 500.0,
        _number(state.get("act")) / 3.0,
        len(_objects(combat.get("player_powers"))) / 10.0,
        float(hp <= 20),
        float(hp <= 40),
        float(max(enemy_hp, default=0.0) >= 150),
    ]
    # Prevent arbitrarily large public counters from overwhelming gradients.
    sparse = {
        i: max(-8.0, min(8.0, value)) for i, value in enumerate(features)
        if value != 0.0
    }
    tokens: Counter[str] = Counter()
    for enemy in enemies:
        tokens[f"enemy:{enemy.get('enemy_id')}"] += 1
        tokens[f"move:{enemy.get('move_id')}"] += 1
    for card in deck:
        tokens[f"deck:{card.get('card_id')}"] += 1
        if _number(card.get("upgrade_level")) > 0:
            tokens[f"upgraded:{card.get('card_id')}"] += 1
    for potion in potions:
        tokens[f"potion:{potion.get('potion_id')}"] += 1
    for relic in relics:
        tokens[f"relic:{relic.get('relic_id')}"] += 1
    tokens[f"boss:{state.get('act_one_boss_encounter_id')}"] += 1
    for token, count in tokens.items():
        digest = hashlib.sha256(("combat-entry-v2:" + token).encode()).digest()
        index = NUMERIC_FEATURES + (
            int.from_bytes(digest[:8], "big") % (dimension - NUMERIC_FEATURES)
        )
        sparse[index] = min(8.0, sparse.get(index, 0.0) + min(count, 5) / 5.0)
    return sparse
