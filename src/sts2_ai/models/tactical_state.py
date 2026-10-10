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
RELATIONAL_NUMERIC_COORDINATES = 36
RELATIONAL_TACTICAL_STATE_SCHEMA = "sts2-tactical-public-state-v5-relational"


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
    state: dict[str, Any], dimension: int, *,
    relational: bool = False,
) -> dict[int, float]:
    """Encode visible tactical state, optionally preserving enemy/intent binding.

    The default is the unmodified v4 feature schema for old models. v5 adds
    relational enemy tokens and optional player-visible damage/hit numerics.
    Nothing is inferred from game content, private RNG, or future moves.
    """
    reserved = (
        RELATIONAL_NUMERIC_COORDINATES if relational else NUMERIC_COORDINATES
    )
    if dimension <= reserved:
        raise ValueError(
            f"Tactical state dimension must exceed {reserved}"
        )
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
    if relational:
        # These fields are OPTIONAL. Current native emulator observations
        # expose move_id but not intent_damage/intent_hits. Never reconstruct
        # missing threat values from the private enemy move definitions.
        # intent_damage denotes visible per-hit damage; intent_hits denotes
        # visible hit count, with a single hit as the fallback.
        public_attacks: list[tuple[float, float]] = []
        for enemy in enemies:
            if not isinstance(enemy.get("move_id"), str):
                continue  # A hidden/unknown intent must stay hidden.
            raw_damage = enemy.get("intent_damage")
            raw_hits = enemy.get("intent_hits")
            if (
                not isinstance(raw_damage, (int, float))
                or isinstance(raw_damage, bool)
                or not math.isfinite(raw_damage)
                or raw_damage < 0
            ):
                continue
            hits = (
                float(raw_hits)
                if isinstance(raw_hits, int) and not isinstance(raw_hits, bool)
                and 1 <= raw_hits <= 100
                else 1.0
            )
            public_attacks.append((float(raw_damage), hits))
        values.extend([
            sum(d * h for d, h in public_attacks) / 100.0,
            max((d * h for d, h in public_attacks), default=0.0) / 100.0,
            sum(h for _, h in public_attacks) / 10.0,
            len(public_attacks) / 5.0,
        ])
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

    if relational:
        for enemy in enemies:
            enemy_id = enemy.get("enemy_id")
            if not isinstance(enemy_id, str) or not enemy_id:
                continue
            # The announced move belongs to THIS enemy, not to a global bag
            # of announced moves. Including it in the same token is crucial
            # for untargeted Defend, draw, and potion actions.
            move_id = enemy.get("move_id")
            move = move_id if isinstance(move_id, str) and move_id else "unknown"
            prefix = f"enemy={enemy_id}|move={move}"
            tokens[f"relation:{prefix}"] += 1
            hp_value = _number(enemy.get("hp"))
            block_value = _number(enemy.get("block"))
            tokens[f"relation:{prefix}|hp10={max(0, int(hp_value)) // 10}"] += 1
            tokens[f"relation:{prefix}|block10={max(0, int(block_value)) // 10}"] += 1
            # Powers/statuses must remain associated with their owner.
            for power in _objects(enemy.get("powers")):
                power_id = power.get("power_id")
                if isinstance(power_id, str) and power_id:
                    tokens[f"relation:{prefix}|power={power_id}"] += 1
            statuses = enemy.get("statuses")
            if isinstance(statuses, dict):
                for status_id, stacks in statuses.items():
                    if isinstance(status_id, str) and _number(stacks) > 0:
                        tokens[f"relation:{prefix}|status={status_id}"] += 1

    boss = state.get("act_one_boss_encounter_id")
    if isinstance(boss, str) and boss:
        tokens[f"boss:{boss}"] += 1
    for token, count in tokens.items():
        digest = hashlib.sha256(
            (("tactical-state-v5:" if relational else "tactical-state-v4:")
             + token).encode("utf-8")
        ).digest()
        index = reserved + (
            int.from_bytes(digest[:8], "big") % (dimension - reserved)
        )
        features[index] = min(
            8.0, features.get(index, 0.0) + min(count, 8) / 8.0
        )
    return features


def relational_tactical_state_features(
    state: dict[str, Any], dimension: int,
) -> dict[int, float]:
    """v5 public-only features retaining each enemy's announced move."""
    return tactical_state_features(state, dimension, relational=True)
