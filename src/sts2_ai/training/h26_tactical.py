"""Public-only H26 damage features and bounded combat reward shaping.

The projection is a *known damage lower bound*, not a guaranteed incoming HP
loss: heterogeneous attacks, status effects and concealed intents remain
unknown. Next-turn law is a separate emulator interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sts2_ai.models.shared_game_objects import (
    IDENTITY_BUCKETS,
    KINDS,
    ZONES,
    ObjectToken,
    _bucket,
    _numbers,
)
from sts2_ai.training.h24_joint_act_model import H24EncodedDecision


@dataclass(frozen=True)
class VisibleIncomingThreat:
    known_damage: int
    known_hits: int
    player_block: int
    hp: int
    unknown_intents: int
    living_enemies: int

    @property
    def known_hp_loss(self) -> int:
        return max(0, self.known_damage - self.player_block)

    @property
    def known_lethal(self) -> bool:
        return self.known_hp_loss >= self.hp and self.hp > 0

    @property
    def complete(self) -> bool:
        return self.unknown_intents == 0


def visible_incoming_threat(state: dict[str, Any]) -> VisibleIncomingThreat:
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Incoming threat requires public combat observation")
    hp = state.get("hp")
    block = combat.get("player_block")
    enemies = combat.get("enemies")
    if (type(hp) is not int or hp < 0 or type(block) is not int
            or block < 0 or not isinstance(enemies, list)):
        raise ValueError("Malformed public combat HP, block or enemies")
    total = hits = unknown = living = 0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            raise ValueError("Malformed public enemy")
        enemy_hp = enemy.get("hp")
        if type(enemy_hp) is not int:
            raise ValueError("Malformed public enemy HP")
        if enemy_hp <= 0:
            continue
        living += 1
        damage = enemy.get("intent_damage")
        count = enemy.get("intent_hits")
        if type(damage) is int and type(count) is int and damage >= 0 and count > 0:
            total += damage * count
            hits += count
        else:
            unknown += 1  # missing includes nonattacks AND unquantified attacks
    return VisibleIncomingThreat(total, hits, block, hp, unknown, living)


def attach_visible_threat(
    encoded: H24EncodedDecision, state: dict[str, Any],
) -> H24EncodedDecision:
    """Add a distinct tactical policy token; keep value-neutral encoding intact."""
    if not isinstance(state.get("combat"), dict):
        return encoded
    threat = visible_incoming_threat(state)
    values, masks = _numbers({
        "intent_damage": threat.known_damage,
        "intent_hits": threat.known_hits,
        "block": threat.player_block,
        "hp": threat.hp,
        "stacks": threat.unknown_intents,
        "count": threat.living_enemies,
    })
    marker = ObjectToken(
        kind=KINDS.index("status"),
        identity=_bucket("h26.public.visible-incoming-threat.v1", IDENTITY_BUCKETS),
        zone=ZONES.index("player"),
        numeric=values,
        observed=masks,
    )
    if len(encoded.objects) >= 256:
        raise ValueError("H26 tactical token exceeds public object cap")
    return H24EncodedDecision(
        encoded.schema, encoded.phase,
        encoded.objects + (marker,), encoded.actions, encoded.relations,
        encoded.neutral_value,
    )


def combat_reward_potential(
    frame: dict[str, Any], *, initial_enemy_hp: dict[int, int],
    hp_reference: int,
) -> float:
    """Bounded public combat potential, independent of hidden RNG and PUCT."""
    if hp_reference <= 0 or type(frame.get("hp")) is not int:
        raise ValueError("Missing valid HP")
    hp_frac = min(1.0, max(0.0, frame["hp"] / hp_reference))
    combat = frame.get("combat")
    if not isinstance(combat, dict):
        return 0.0
    remaining = dict(initial_enemy_hp)
    for enemy in combat.get("enemies", []):
        if (isinstance(enemy, dict)
                and type(enemy.get("instance_id")) is int
                and enemy["instance_id"] in remaining
                and type(enemy.get("hp")) is int):
            remaining[enemy["instance_id"]] = max(0, enemy["hp"])
    total = sum(initial_enemy_hp.values())
    damage_frac = 1.0 - sum(remaining.values()) / total if total else 0.0
    return 0.1 * hp_frac + 0.1 * min(1.0, max(0.0, damage_frac))


def combat_dense_rewards(
    frames: list[dict[str, Any]], *, gamma: float = 0.99,
) -> list[float]:
    """Discount-consistent potential-based per-step reward differences."""
    if len(frames) < 2 or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Invalid H26 combat frames or discount")
    first = frames[0]
    combat = first.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("First H26 frame must be combat")
    initial = {
        e["instance_id"]: max(0, e["hp"])
        for e in combat.get("enemies", [])
        if isinstance(e, dict) and type(e.get("instance_id")) is int
        and type(e.get("hp")) is int and e["hp"] > 0
        and not e.get("is_minion", False)
    }
    hp_ref = first.get("max_hp")
    if type(hp_ref) is not int or hp_ref <= 0:
        raise ValueError("Invalid initial maximum HP")
    phi = [
        combat_reward_potential(
            frame, initial_enemy_hp=initial, hp_reference=hp_ref,
        )
        for frame in frames
    ]
    return [gamma * phi[i + 1] - phi[i] for i in range(len(phi) - 1)]
