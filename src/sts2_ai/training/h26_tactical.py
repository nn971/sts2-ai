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
    RelationToken,
    _bucket,
    _numbers,
)
from sts2_ai.training.h16_episode_metrics import _is_minion
from sts2_ai.training.h21_macro_offer_codec import (
    H21_OBJECT_LIMIT,
    VISIBLE_VALUE,
)
from sts2_ai.training.h24_joint_act_model import H24EncodedDecision
from sts2_ai.training.h26_damage_risk import conditional_next_gross_damage
from sts2_ai.training.h26_public_intent_law import summarize_next_intents


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
    *, intent_catalog: dict[str, Any] | None = None,
    intent_histories: dict[int, tuple[str, ...]] | None = None,
    damage_ascension: int | None = None,
) -> H24EncodedDecision:
    """Add current threat and optional source-grid next-intent support tokens.

    Candidate identity encodes the publicly known move ID, its observed
    probability is a numeric feature, and ownership joins it to the
    corresponding visible enemy. Existing pointers and neutral value encoding
    remain unchanged.
    """
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
    # H21 has an independent 1,536-object limit (H7's legacy cap is 256).
    # H26 contributes exactly one extra policy object. Appending preserves
    # all H21 owner indices, relation endpoints and action pointers.
    if len(encoded.objects) > H21_OBJECT_LIMIT:
        raise ValueError("Malformed H21 decision already exceeds its object cap")
    extra: list[ObjectToken] = [marker]
    edges: list[RelationToken] = []
    if intent_catalog is not None:
        enemies = state["combat"].get("enemies")
        if not isinstance(enemies, list):
            raise ValueError("Malformed public enemy list for H26 intents")
        # H21 combat augmentation certifies the per-enemy token correspondence.
        enemy_indices = [
            index for index, token in enumerate(encoded.objects)
            if token.kind == KINDS.index("enemy")
            and token.zone == ZONES.index("enemy")
        ]
        if len(enemy_indices) != len(enemies):
            raise ValueError("H26 enemy law ownership differs from H21 codec")
        for owner, enemy, law in zip(
            enemy_indices, enemies,
            summarize_next_intents(
                enemies, intent_catalog, histories=intent_histories,
            ),
            strict=True,
        ):
            risk = (
                conditional_next_gross_damage(
                    state, enemy, intent_catalog,
                    ascension=damage_ascension,
                    executed_moves=(
                        intent_histories.get(enemy.get("instance_id"))
                        if intent_histories is not None else None
                    ),
                    intent_law=law,
                )
                if damage_ascension is not None
                else None
            )
            by_move = (
                {outcome.move_id: outcome for outcome in risk.outcomes}
                if risk is not None and risk.available else {}
            )
            # Explicitly mark unsupported/ambiguous law cases. No guessed
            # zero threat or falsely uniform distribution reaches the model.
            outcomes = (
                law.outcomes if law.available
                else (("UNAVAILABLE", 0),)
            )
            for move_id, numerator in outcomes:
                if law.available:
                    fields = {"count": 10.0 * numerator / law.denominator}
                    damage = by_move.get(move_id)
                    if damage is not None:
                        fields["intent_damage"] = damage.gross_damage
                        fields["intent_hits"] = damage.hits
                    numeric, observed = _numbers(fields)
                else:
                    numeric, observed = _numbers({"count": 0})
                pos = len(encoded.objects) + len(extra)
                extra.append(ObjectToken(
                    kind=KINDS.index("status"),
                    identity=_bucket(
                        "h26.next-intent.v1:" + move_id,
                        IDENTITY_BUCKETS,
                    ),
                    zone=ZONES.index("enemy"),
                    numeric=numeric,
                    observed=observed,
                    owner=owner,
                ))
                edges.append(RelationToken(VISIBLE_VALUE, owner, pos))
            if risk is not None:
                # A separate summary token distinguishes the expected gross
                # pre-Block threat from current displayed incoming damage.
                # Its two damage scales are mean and standard deviation;
                # "count" encodes 10 times the gross>=HP probability.
                if risk.available:
                    numeric, observed = _numbers({
                        "intent_damage": risk.expected_gross_damage,
                        "intent_base_damage": math.sqrt(
                            risk.variance_gross_damage,
                        ),
                        "count": 10.0 * risk.probability_gross_at_least_hp,
                    })
                    identity = "h26.next-gross-risk.v1"
                else:
                    numeric, observed = _numbers({})
                    identity = "h26.next-gross-unavailable.v1"
                pos = len(encoded.objects) + len(extra)
                extra.append(ObjectToken(
                    kind=KINDS.index("status"),
                    identity=_bucket(identity, IDENTITY_BUCKETS),
                    zone=ZONES.index("enemy"),
                    numeric=numeric,
                    observed=observed,
                    owner=owner,
                ))
                edges.append(RelationToken(VISIBLE_VALUE, owner, pos))
    return H24EncodedDecision(
        encoded.schema, encoded.phase,
        encoded.objects + tuple(extra), encoded.actions,
        encoded.relations + tuple(edges),
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
        and not _is_minion(e)
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
