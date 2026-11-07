"""H26-D: read-only source-defined conditional gross-damage distributions.

The exact Native-grid intent law is combined with source move effects and
public powers. Damage is conditional on the *supplied* public state and its
power/relic modifiers remaining unchanged until each future intent executes,
and on the currently displayed intent executing normally. The result is
pre-Block GROSS damage, not future HP loss or an unconditional lethal risk.
It never reads a hidden canonical state or samples an intent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sts2_ai.training.h26_public_intent_law import (
    PublicNextIntentLaw,
    next_intent_law,
)


@dataclass(frozen=True)
class ConditionalMoveDamage:
    move_id: str
    numerator: int
    gross_damage: int
    hits: int


@dataclass(frozen=True)
class ConditionalDamageRisk:
    status: str
    reason: str
    denominator: int
    outcomes: tuple[ConditionalMoveDamage, ...]
    expected_gross_damage: float | None
    variance_gross_damage: float | None
    probability_gross_at_least_hp: float | None

    @property
    def available(self) -> bool:
        return self.status == "CONDITIONAL_EXACT"


def _unavailable(reason: str) -> ConditionalDamageRisk:
    return ConditionalDamageRisk("UNAVAILABLE", reason, 0, (), None, None, None)


def _positive_power_stacks(raw: Any) -> list[tuple[str, int]] | None:
    if not isinstance(raw, list):
        return None
    result: list[tuple[str, int]] = []
    for item in raw:
        if (not isinstance(item, dict)
                or not isinstance(item.get("power_id"), str)
                or type(item.get("stacks")) is not int):
            return None
        if item["stacks"] > 0:
            result.append((item["power_id"], item["stacks"]))
    return result


def _effective_enemy_powers(
    enemy: dict[str, Any], catalog: dict[str, Any],
) -> list[tuple[str, int]] | None:
    result = _positive_power_stacks(enemy.get("powers"))
    statuses = enemy.get("statuses")
    if result is None or not isinstance(statuses, dict):
        return None
    lookup = catalog["enemy_status_power_ids"]
    for status_id, stacks in statuses.items():
        if type(stacks) is not int or stacks < 0:
            return None
        if stacks == 0:
            continue
        power_id = lookup.get(status_id)
        if power_id is None:
            # The source does not certify how this legacy status changes damage.
            return None
        if any(power == power_id for power, _ in result):
            # A mixed canonical/legacy projection would double-count stacks.
            return None
        result.append((power_id, stacks))
    return result


def _scaled(base: int, deltas: list[dict[str, Any]], ascension: int) -> int:
    return base + sum(
        item["delta"] for item in deltas
        if ascension >= item["min_ascension"]
    )


def _gross_move_damage(
    move: dict[str, Any], *,
    state: dict[str, Any], enemy: dict[str, Any],
    catalog: dict[str, Any],
    powers: list[tuple[str, int]], player_powers: list[tuple[str, int]],
    power_rules: dict[str, dict[str, Any]],
    ascension: int,
    executed_moves: tuple[str, ...] | None,
) -> tuple[int, int] | None:
    act = state.get("act")
    if type(act) is not int or act < 1:
        return None
    damage = hits = 0
    effects = move.get("effects")
    if not isinstance(effects, list):
        return None
    if any(power_id == catalog.get("surrounded_power_id")
           for power_id, _ in player_powers):
        # The source needs the player's unprojected facing stored value.
        return None
    relic_red = {
        r["relic_id"]: r["weak_attack_damage_reduction_percent"]
        for r in catalog["damage_relics"]
    }
    bonus_weak = 0
    for relic in state.get("relics", []):
        if not isinstance(relic, dict) or not isinstance(relic.get("relic_id"), str):
            return None
        if not relic.get("is_melted", False):
            bonus_weak += relic_red.get(relic["relic_id"], 0)
    for effect in effects:
        if (not isinstance(effect, dict)
                or effect.get("earlier_non_damage_effect")
                or effect.get("use_stored_enemy_damage")
                or not effect.get("is_attack", False)):
            return None
        power_condition = effect.get("owner_power_stack_condition_id")
        if power_condition:
            stacks = sum(n for pid, n in powers if pid == power_condition)
            if (effect.get("owner_power_stacks_less_than") is not None
                    and stacks >= effect["owner_power_stacks_less_than"]):
                continue
            if (effect.get("owner_power_stacks_at_least") is not None
                    and stacks < effect["owner_power_stacks_at_least"]):
                continue
        if (effect["extra_amount_per_prior_move_use"] != 0
                and executed_moves is None):
            return None
        prior_uses = (
            executed_moves.count(move["move_id"]) +
            int(enemy.get("move_id") == move["move_id"])
            if executed_moves is not None else 0
        )
        amount = _scaled(
            effect["amount"], effect["ascension_deltas"], ascension,
        ) + (effect["amount_per_act"] * (act - 1)
             + effect["extra_amount_per_prior_move_use"] * prior_uses)
        repetitions = _scaled(
            effect["repetitions"],
            effect["repetition_ascension_deltas"], ascension,
        )
        if repetitions < 0:
            return None
        extra_power = effect.get("extra_amount_from_owner_power_id")
        if extra_power:
            amount += sum(n for pid, n in powers if pid == extra_power)
        amount += sum(
            power_rules[pid]["enemy_attack_damage_bonus_per_stack"] * stacks
            for pid, stacks in powers
            if pid in power_rules
        )
        if any(pid not in power_rules for pid, _ in powers + player_powers):
            return None
        for pid, stacks in powers:
            # Enemy weak/status outgoing multipliers apply when stacks > 0.
            rule = power_rules[pid]
            num = rule["enemy_outgoing_damage_numerator"]
            den = rule["enemy_outgoing_damage_denominator"]
            if den <= 0:
                return None
            if (pid == catalog["enemy_status_power_ids"].get("proto.status.weak")
                    and bonus_weak):
                num = max(0, 100 * num - bonus_weak * den)
                den *= 100
            # In the source, these multipliers apply to enemy effect statuses
            # only. Ordinary powers have their outgoing fraction at identity.
            if num != den:
                amount = amount * num // den
        for pid, stacks in powers:
            rule = power_rules[pid]
            den = rule["enemy_outgoing_powered_attack_denominator"]
            if den <= 0:
                return None
            amount = (
                amount * rule["enemy_outgoing_powered_attack_numerator"] // den
            )
        for pid, stacks in player_powers:
            rule = power_rules[pid]
            den = rule["player_incoming_attack_damage_denominator"]
            if den <= 0:
                return None
            amount = amount * rule["player_incoming_attack_damage_numerator"] // den
        tainted = catalog.get("tainted_power_id")
        amount += sum(n for pid, n in player_powers if pid == tainted)
        caps = [
            power_rules[pid]["player_incoming_damage_cap"]
            for pid, _ in player_powers
            if power_rules[pid]["player_incoming_damage_cap"] > 0
        ]
        if caps:
            amount = min(amount, min(caps))
        damage += max(0, amount) * repetitions
        hits += repetitions
    return damage, hits


def conditional_next_gross_damage(
    state: dict[str, Any], enemy: dict[str, Any],
    catalog: dict[str, Any], *,
    ascension: int,
    executed_moves: tuple[str, ...] | None = None,
    intent_law: PublicNextIntentLaw | None = None,
) -> ConditionalDamageRisk:
    """Exact weighted source gross damage under the current public modifiers.

    Use after the public state is observed (including after a card/potion
    action). Any change in HP, powers or enemy formation requires recomputing
    the result. Prospective values for unexecuted player actions are NOT
    computed by this API.
    """
    if type(ascension) is not int or ascension < 0:
        raise ValueError("A known run ascension is required for damage scaling")
    if catalog.get("schema") != "sts2-h26-public-intent-rules-v3":
        return _unavailable("damage_catalog_v3_required")
    if not isinstance(state.get("combat"), dict) or not isinstance(enemy, dict):
        return _unavailable("public_combat_state_required")
    enemies = state["combat"].get("enemies")
    if not isinstance(enemies, list):
        return _unavailable("public_enemy_formation_required")
    law = intent_law if intent_law is not None else next_intent_law(
        enemy, catalog, executed_moves=executed_moves,
        public_enemies=enemies,
    )
    if not law.available:
        return _unavailable("intent_law_" + law.reason)
    if not isinstance(enemy.get("move_id"), str):
        return _unavailable("visible_current_intent_required")
    powers = _effective_enemy_powers(enemy, catalog)
    player_powers = _positive_power_stacks(
        state["combat"].get("player_powers"),
    )
    if powers is None or player_powers is None:
        return _unavailable("public_power_projection_incomplete")
    power_rules = {r["power_id"]: r for r in catalog["damage_powers"]}
    enemy_rule = next(
        (r for r in catalog["enemies"]
         if r["enemy_id"] == enemy.get("enemy_id")), None,
    )
    if enemy_rule is None:
        return _unavailable("enemy_source_rules_missing")
    damage_moves = enemy_rule.get("move_damage")
    if not isinstance(damage_moves, list):
        return _unavailable("enemy_damage_effects_missing")
    values: list[ConditionalMoveDamage] = []
    for move_id, weight in law.outcomes:
        candidates = [m for m in damage_moves if m["move_id"] == move_id]
        if not candidates:
            return _unavailable("missing_move_damage_definition")
        # A repeated source move ID is safe if every matching move describes
        # identical damage effects; otherwise its identity is ambiguous.
        if any(item["effects"] != candidates[0]["effects"] for item in candidates):
            return _unavailable("duplicate_move_damage_ambiguous")
        resolved = _gross_move_damage(
            candidates[0], state=state, enemy=enemy, catalog=catalog,
            powers=powers, player_powers=player_powers,
            power_rules=power_rules, ascension=ascension,
            executed_moves=executed_moves,
        )
        if resolved is None:
            return _unavailable("dynamic_or_unobserved_damage_modifier")
        values.append(ConditionalMoveDamage(
            move_id, weight, resolved[0], resolved[1],
        ))
    if not values or sum(v.numerator for v in values) != law.denominator:
        raise ValueError("Next-intent damage probability mass is invalid")
    expectation = sum(
        x.numerator * x.gross_damage for x in values
    ) / law.denominator
    variance = sum(
        x.numerator * (x.gross_damage-expectation)**2 for x in values
    ) / law.denominator
    hp = state.get("hp")
    if type(hp) is not int:
        return _unavailable("public_player_hp_missing")
    threshold = sum(
        x.numerator for x in values if x.gross_damage >= hp
    ) / law.denominator
    if not all(math.isfinite(x) for x in (expectation, variance, threshold)):
        raise ValueError("Nonfinite public damage-risk feature")
    return ConditionalDamageRisk(
        "CONDITIONAL_EXACT",
        "frozen_public_modifiers_before_next_enemy_action_pre_block",
        law.denominator, tuple(values), expectation, variance, threshold,
    )
