"""Exact source-grid next-intent laws conditionally inferred from public moves.

The version-pinned catalog is exported by the emulator: it contains only
public game rules, NEVER a canonical RunState or private AI-state counters.
A forecast is conditional on the displayed current intent executing normally
and the enemy surviving to its next selection. Ambiguous states are UNAVAILABLE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SCHEMA = "sts2-h26-public-intent-rules-v1"


@dataclass(frozen=True)
class PublicNextIntentLaw:
    status: str
    reason: str
    denominator: int
    outcomes: tuple[tuple[str, int], ...]

    @property
    def available(self) -> bool:
        return self.status == "CONDITIONAL_EXACT"


def validate_public_intent_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate integer grid weights and move identities before training."""
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("Unexpected H26 public intent rule schema")
    denominator = raw.get("probability_denominator")
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("Invalid Native NextInt probability denominator")
    weights = raw.get("native_next_int_weights")
    entries = raw.get("enemies")
    if not isinstance(entries, list) or not isinstance(weights, dict):
        raise ValueError("H26 intent catalog requires enemies and Native grid masses")
    seen = set()
    maximum = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed enemy rule")
        eid, moves = entry.get("enemy_id"), entry.get("move_ids")
        limits = entry.get("max_consecutive_uses")
        policy = entry.get("move_policy")
        if (not isinstance(eid, str) or not eid or eid in seen
                or not isinstance(moves, list) or not moves
                or any(not isinstance(m, str) or not m for m in moves)
                or len(moves) != len(set(moves))
                or not isinstance(limits, list)
                or len(limits) != len(moves)
                or any(type(x) is not int or x < 0 for x in limits)
                or policy not in ("SequentialLoop", "UniformRandomAfterOpener",
                                  "StateMachine")):
            raise ValueError("Invalid H26 enemy rule identity or policy")
        seen.add(eid)
        maximum = max(maximum, len(moves))
        opener = entry.get("opening_move_indices")
        if (not isinstance(opener, list) or not opener
                or any(type(x) is not int or not 0 <= x < len(moves)
                       for x in opener)
                or any(type(entry.get(x)) is not int
                       or not 0 <= entry[x] < len(moves)
                       for x in ("move_loop_start_index",
                                 "random_move_pool_start_index"))):
            raise ValueError("Invalid H26 enemy move indices")
    for n in range(1, maximum + 1):
        mass = weights.get(str(n))
        if (not isinstance(mass, list) or len(mass) != n
                or any(type(w) is not int or w <= 0 for w in mass)
                or sum(mass) != denominator):
            raise ValueError("Native NextInt grid masses do not sum to one")
    return raw


def _unavailable(reason: str) -> PublicNextIntentLaw:
    return PublicNextIntentLaw("UNAVAILABLE", reason, 0, ())


def _mass(
    indices: list[int], moves: list[str], catalog: dict[str, Any],
) -> PublicNextIntentLaw:
    if not indices:
        return _unavailable("no_valid_move_support")
    weights = catalog["native_next_int_weights"][str(len(indices))]
    denominator = catalog["probability_denominator"]
    items: dict[str, int] = {}
    for index, count in zip(indices, weights, strict=True):
        items[moves[index]] = items.get(moves[index], 0) + count
    result = PublicNextIntentLaw(
        "CONDITIONAL_EXACT", "current_intent_executes_normally",
        denominator, tuple(sorted(items.items())),
    )
    if sum(mass for _, mass in result.outcomes) != denominator:
        raise ValueError("Malformed H26 intent mass total")
    return result


def next_intent_law(
    enemy: dict[str, Any], catalog: dict[str, Any],
) -> PublicNextIntentLaw:
    """Return a certified law only when the visible intent uniquely conditions.

    The current move is known; its history/AI state is NOT assumed visible.
    For a random pool move with a maximum repeat limit greater than one,
    the unknown consecutive-use counter may change eligibility, so the law
    is withheld. Opening-stage ambiguity is treated the same way.
    """
    if not isinstance(enemy, dict) or not isinstance(catalog, dict):
        raise ValueError("Invalid H26 public-law inputs")
    if type(enemy.get("hp")) is not int or enemy["hp"] <= 0:
        return _unavailable("enemy_not_alive")
    eid, current = enemy.get("enemy_id"), enemy.get("move_id")
    if not isinstance(eid, str) or not isinstance(current, str):
        return _unavailable("current_intent_unavailable")
    rule = next(
        (e for e in catalog["enemies"] if e["enemy_id"] == eid), None,
    )
    if rule is None:
        return _unavailable("unknown_enemy_rule")
    moves = rule["move_ids"]
    if current not in moves:
        return _unavailable("unrecognized_visible_intent")
    current_index = moves.index(current)
    policy = rule["move_policy"]
    if policy == "SequentialLoop":
        loop = rule["move_loop_start_index"]
        next_index = (
            loop if current_index == len(moves) - 1 else current_index + 1
        )
        return _mass([next_index], moves, catalog)
    if policy == "StateMachine":
        return (
            _mass([0], moves, catalog) if len(moves) == 1
            else _unavailable("state_machine_public_history_needed")
        )
    if policy != "UniformRandomAfterOpener":
        return _unavailable("unsupported_enemy_policy")
    opening = rule["opening_move_indices"]
    pool_start = rule["random_move_pool_start_index"]

    # A visible move may have been drawn from either the opening sequence
    # or the random pool. A unique opener-only move determines its stage.
    in_opener = current_index in opening
    in_random = current_index >= pool_start
    if in_opener and in_random:
        return _unavailable("opener_or_random_stage_ambiguous")
    if in_opener:
        appearances = [i for i, m in enumerate(opening) if m == current_index]
        if len(appearances) != 1:
            return _unavailable("repeated_opener_move_ambiguous")
        j = appearances[0]
        if j + 1 < len(opening):
            return _mass([opening[j + 1]], moves, catalog)
        # The last opener will execute before the first random draw.
    elif not in_random:
        return _unavailable("visible_move_outside_random_pool")

    limits = rule["max_consecutive_uses"]
    # After current intent executes, all eligibility restrictions concern
    # only the just-played move. For max=1 it must be excluded, max=0
    # means unlimited. max>=2 needs the public consecutive-use history.
    if limits[current_index] > 1:
        return _unavailable("consecutive_move_history_needed")
    eligible = [
        i for i in range(pool_start, len(moves))
        if i != current_index or limits[i] == 0
    ]
    return _mass(eligible, moves, catalog)


def summarize_next_intents(
    enemies: list[dict[str, Any]], catalog: dict[str, Any],
) -> tuple[PublicNextIntentLaw, ...]:
    return tuple(next_intent_law(enemy, catalog) for enemy in enemies)
