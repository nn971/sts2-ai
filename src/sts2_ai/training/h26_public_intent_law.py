"""Exact source-grid next-intent laws conditionally inferred from public moves.

The version-pinned catalog is exported by the emulator: it contains only
public game rules, NEVER a canonical RunState or private AI-state counters.
A forecast is conditional on the displayed current intent executing normally
and the enemy surviving to its next selection. Ambiguous states are UNAVAILABLE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA = "sts2-h26-public-intent-rules-v2"
PREVIOUS_SCHEMA = "sts2-h26-public-intent-rules-v1"


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
    if not isinstance(raw, dict) or raw.get("schema") not in (
        SCHEMA, PREVIOUS_SCHEMA,
    ):
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
                or not isinstance(limits, list)
                or len(limits) != len(moves)
                or any(type(x) is not int or x < 0 for x in limits)
                or policy not in ("SequentialLoop", "UniformRandomAfterOpener",
                                  "StateMachine")):
            raise ValueError("Invalid H26 enemy rule identity or policy")
        seen.add(eid)
        maximum = max(maximum, len(moves))
        ai = entry.get("ai")
        if policy == "StateMachine" and raw["schema"] == SCHEMA:
            if (not isinstance(ai, dict)
                    or not isinstance(ai.get("initial_state_id"), str)
                    or not isinstance(ai.get("states"), list)):
                raise ValueError("V2 state machines require public source AI graph")
        if ai is not None:
            if not isinstance(ai, dict):
                raise ValueError("Malformed public AI graph")
            states = ai.get("states")
            if not isinstance(states, list):
                raise ValueError("Malformed public AI states")
            state_ids = set()
            for state in states:
                if (not isinstance(state, dict)
                        or not isinstance(state.get("id"), str)
                        or state["id"] in state_ids
                        or state.get("kind") not in
                        ("Move", "Random", "Conditional")
                        or not isinstance(state.get("branches"), list)
                        or not isinstance(state.get("conditional_branches"), list)):
                    raise ValueError("Invalid public AI state")
                state_ids.add(state["id"])
                if state["kind"] == "Random":
                    branches = state["branches"]
                    total = sum(
                        b["weight"] for b in branches
                        if isinstance(b, dict)
                        and type(b.get("weight")) is int
                        and b["weight"] > 0
                    )
                    maximum = max(maximum, min(total, 256))
            if ai["initial_state_id"] not in state_ids:
                raise ValueError("Unknown initial public AI state")
        opener = entry.get("opening_move_indices")
        if (not isinstance(opener, list) or not opener
                or any(type(x) is not int or not 0 <= x < len(moves)
                       for x in opener)
                or any(type(entry.get(x)) is not int
                       or not 0 <= entry[x] < len(moves)
                       for x in ("move_loop_start_index",
                                 "random_move_pool_start_index"))):
            raise ValueError("Invalid H26 enemy move indices")
    # V2 can expose ticket bounds above the move pool size. They are
    # required only for eligible weighted branches during evaluation.
    for n in range(1, max(len(e["move_ids"]) for e in entries) + 1):
        mass = weights.get(str(n))
        if (not isinstance(mass, list) or len(mass) != n
                or any(type(w) is not int or w <= 0 for w in mass)
                or sum(mass) != denominator):
            raise ValueError("Native NextInt grid masses do not sum to one")
    for bound, mass in weights.items():
        if (not bound.isdigit() or int(bound) <= 0
                or not isinstance(mass, list) or len(mass) != int(bound)
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



def _weighted_mass(
    candidates: list[tuple[str, int]], catalog: dict[str, Any],
) -> PublicNextIntentLaw:
    """Native integer-ticket weights, respecting ordered branch intervals."""
    total = sum(weight for _, weight in candidates)
    weights = catalog["native_next_int_weights"].get(str(total))
    if weights is None:
        return _unavailable("native_weighted_ticket_bound_unavailable")
    items: dict[str, int] = {}
    start = 0
    for move, raw in candidates:
        items[move] = items.get(move, 0) + sum(weights[start:start + raw])
        start += raw
    if start != total:
        raise ValueError("Incorrect Native weighted branch interval")
    return PublicNextIntentLaw(
        "CONDITIONAL_EXACT",
        "current_intent_executes_normally_public_conditions_unchanged",
        catalog["probability_denominator"], tuple(sorted(items.items())),
    )


def _branch_eligibility(
    branch: dict[str, Any], move_id: str, current: str,
    executed_moves: tuple[str, ...] | None,
) -> bool | None:
    if branch.get("requires_available_summon"):
        return None  # source additionally uses private coordinated counters
    repeat = branch.get("repeat_rule")
    if repeat == "CanRepeatForever":
        return True
    if repeat == "CannotRepeat":
        return move_id != current
    if repeat == "CanRepeatXTimes":
        maximum = branch.get("max_times")
        if type(maximum) is not int or maximum <= 0:
            return False
        if move_id != current:
            return True
        if maximum == 1:
            return False
        if executed_moves is None:
            return None
        # The most recent currently announced move is assumed to execute.
        # Previous moves are recorded only when public turns were observed.
        n = 1
        for observed in reversed(executed_moves):
            if observed != current:
                break
            n += 1
        return n < maximum
    if repeat == "UseOnlyOnce":
        if move_id == current:
            return False
        if executed_moves is None:
            return None
        return move_id not in executed_moves
    return None


def _state_machine_law(
    rule: dict[str, Any], current: str,
    catalog: dict[str, Any], *,
    executed_moves: tuple[str, ...] | None = None,
    public_enemies: list[dict[str, Any]] | None = None,
    enemy: dict[str, Any] | None = None,
) -> PublicNextIntentLaw:
    ai = rule.get("ai")
    if not isinstance(ai, dict):
        return _unavailable("state_machine_public_history_needed")
    moves = rule["move_ids"]
    states = ai["states"]
    by_id = {s["id"]: s for s in states}
    matches = [
        s for s in states if s["kind"] == "Move"
        and type(s.get("move_index")) is int
        and 0 <= s["move_index"] < len(moves)
        and moves[s["move_index"]] == current
    ]
    # Different move states may have the same visible move ID. In that
    # case retain only next states uniquely reconstructible from sight.
    next_ids = {s.get("next_state_id") for s in matches}
    if not matches or len(next_ids) != 1:
        return _unavailable("state_machine_current_state_ambiguous")
    state_id = next(iter(next_ids)) or ai["initial_state_id"]
    for _ in range(32):
        state = by_id.get(state_id)
        if state is None:
            return _unavailable("state_machine_invalid_next_state")
        if state["kind"] == "Move":
            index = state.get("move_index")
            if type(index) is not int or not 0 <= index < len(moves):
                return _unavailable("state_machine_invalid_move")
            return _mass([index], moves, catalog)
        if state["kind"] == "Random":
            candidates: list[tuple[str, int]] = []
            for branch in state["branches"]:
                target = by_id.get(branch.get("target_state_id"))
                if target is None or target["kind"] != "Move":
                    return _unavailable("state_machine_random_target_unverified")
                index = target.get("move_index")
                weight = branch.get("weight")
                if (type(index) is not int or not 0 <= index < len(moves)
                        or type(weight) is not int):
                    return _unavailable("state_machine_random_branch_invalid")
                if weight <= 0:
                    continue
                eligible = _branch_eligibility(
                    branch, moves[index], current, executed_moves,
                )
                if eligible is None:
                    return _unavailable("state_machine_branch_history_needed")
                if eligible:
                    candidates.append((moves[index], weight))
            if not candidates:
                return _unavailable("state_machine_no_legal_branch")
            return _weighted_mass(candidates, catalog)
        if state["kind"] == "Conditional":
            branches = state["conditional_branches"]
            selection: str | None = None
            for branch in branches:
                outcome = _public_condition(
                    branch, enemy, public_enemies, executed_moves,
                    current,
                )
                if outcome is None:
                    return _unavailable("state_machine_condition_history_needed")
                if outcome:
                    selection = branch["target_state_id"]
                    break
            if selection is None:
                return _unavailable("state_machine_no_matching_condition")
            state_id = selection
            continue
        return _unavailable("state_machine_unknown_state_kind")
    return _unavailable("state_machine_resolution_depth_exceeded")


def _public_condition(
    branch: dict[str, Any], enemy: dict[str, Any] | None,
    enemies: list[dict[str, Any]] | None,
    executed_moves: tuple[str, ...] | None, current: str,
) -> bool | None:
    """Source graph conditions requiring public *post-execution* state.

    If the input only describes the current (pre-execution) player turn,
    a conditional law assumes those listed public attributes do not
    change before the next intent is committed.
    """
    if enemy is None:
        return None
    kind = branch.get("condition")
    value = branch.get("value")
    if kind in ("MoveUsedFewerThan", "MoveUsedAtLeast"):
        if executed_moves is None or not isinstance(value, str) or ":" not in value:
            return None
        move, count = value.rsplit(":", 1)
        if not count.isdigit():
            return None
        uses = executed_moves.count(move) + int(move == current)
        return uses < int(count) if kind == "MoveUsedFewerThan" else uses >= int(count)
    if kind in ("HpAtLeastHalf", "HpBelowHalf"):
        hp, maximum = enemy.get("hp"), enemy.get("max_hp")
        if type(hp) is not int or type(maximum) is not int or maximum <= 0:
            return None
        half = maximum // 2
        return hp >= half if kind == "HpAtLeastHalf" else hp < half
    if kind in ("HasEnemyPower", "LacksEnemyPower"):
        powers = enemy.get("powers")
        if not isinstance(powers, list) or not isinstance(value, str):
            return None
        present = any(
            isinstance(p, dict) and p.get("power_id") == value
            and type(p.get("stacks")) is int and p["stacks"] > 0
            for p in powers
        )
        return present if kind == "HasEnemyPower" else not present
    if enemies is None or any(not isinstance(x, dict) for x in enemies):
        return None
    living = [e for e in enemies if type(e.get("hp")) is int and e["hp"] > 0]
    if kind == "IsAlone":
        return len(living) == 1
    if kind in ("LivingEnemiesAtMost", "LivingEnemiesGreaterThan"):
        if not isinstance(value, str) or not value.isdigit():
            return None
        return (
            len(living) <= int(value) if kind == "LivingEnemiesAtMost"
            else len(living) > int(value)
        )
    if kind in ("HasLivingEnemyId", "LacksLivingEnemyId"):
        present = any(e.get("enemy_id") == value for e in living)
        return present if kind == "HasLivingEnemyId" else not present
    # Formation position/slot names and off-balance status need a proved
    # player-public source projection; decline rather than guess.
    return None

def next_intent_law(
    enemy: dict[str, Any], catalog: dict[str, Any], *,
    executed_moves: tuple[str, ...] | None = None,
    public_enemies: list[dict[str, Any]] | None = None,
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
    if moves.count(current) != 1:
        return _unavailable("duplicate_visible_move_id")
    current_index = moves.index(current)
    policy = rule["move_policy"]
    if policy == "SequentialLoop":
        loop = rule["move_loop_start_index"]
        next_index = (
            loop if current_index == len(moves) - 1 else current_index + 1
        )
        return _mass([next_index], moves, catalog)
    if policy == "StateMachine":
        return _state_machine_law(
            rule, current, catalog, executed_moves=executed_moves,
            public_enemies=public_enemies, enemy=enemy,
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
    *, histories: dict[int, tuple[str, ...]] | None = None,
) -> tuple[PublicNextIntentLaw, ...]:
    return tuple(
        next_intent_law(
            enemy, catalog, public_enemies=enemies,
            executed_moves=(
                histories.get(enemy.get("instance_id"))
                if histories is not None else None
            ),
        )
        for enemy in enemies
    )


class PublicIntentHistory:
    """Public displayed-intent history, keyed by encounter and enemy instance.

    A move is recorded only on an observed player-turn transition while the
    same enemy survives. Counts assume its announced intent executed normally,
    which is stated explicitly in the resulting conditional law.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._floor: int | None = None
        self._turn: int | None = None
        self._last: dict[int, str] = {}
        self._moves: dict[int, list[str]] = {}

    def observe(self, state: dict[str, Any]) -> dict[int, tuple[str, ...]]:
        combat = state.get("combat")
        if not isinstance(combat, dict):
            self.reset()
            return {}
        floor, turn = state.get("floor"), combat.get("turn")
        if type(floor) is not int or type(turn) is not int:
            self.reset()
            return {}
        if self._floor != floor or (
            self._turn is not None and (turn < self._turn or turn > self._turn + 1)
        ):
            self.reset()
        enemies = combat.get("enemies", [])
        if not isinstance(enemies, list):
            self.reset()
            return {}
        now = {
            enemy["instance_id"]: enemy["move_id"]
            for enemy in enemies if isinstance(enemy, dict)
            and type(enemy.get("instance_id")) is int
            and type(enemy.get("hp")) is int and enemy["hp"] > 0
            and isinstance(enemy.get("move_id"), str)
        }
        if self._turn is not None and turn > self._turn:
            for instance_id, move in self._last.items():
                if instance_id in now:
                    self._moves.setdefault(instance_id, []).append(move)
        for instance_id in now:
            self._moves.setdefault(instance_id, [])
        self._floor, self._turn, self._last = floor, turn, now
        return {k: tuple(self._moves[k]) for k in now}


def public_intent_coverage(
    episodes: list[Any], catalog: dict[str, Any],
) -> dict[str, Any]:
    """Distinct visible turns, with per-enemy exact law coverage and reasons."""
    from sts2_ai.training.h16_joint_act_pilot import public_state

    seen: set[tuple[Any, ...]] = set()
    reasons: dict[str, int] = {}
    by_enemy: dict[str, dict[str, Any]] = {}
    available = total = living = living_available = 0
    for episode in episodes:
        for decision in episode.decisions:
            state = public_state(decision.observation)
            combat = state.get("combat")
            if not isinstance(combat, dict):
                continue
            enemies = combat.get("enemies")
            if not isinstance(enemies, list):
                continue
            history = getattr(decision, "h26_public_intent_history", None)
            for enemy in enemies:
                if not isinstance(enemy, dict):
                    continue
                name = str(enemy.get("enemy_id", "UNKNOWN"))
                key = (
                    episode.seed, state.get("floor"), combat.get("turn"),
                    enemy.get("instance_id"), enemy.get("move_id"),
                )
                if key in seen:
                    continue
                seen.add(key)
                law = next_intent_law(
                    enemy, catalog, public_enemies=enemies,
                    executed_moves=(
                        history.get(enemy.get("instance_id"))
                        if isinstance(history, dict) else None
                    ),
                )
                total += 1
                bucket = by_enemy.setdefault(
                    name, {"total": 0, "available": 0,
                           "living": 0, "living_available": 0,
                           "unavailable_reasons": {}},
                )
                bucket["total"] += 1
                alive = type(enemy.get("hp")) is int and enemy["hp"] > 0
                if alive:
                    living += 1
                    bucket["living"] += 1
                if law.available:
                    available += 1
                    bucket["available"] += 1
                    if alive:
                        living_available += 1
                        bucket["living_available"] += 1
                else:
                    reasons[law.reason] = reasons.get(law.reason, 0) + 1
                    bucket["unavailable_reasons"][law.reason] = (
                        bucket["unavailable_reasons"].get(law.reason, 0) + 1
                    )
    for record in by_enemy.values():
        record["living_availability_rate"] = (
            record["living_available"] / record["living"]
            if record["living"] else None
        )
        record["unavailable_reasons"] = dict(
            sorted(record["unavailable_reasons"].items()),
        )
    # A conservative public-only consistency probe: observe the actual next
    # announced intent only when the enemy survived with unchanged visible
    # HP/effects and no formation membership change. This detects support
    # contradictions but is not a source/native parity certification.
    checked = contradictions = 0
    examples: list[dict[str, Any]] = []
    for episode in episodes:
        for i in range(len(episode.decisions)):
            previous, following = episode.frames[i:i + 2]
            past = previous.get("combat")
            future = following.get("combat")
            if (not isinstance(past, dict)
                    or not isinstance(future, dict)
                    or previous.get("floor") != following.get("floor")
                    or type(past.get("turn")) is not int
                    or future.get("turn") != past["turn"] + 1):
                continue
            past_enemies, new_enemies = past.get("enemies"), future.get("enemies")
            if not isinstance(past_enemies, list) or not isinstance(new_enemies, list):
                continue
            previous_ids = {
                enemy.get("instance_id") for enemy in past_enemies
                if isinstance(enemy, dict) and type(enemy.get("hp")) is int
                and enemy["hp"] > 0
            }
            future_by_id = {
                enemy.get("instance_id"): enemy for enemy in new_enemies
                if isinstance(enemy, dict) and type(enemy.get("hp")) is int
                and enemy["hp"] > 0
            }
            if previous_ids != set(future_by_id):
                continue
            histories = getattr(
                episode.decisions[i], "h26_public_intent_history", None,
            )
            for enemy in past_enemies:
                if not isinstance(enemy, dict):
                    continue
                actual = future_by_id.get(enemy.get("instance_id"))
                if actual is None or any(
                    enemy.get(key) != actual.get(key)
                    for key in ("hp", "powers", "statuses")
                ):
                    continue
                law = next_intent_law(
                    enemy, catalog, public_enemies=past_enemies,
                    executed_moves=(
                        histories.get(enemy.get("instance_id"))
                        if isinstance(histories, dict) else None
                    ),
                )
                if not law.available or not isinstance(actual.get("move_id"), str):
                    continue
                checked += 1
                if actual["move_id"] not in {
                    move for move, mass in law.outcomes if mass > 0
                }:
                    contradictions += 1
                    if len(examples) < 8:
                        examples.append({
                            "enemy_id": enemy.get("enemy_id"),
                            "floor": previous.get("floor"),
                            "turn": past["turn"],
                            "before": enemy.get("move_id"),
                            "after": actual["move_id"],
                            "supported": [m for m, _ in law.outcomes],
                        })
    return {
        "available": available, "total": total,
        "availability_rate": available / total if total else None,
        "living": living,
        "living_available": living_available,
        "living_availability_rate": (
            living_available / living if living else None
        ),
        "unavailable_reasons": dict(sorted(reasons.items())),
        "by_enemy": dict(sorted(by_enemy.items())),
        "next_turn_support_audit": {
            "checked": checked,
            "out_of_support": contradictions,
            "examples": examples,
        },
    }
