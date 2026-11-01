"""Public-history floor metrics and bounded enemy-progress combat credit.

Only observable combat enemies, HP, and room boundaries are used. Values
recorded for an unfinished fight are descriptive targets, not terminal wins.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Sequence

MINION_POWER_ID = "proto.power.minion"


@dataclass(frozen=True)
class CombatResult:
    start: int
    end: int
    floor: int | None
    won: bool
    enemy_progress: float

    def quality(self, loss_progress_cap: float = 0.2) -> float:
        if not 0 <= loss_progress_cap < 1:
            raise ValueError("Loss progress cap must be in [0, 1)")
        return 1.0 if self.won else loss_progress_cap * self.enemy_progress


def _is_minion(enemy: dict[str, Any]) -> bool:
    if isinstance(enemy.get("is_minion"), bool):
        return enemy["is_minion"]
    powers = enemy.get("powers")
    return isinstance(powers, list) and any(
        isinstance(power, dict) and power.get("power_id") == MINION_POWER_ID
        for power in powers
    )


def enemy_hp_progress(combat_frames: Sequence[dict[str, Any]]) -> float:
    """Net normalized removal of primary enemy HP in one unfinished fight.

    A disappearing enemy is treated as defeated. New monsters and visible
    phase changes add their HP to the denominator. Healing within a phase
    reduces progress. Minions are ignored when primary enemies exist.
    """
    if not combat_frames:
        return 0.0
    stages: dict[tuple[int, int], tuple[int, int, bool]] = {}
    active: dict[int, tuple[tuple[int, int], str, int, int]] = {}
    generations: dict[int, int] = {}
    for frame in combat_frames:
        combat = frame.get("combat")
        if not isinstance(combat, dict):
            continue
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            continue
        present: set[int] = set()
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            ref = enemy.get("instance_id")
            hp = enemy.get("hp")
            enemy_id = enemy.get("enemy_id")
            raw_max = enemy.get("max_hp")
            if type(ref) is not int or type(hp) is not int or not isinstance(enemy_id, str):
                continue
            past = active.get(ref)
            maximum = (
                raw_max if type(raw_max) is int and raw_max > 0
                else past[3] if past is not None and past[1] == enemy_id
                else max(1, hp)
            )
            present.add(ref)
            # A changed public identity identifies a new phase. Max HP can
            # change within an ordinary phase (buffs/debuffs); treating that
            # as a kill would fabricate defeat-progress credit.
            # Reappearance following absence is a new observable generation.
            if past is None or past[1] != enemy_id:
                if past is not None:
                    # A visible boss/monster phase transition retires the
                    # preceding HP bar; it is not remaining enemy HP.
                    prior_key = past[0]
                    prior_max, _, prior_minion = stages[prior_key]
                    stages[prior_key] = (prior_max, 0, prior_minion)
                generation = generations.get(ref, -1) + 1
                generations[ref] = generation
                key = (ref, generation)
                stages[key] = (maximum, min(maximum, max(0, hp)), _is_minion(enemy))
            else:
                key = past[0]
                initial_max, _, minion = stages[key]
                # A stage can heal; use its current HP, never accumulated damage.
                stages[key] = (max(initial_max, maximum),
                               min(max(initial_max, maximum), max(0, hp)), minion)
            active[ref] = (key, enemy_id, hp, maximum)
        for ref in set(active) - present:
            active.pop(ref)
            # The last observation with this enemy was followed by its
            # disappearance. Mark its phase as defeated.
            # (A later reappearance gets its own generation.)
            key = (ref, generations[ref])
            maximum, _, minion = stages[key]
            stages[key] = (maximum, 0, minion)

    primary = [entry for entry in stages.values() if not entry[2]]
    included = primary if primary else list(stages.values())
    total = sum(maximum for maximum, _, _ in included)
    if total <= 0:
        return 0.0
    remaining = sum(hp for _, hp, _ in included)
    return max(0.0, min(1.0, (total - remaining) / total))


def combat_results(frames: Sequence[dict[str, Any]], outcome: str) -> tuple[CombatResult, ...]:
    """Partition the decisions in an Act episode into completed combats."""
    if outcome not in ("act_clear", "defeat", "censored"):
        raise ValueError("Unknown episode outcome")
    results: list[CombatResult] = []
    i = 0
    while i < len(frames) - 1:
        if not isinstance(frames[i].get("combat"), dict):
            i += 1
            continue
        j = i + 1
        while j < len(frames) and isinstance(frames[j].get("combat"), dict):
            j += 1
        # The terminal frame can still contain combat. A censored battle
        # has no known outcome and supplies no learning label.
        if j == len(frames) and outcome == "censored":
            i = j
            continue
        end = min(j, len(frames) - 1)
        following = frames[j] if j < len(frames) else frames[-1]
        survived = (j < len(frames) and type(following.get("hp")) is int
                    and following["hp"] > 0)
        # An Act-level defeat can follow a won earlier combat, but never
        # promotes the final unfinished battle into a victory.
        # The terminal Act-clear snapshot can still retain combat state.
        # An explicitly certified Act clear includes a won boss fight.
        won = ((j == len(frames) and outcome == "act_clear")
               or (survived and not (j == len(frames) - 1
                                     and outcome == "defeat"
                                     and following.get("hp", 0) <= 0)))
        results.append(CombatResult(
            i, end,
            frames[i].get("floor") if type(frames[i].get("floor")) is int else None,
            won,
            1.0 if won else enemy_hp_progress(frames[i:j]),
        ))
        i = max(j, i + 1)
    return tuple(results)


def episode_floor_record(
    frames: Sequence[dict[str, Any]], seed: str, outcome: str, decisions: int,
) -> dict[str, Any]:
    reached = [frame["floor"] for frame in frames
               if type(frame.get("floor")) is int and 0 <= frame["floor"] <= 16]
    completed: dict[int, dict[str, int]] = {}
    for frame in frames:
        visits = frame.get("completed_rooms")
        if not isinstance(visits, list):
            continue
        for visit in visits:
            if not isinstance(visit, dict) or visit.get("act") != 1:
                continue
            floor = visit.get("floor")
            if type(floor) is not int or not 1 <= floor <= 16 or floor in completed:
                continue
            hp, cap = frame.get("hp"), frame.get("max_hp")
            if type(hp) is int and type(cap) is int:
                completed[floor] = {"hp": hp, "max_hp": cap}
    last = frames[-1]
    return {
        "seed": seed, "outcome": outcome, "decisions": decisions,
        "max_floor_reached": max(reached, default=None),
        "max_floor_completed": max(completed, default=0),
        "death_floor": last.get("floor") if outcome == "defeat" else None,
        "terminal_hp": last.get("hp"),
        "post_room_hp": {str(k): v for k, v in sorted(completed.items())},
    }


def summarize_floor_progress(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    floors = [r["max_floor_reached"] for r in records
              if type(r.get("max_floor_reached")) is int]
    death_floors: dict[str, int] = {}
    for record in records:
        floor = record.get("death_floor")
        if type(floor) is int:
            key = str(floor)
            death_floors[key] = death_floors.get(key, 0) + 1
    room_hp: dict[str, dict[str, float | int]] = {}
    for floor in range(1, 17):
        vals = [r["post_room_hp"][str(floor)] for r in records
                if str(floor) in r["post_room_hp"]]
        room_hp[str(floor)] = {
            "completions": len(vals),
            "mean_hp_fraction": (
                statistics.mean(v["hp"] / max(1, v["max_hp"]) for v in vals)
                if vals else None
            ),
        }
    return {
        "mean_max_floor_reached": statistics.mean(floors) if floors else None,
        "median_max_floor_reached": statistics.median(floors) if floors else None,
        "reached_floors": {
            str(floor): sum((r["max_floor_reached"] or 0) >= floor for r in records)
            for floor in range(1, 17)
        },
        "completed_floors": {
            str(floor): sum(r["max_floor_completed"] >= floor for r in records)
            for floor in range(1, 17)
        },
        "death_floors": dict(sorted(death_floors.items(), key=lambda p: int(p[0]))),
        "per_floor_hp": room_hp,
        "runs": list(records),
    }
