"""Versioned, engine-supported combat contexts for Nature challenge studies.

A scenario selects a *conditioned source fixture* BEFORE any combat chance event.
Nature can control only unresolved physical draws. Enemy intent remains governed
by the emulator's declared Chance Law. This is not a native encounter mixture.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

BASELINE_SCENARIO = "shrinker-a0"
SCENARIO_SCHEMA = "sts2-ai-nature-conditioned-single-combat-scenario-v1"


@dataclass(frozen=True)
class CombatScenario:
    scenario_id: str
    encounter_id: str
    native_entry: str
    ascension: int
    act: int = 1
    total_floor: int = 2
    first_monster_combat_id: int = 1
    map_column: int = 3
    map_row: int = 1

    def __post_init__(self) -> None:
        if (
            self.act != 1 or self.total_floor != 2
            or self.first_monster_combat_id != 1
            or not 0 <= self.ascension <= 10
            or self.map_column != 3 or self.map_row != 1
        ):
            raise ValueError("Unsupported H2 conditioned encounter fixture")


SCENARIOS: dict[str, CombatScenario] = {
    "shrinker-a0": CombatScenario(
        "shrinker-a0", "proto.encounter.shrinker_beetle_weak",
        "ShrinkerBeetleWeak", 0,
    ),
    "shrinker-a10": CombatScenario(
        "shrinker-a10", "proto.encounter.shrinker_beetle_weak",
        "ShrinkerBeetleWeak", 10,
    ),
    "nibbits-a0": CombatScenario(
        "nibbits-a0", "proto.encounter.nibbits_weak", "NibbitsWeak", 0,
    ),
    "nibbits-a10": CombatScenario(
        "nibbits-a10", "proto.encounter.nibbits_weak", "NibbitsWeak", 10,
    ),
}


def parse_scenario_pool(raw: str) -> tuple[str, ...]:
    """An explicit uniform pool, without source-level encounter rerolls."""
    names = tuple(x.strip() for x in raw.split(","))
    if (
        not names or len(names) > len(SCENARIOS)
        or any(name not in SCENARIOS for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Unknown, duplicate, empty, or oversized combat scenario pool")
    return names


def scenario_for_seed(source_seed: str, pool: tuple[str, ...]) -> CombatScenario:
    if not isinstance(source_seed, str) or not source_seed:
        raise ValueError("Missing source seed")
    if (
        not pool or any(key not in SCENARIOS for key in pool)
        or len(pool) != len(set(pool))
    ):
        raise ValueError("Invalid conditioned scenario pool")
    # Stable, explicit scenario assignment: no extra emulator RNG consumed.
    # Sorted pool makes the source fixture invariant under CLI argument order.
    canonical = sorted(pool)
    bucket = int.from_bytes(hashlib.sha256(
        ("h2-context-v1:" + source_seed).encode("utf-8")
    ).digest()[:8], "big")
    return SCENARIOS[canonical[bucket % len(canonical)]]
