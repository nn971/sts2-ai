"""Boss health diagnostics from player-visible Act-1 Overgrowth observations.

No hidden state or emulator hooks. For multi-enemy bosses (The Kin), the
encounter HP budget is the sum of the enemies present in the first visible
boss-combat frame. Later summons are deliberately excluded; this is an
initial-roster damage proxy, NOT guaranteed phase-aware native boss HP.
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH

if TYPE_CHECKING:
    from sts2_ai.evaluation.run import RunSummary


@dataclass(frozen=True, slots=True)
class BossProgress:
    encounter_id: str | None
    initial_hp: int
    remaining_hp: int

    @property
    def damage_fraction(self) -> float:
        if self.initial_hp <= 0:
            raise ValueError("Boss entry HP must be positive")
        return max(0.0, min(1.0, 1.0 - self.remaining_hp / self.initial_hp))


class BossProgressTracker:
    """Passive public-frame tracker; does not influence the actor's inputs."""

    def __init__(self, environment: str) -> None:
        self.environment = environment
        self._initial: dict[int, int] | None = None
        self._current: dict[int, int] = {}
        self._encounter_id: str | None = None

    def observe(self, state: dict[str, Any]) -> None:
        if (
            self.environment != NATIVE_OVERGROWTH
            or state.get("act") != 1
            or state.get("floor") != 16
        ):
            return
        combat = state.get("combat")
        if not isinstance(combat, dict):
            return
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            return

        roster: dict[int, int] = {}
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            identity, hp = enemy.get("instance_id"), enemy.get("hp")
            if type(identity) is int and type(hp) is int:
                roster[identity] = max(0, hp)

        if self._initial is None:
            initial = {i: hp for i, hp in roster.items() if hp > 0}
            if not initial:
                return
            self._initial = initial
            self._current = initial.copy()
            identity = state.get("act_one_boss_encounter_id")
            if isinstance(identity, str):
                self._encounter_id = identity

        # Track only the entry roster: newly spawned minions must never
        # inflate the initial HP denominator or reduce credited damage.
        for identity in self._initial:
            self._current[identity] = roster.get(identity, 0)

    def result(self, *, act1_cleared: bool = False) -> BossProgress | None:
        if self._initial is None:
            return None
        start = sum(self._initial.values())
        remaining = 0 if act1_cleared else min(start, sum(self._current.values()))
        return BossProgress(self._encounter_id, start, remaining)


def summarize_boss_runs(runs: Sequence[RunSummary]) -> dict[str, Any]:
    """Boss diagnostics never label capped or failed emulator runs as defeats."""
    valid = [r for r in runs if not r.censored]
    entered = [r for r in valid if r.boss_progress is not None]
    defeated = [
        r for r in entered
        if r.outcome == "defeat" and r.act1_cleared is not True
    ]
    fractions = [
        r.boss_progress.damage_fraction
        for r in defeated if r.boss_progress is not None
    ]
    return {
        "completed": len(valid),
        "boss_entries": len(entered),
        "boss_defeats": len(defeated),
        "boss_clears": sum(r.act1_cleared is True for r in entered),
        "mean_boss_damage_fraction_on_defeat": (
            statistics.fmean(fractions) if fractions else None
        ),
        "boss_near_kills_on_defeat_80pct": sum(x >= 0.8 for x in fractions),
    }
