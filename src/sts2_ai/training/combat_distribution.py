"""Conditional empirical distribution of PUBLIC combat outcomes.

A risk-neutral scalar mean is not enough for future tactical objectives:
retain survival probability, individual HP outcomes, and correlated potion
inventories. This module is a deliberately transparent non-neural outcome
baseline. It accepts only already-observed combat boundaries, never exact
states or hidden simulator streams.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "sts2-combat-empirical-distribution-v1"


def _key(outcome: dict[str, Any]) -> str:
    entry = outcome["entry"]
    # Condition on specific fight, potion MULTISET, and entry HP band.
    # Do not condition on policy actions/outcome or on post-combat information.
    enemies = sorted(str(x) for x in outcome["enemy_ids"])
    potion_ids = sorted(str(x) for x in entry["potions"])
    hp = entry["hp"]
    maximum = entry["max_hp"]
    hp_bin = (
        min(9, max(0, int(10 * hp / maximum)))
        if type(hp) is int and type(maximum) is int and maximum > 0
        else -1
    )
    return json.dumps(
        [enemies, potion_ids, hp_bin], separators=(",", ":"), sort_keys=True
    )


@dataclass(frozen=True)
class OutcomePoint:
    win: bool
    exit_hp: int | None
    exit_potions: tuple[str, ...]
    entry_hp: int | None

    @classmethod
    def parse(cls, outcome: dict[str, Any]) -> OutcomePoint:
        entry = outcome["entry"]
        exit_ = outcome["exit"]
        return cls(
            win=outcome["result"] == "victory",
            exit_hp=exit_["hp"] if type(exit_["hp"]) is int else None,
            exit_potions=tuple(sorted(str(x) for x in exit_["potions"])),
            entry_hp=entry["hp"] if type(entry["hp"]) is int else None,
        )


def load_combat_samples(paths: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or (
                    record.get("schema") != "sts2-public-combat-sample-v1"
                ):
                    raise ValueError(f"{path}:{lineno}: incompatible sample schema")
                outcome = record.get("outcome")
                if not isinstance(outcome, dict):
                    raise ValueError(f"{path}:{lineno}: missing outcome")
                samples.append(outcome)
    return samples


class EmpiricalCombatDistribution:
    """Discrete empirical joint exit-resource distribution with fallback.

    Small strata fall back to the full dataset. Outputs are descriptive
    on-policy frequencies, NOT calibrated forecasts for a new tactical
    policy or counterfactual actions.
    """

    def __init__(
        self, outcomes: list[dict[str, Any]], *, min_group_size: int = 8
    ) -> None:
        if min_group_size < 1:
            raise ValueError("min_group_size must be positive")
        if not outcomes:
            raise ValueError("No completed combat outcome samples")
        self.min_group_size = min_group_size
        self.groups: dict[str, list[OutcomePoint]] = defaultdict(list)
        self.all_points: list[OutcomePoint] = []
        for outcome in outcomes:
            point = OutcomePoint.parse(outcome)
            self.groups[_key(outcome)].append(point)
            self.all_points.append(point)

    def estimate(self, entry_and_encounter: dict[str, Any]) -> dict[str, Any]:
        key = _key(entry_and_encounter)
        group = self.groups.get(key, [])
        selected = group if len(group) >= self.min_group_size else self.all_points
        wins = sum(point.win for point in selected)
        hp = [point.exit_hp for point in selected if point.exit_hp is not None]
        exit_states: dict[tuple[int | None, tuple[str, ...]], int] = {}
        for point in selected:
            state = (point.exit_hp, point.exit_potions)
            exit_states[state] = exit_states.get(state, 0) + 1
        n = len(selected)
        return {
            "schema": SCHEMA,
            "matched_context": len(group) >= self.min_group_size,
            "matched_sample_count": len(group),
            "sample_count": n,
            "observed_win_probability": wins / n,
            "exit_hp_mean": sum(hp) / len(hp) if hp else None,
            "exit_hp_normalized_variance": (
                sum((x - sum(hp) / len(hp)) ** 2 for x in hp)
                / (len(hp) * max(
                    1.0,
                    float(entry_and_encounter["entry"].get("max_hp") or 1),
                ) ** 2)
                if hp else None
            ),
            "joint_exit_outcomes": [
                {"hp": state[0], "potions": list(state[1]), "count": count}
                for state, count in sorted(
                    exit_states.items(),
                    key=lambda item: (item[0][0] is None, item[0][0] or 0, item[0][1]),
                )
            ],
        }
