"""Act-1 boss health is a public-only, monotone-with-damage diagnostic."""
from __future__ import annotations

import pytest

from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation.boss_progress import (
    BossProgress,
    BossProgressTracker,
    summarize_boss_runs,
)
from sts2_ai.evaluation.run import RunSummary
from sts2_ai.training.selfplay import Episode, _bounded_return, _normalized_progress


def frame(floor: int, enemy_hp: tuple[tuple[int, int], ...]) -> dict[str, object]:
    return {
        "act": 1, "floor": floor,
        "act_one_boss_encounter_id": "proto.encounter.the_kin_boss",
        "combat": {"enemies": [
            {"instance_id": identity, "hp": hp}
            for identity, hp in enemy_hp
        ]},
    }


def test_public_boss_entry_roster_ignores_summons_and_absent_dead_enemies() -> None:
    track = BossProgressTracker(NATIVE_OVERGROWTH)
    track.observe(frame(15, ((1, 250),)))
    assert track.result() is None
    track.observe(frame(16, ((1, 200), (2, 50), (3, 50))))
    assert track.result() == BossProgress(
        "proto.encounter.the_kin_boss", 300, 300
    )
    # Encounter adds a 500 HP minion; no inflation of the old initial roster.
    track.observe(frame(16, ((1, 80), (3, 20), (99, 500))))
    result = track.result()
    assert result is not None
    assert result.initial_hp == 300
    assert result.remaining_hp == 100
    assert result.damage_fraction == pytest.approx(2 / 3)
    assert track.result(act1_cleared=True).remaining_hp == 0  # type: ignore[union-attr]


def test_boss_damage_can_not_outscore_actual_clear_and_default_is_invariant() -> None:
    args = ("seed", (), "defeat", 1, 16, 0.0, True, NATIVE_OVERGROWTH)
    no_damage = Episode(*args, BossProgress(None, 300, 300))
    near_kill = Episode(*args, BossProgress(None, 300, 15))
    act_two = Episode("clear", (), "defeat", 2, 1, 0.0, True, NATIVE_OVERGROWTH)

    # Without the flag all previous reward values stay exactly unchanged.
    assert _normalized_progress(no_damage) == 16 / 28
    assert _normalized_progress(near_kill) == 16 / 28
    assert _bounded_return(no_damage, 0.4) == _bounded_return(near_kill, 0.4)

    assert _normalized_progress(no_damage, boss_damage_weight=1) == 15 / 28
    assert _normalized_progress(near_kill, boss_damage_weight=1) == pytest.approx(
        15.95 / 28
    )
    assert _normalized_progress(
        near_kill, boss_damage_weight=1
    ) < _normalized_progress(act_two, boss_damage_weight=1)
    assert _bounded_return(
        near_kill, 0.4, boss_damage_weight=1,
    ) > _bounded_return(no_damage, 0.4, boss_damage_weight=1)

    # No invented boss progress for unobserved fights.
    unknown = Episode("x", (), "defeat", 1, 16, 0.0, True, NATIVE_OVERGROWTH)
    assert _normalized_progress(unknown, boss_damage_weight=1) == 16 / 28


def test_boss_summaries_exclude_censored_and_nonboss_defeats() -> None:
    def make(
        seed: str, *, outcome: str, boss: BossProgress | None,
        censored: bool = False, cleared: bool | None = False,
    ) -> RunSummary:
        return RunSummary(
            seed=seed, outcome=outcome, terminal_act=1, terminal_floor=16,
            decisions=10, emulator_transitions=10, wall_seconds=1.0,
            agent_compute_seconds=0.1, frontier_progress=15.0,
            frontier_enemy_hp=80, hp_trajectory=(70, 0),
            act1_cleared=cleared, censored=censored,
            environment=NATIVE_OVERGROWTH, boss_progress=boss,
        )

    record = summarize_boss_runs([
        make("a", outcome="defeat", boss=BossProgress(None, 100, 10)),
        make("b", outcome="defeat", boss=BossProgress(None, 100, 90)),
        make("c", outcome="victory", boss=BossProgress(None, 100, 0),
             cleared=True),
        make("d", outcome="defeat", boss=None),
        make("e", outcome="truncated", boss=BossProgress(None, 100, 0),
             censored=True, cleared=None),
    ])
    assert record["completed"] == 4
    assert record["boss_entries"] == 3
    assert record["boss_defeats"] == 2
    assert record["boss_clears"] == 1
    assert record["boss_near_kills_on_defeat_80pct"] == 1
    assert record["mean_boss_damage_fraction_on_defeat"] == pytest.approx(0.5)


def test_boss_progress_rejects_invalid_negative_hp_inputs() -> None:
    # Clamp healing or overshoot to the first observed HP budget.
    track = BossProgressTracker(NATIVE_OVERGROWTH)
    track.observe(frame(16, ((1, 10),)))
    track.observe(frame(16, ((1, 200),)))
    assert track.result() is not None
    assert track.result().damage_fraction == 0  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="positive"):
        assert BossProgress(None, 0, 0).damage_fraction == 0
