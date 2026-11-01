"""Focused H16 read-only floor replay summary tests."""
from __future__ import annotations

from types import SimpleNamespace

from tools.report_h16_floor_progress import (
    episode_floor_record,
    summarize_floor_records,
)


def frame(floor: int, hp: int, completed: tuple[int, ...] = ()) -> dict:
    return {
        "act": 1, "floor": floor, "hp": hp, "max_hp": 70,
        "completed_rooms": [{"act": 1, "floor": f} for f in completed],
    }


def test_defeat_records_first_completion_hp_and_actual_death_floor():
    episode = SimpleNamespace(
        seed="test-seed",
        outcome="defeat",
        decisions=(None, None, None, None),
        frames=(
            frame(0, 70),
            frame(1, 60),
            frame(1, 51, (1,)),
            frame(2, 48, (1,)),
            frame(2, 0, (1,)),
        ),
    )
    record = episode_floor_record(episode)
    assert record["seed"] == "test-seed"
    assert record["max_floor_reached"] == 2
    assert record["max_floor_completed"] == 1
    assert record["terminal_floor"] == 2
    assert record["terminal_hp"] == 0
    assert record["post_room_hp"] == {"1": {"hp": 51, "max_hp": 70}}


def test_clear_and_floor_reach_summary():
    clear = SimpleNamespace(
        seed="clear",
        outcome="act_clear",
        decisions=(None,),
        frames=(frame(15, 23, tuple(range(1, 15))),
                frame(16, 11, tuple(range(1, 17)))),
    )
    defeat = SimpleNamespace(
        seed="defeat",
        outcome="defeat",
        decisions=(None,),
        frames=(frame(7, 22, tuple(range(1, 7))),
                frame(7, 0, tuple(range(1, 7)))),
    )
    records = [episode_floor_record(clear), episode_floor_record(defeat)]
    summary = summarize_floor_records(records)
    assert summary["episodes"] == 2
    assert summary["clears"] == 1
    assert summary["mean_max_floor_reached"] == 11.5
    assert summary["median_max_floor_reached"] == 11.5
    assert summary["reached_floors"]["10"] == 1
    assert summary["completed_floors"]["16"] == 1
