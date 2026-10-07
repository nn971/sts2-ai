from sts2_ai.evaluation.run import _continuous_progress, _remaining_enemy_hp


def test_continuous_progress_rewards_damage_within_floor() -> None:
    healthy = {
        "act": 1,
        "floor": 6,
        "combat": {"enemies": [{"hp": 100}, {"hp": 40}]},
    }
    damaged = {
        "act": 1,
        "floor": 6,
        "combat": {"enemies": [{"hp": 10}, {"hp": 0}]},
    }

    assert _continuous_progress(damaged) > _continuous_progress(healthy)
    assert _remaining_enemy_hp(healthy) == 140
    assert _remaining_enemy_hp(damaged) == 10


def test_continuous_progress_uses_six_floor_acts() -> None:
    assert _continuous_progress({"act": 1, "floor": 6}) == 6.0
    assert _continuous_progress({"act": 2, "floor": 0}) == 6.0
