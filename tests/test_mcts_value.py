import json

from sts2_ai.emulator import Observation
from sts2_ai.search import sts2_value


def _observation(payload: dict[str, object]) -> Observation:
    return Observation(
        policy_id="test",
        payload_json=json.dumps(payload),
        observation_hash="hash",
    )


def test_cutoff_value_rewards_enemy_damage_within_same_combat() -> None:
    healthy_enemy = _observation(
        {
            "act": 1,
            "floor": 2,
            "hp": 60,
            "max_hp": 70,
            "combat": {"enemies": [{"hp": 80}, {"hp": 40}]},
        }
    )
    damaged_enemy = _observation(
        {
            "act": 1,
            "floor": 2,
            "hp": 60,
            "max_hp": 70,
            "combat": {"enemies": [{"hp": 15}, {"hp": 0}]},
        }
    )

    assert sts2_value(damaged_enemy) > sts2_value(healthy_enemy)


def test_cutoff_value_rewards_player_hp() -> None:
    low_hp = _observation({"act": 1, "floor": 3, "hp": 20, "max_hp": 70})
    high_hp = _observation({"act": 1, "floor": 3, "hp": 60, "max_hp": 70})

    assert sts2_value(high_hp) > sts2_value(low_hp)


def test_act_transition_progress_is_continuous() -> None:
    end_act_one = _observation({"act": 1, "floor": 6, "hp": 50, "max_hp": 70})
    start_act_two = _observation({"act": 2, "floor": 0, "hp": 50, "max_hp": 70})

    assert sts2_value(end_act_one) == sts2_value(start_act_two)


def test_terminal_outcomes_override_cutoff_features() -> None:
    victory = _observation(
        {"terminal_outcome": "victory", "hp": 1, "max_hp": 70}
    )
    defeat = _observation(
        {"terminal_outcome": "defeat", "hp": 70, "max_hp": 70}
    )

    assert sts2_value(victory) == 1.0
    assert sts2_value(defeat) == -1.0
