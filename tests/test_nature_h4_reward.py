"""Focused regression tests for H4 reward ordering and contracts."""
import pytest

from sts2_ai.nature.h4_reward import boss_progress_score, opening_enemy_health


def test_victories_rank_above_defeats():
    loss = boss_progress_score(
        "defeat", player_exit_hp=0, player_max_hp=70,
        opening_enemy_hp=200, terminal_enemy_hp=1)
    win = boss_progress_score(
        "victory", player_exit_hp=1, player_max_hp=70,
        opening_enemy_hp=200, terminal_enemy_hp=None)
    assert 0 < loss < 1 < win < 2


def test_progress_on_defeats_and_survival_on_wins():
    def score(outcome, ehp, bhp):
        return boss_progress_score(
            outcome, player_exit_hp=ehp, player_max_hp=70,
            opening_enemy_hp=200, terminal_enemy_hp=bhp)
    assert score("defeat", 0, 200) == 0
    assert score("defeat", 0, 100) == pytest.approx(0.5)
    assert score("victory", 35, None) == pytest.approx(1.5)
    assert score("victory", 70, None) == 2
    assert score("defeat", 0, 250) == 0


def test_reject_missing_terminal_hp_and_invalid_opening():
    with pytest.raises(ValueError, match="terminal enemy HP"):
        boss_progress_score(
            "defeat", player_exit_hp=0, player_max_hp=70,
            opening_enemy_hp=200, terminal_enemy_hp=None)
    with pytest.raises(ValueError, match="opening enemy"):
        opening_enemy_health({"combat": {"enemies": []}})


def test_read_opening_health():
    assert opening_enemy_health({"combat": {"enemies": [
        {"hp": 72}, {"hp": 35}
    ]}}) == 107
