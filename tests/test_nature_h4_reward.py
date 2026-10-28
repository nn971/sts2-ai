"""Focused regression tests for H4 reward ordering and contracts."""
import pytest

from sts2_ai.nature.h4_reward import (
    H4_REWARD, H9_TOTAL_HP_REWARD, boss_progress_score,
    encounter_total_hp_score, opening_enemy_health,
)


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
    with pytest.raises(ValueError, match="Missing opening enemies"):
        opening_enemy_health({"combat": {"enemies": []}})


def test_read_opening_health():
    assert opening_enemy_health({"combat": {"enemies": [
        {"hp": 72}, {"hp": 35}
    ]}}) == 107


def test_total_hp_reward_covers_death_spawns_and_multiple_enemies():
    # Gremlin Merc: 53 HP parent + 15 sneaky + 18 fat; parent is dead.
    assert encounter_total_hp_score(
        "defeat", player_exit_hp=0, player_max_hp=70,
        total_enemy_hp=86, remaining_enemy_hp=21,
    ) == pytest.approx(65 / 86)
    # Phrog Parasite: four non-minion Wrigglers must count in denominator.
    assert encounter_total_hp_score(
        "defeat", player_exit_hp=0, player_max_hp=70,
        total_enemy_hp=146, remaining_enemy_hp=25,
    ) == pytest.approx(121 / 146)
    # Multi-stage same-instance boss (111 + 212 + 313 HP).
    assert encounter_total_hp_score(
        "defeat", player_exit_hp=0, player_max_hp=70,
        total_enemy_hp=636, remaining_enemy_hp=250,
    ) == pytest.approx(386 / 636)


def test_total_hp_reward_preserves_victory_order_and_rejects_missing_totals():
    assert H4_REWARD != H9_TOTAL_HP_REWARD
    assert encounter_total_hp_score(
        "victory", player_exit_hp=35, player_max_hp=70,
        total_enemy_hp=None, remaining_enemy_hp=None,
    ) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="authoritative"):
        encounter_total_hp_score(
            "defeat", player_exit_hp=0, player_max_hp=70,
            total_enemy_hp=None, remaining_enemy_hp=10,
        )
    with pytest.raises(ValueError, match="authoritative"):
        encounter_total_hp_score(
            "defeat", player_exit_hp=0, player_max_hp=70,
            total_enemy_hp=20, remaining_enemy_hp=25,
        )
    with pytest.raises(ValueError, match="authoritative"):
        encounter_total_hp_score(
            "defeat", player_exit_hp=0, player_max_hp=70,
            total_enemy_hp=0, remaining_enemy_hp=0,
        )
