"""Versioned single-combat terminal rewards.

H3's hp_preservation_v1 is left untouched in draw_only_episode.py.
H4's progress score is lexicographic: every victory beats every defeat,
then normalized survivor HP or damage to the original enemy formation.
"""
from __future__ import annotations

from typing import Any

H4_REWARD = "boss_progress_v2"
H9_TOTAL_HP_REWARD = "encounter_nonminion_total_hp_v1"


def boss_progress_score(
    outcome: str, *, player_exit_hp: int, player_max_hp: int,
    opening_enemy_hp: int, terminal_enemy_hp: int | None,
) -> float:
    """Return [0,1] for defeats, (1,2] for victories.

    The capped HP fraction keeps victory strictly preferable to defeat.
    The opening boss/formation HP denominator is a snapshot of *actual*
    initial health, not a static content-table estimate. The defeated
    formation's terminal HP must come from the authoritative engine state.
    """
    if (not isinstance(player_exit_hp, int) or isinstance(player_exit_hp, bool)
            or not isinstance(player_max_hp, int) or isinstance(player_max_hp, bool)
            or player_max_hp <= 0 or not 0 <= player_exit_hp <= player_max_hp
            or not isinstance(opening_enemy_hp, int) or isinstance(opening_enemy_hp, bool)
            or opening_enemy_hp <= 0):
        raise ValueError("Invalid H4 player or opening-enemy HP")
    if outcome == "victory":
        return 1.0 + player_exit_hp / player_max_hp
    if outcome != "defeat":
        raise ValueError("Unknown H4 outcome")
    if (not isinstance(terminal_enemy_hp, int)
            or isinstance(terminal_enemy_hp, bool)
            or terminal_enemy_hp < 0):
        raise ValueError("Defeat requires authoritative terminal enemy HP")
    # Boss healing, summons and phase changes can make this larger than the
    # opening health. They are clamped as progress=0, not negative reward.
    # In H4 the supported boss fixture is a single initial enemy.
    return max(0.0, 1.0 - terminal_enemy_hp / opening_enemy_hp)


def opening_enemy_health(observation: dict[str, Any]) -> int:
    """Read visible opening enemy HP from the first fair player observation."""
    combat = observation.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Missing opening combat")
    enemies = combat.get("enemies")
    if not isinstance(enemies, list) or not enemies:
        raise ValueError("Missing opening enemies")
    hp = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            raise ValueError("Malformed opening enemy")
        value = enemy.get("hp")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Missing opening enemy HP")
        hp.append(value)
    total = sum(hp)
    if total <= 0:
        raise ValueError("Opening enemies already defeated")
    return total


def encounter_total_hp_score(
    outcome: str, *, player_exit_hp: int, player_max_hp: int,
    total_enemy_hp: int | None, remaining_enemy_hp: int | None,
) -> float:
    """H9 total-encounter HP reward, excluding minions.

    The emulator counts every non-minion enemy instance (including death
    summons) and completed in-place forms in total_enemy_hp. Remaining HP
    contains only currently living non-minions. A loss therefore earns credit
    for defeated earlier stages, even when the boss summons replacements
    or regains its HP for a later stage. Victory remains HP-preserving.
    """
    if (type(player_exit_hp) is not int or
            type(player_max_hp) is not int or player_max_hp <= 0 or
            not 0 <= player_exit_hp <= player_max_hp):
        raise ValueError("Invalid player HP for total-encounter reward")
    if outcome == "victory":
        return 1.0 + player_exit_hp / player_max_hp
    if outcome != "defeat":
        raise ValueError("Unknown total-encounter outcome")
    if (type(total_enemy_hp) is not int or total_enemy_hp <= 0 or
            type(remaining_enemy_hp) is not int or
            not 0 <= remaining_enemy_hp <= total_enemy_hp):
        raise ValueError(
            "Defeat requires authoritative total and remaining non-minion HP"
        )
    return 1.0 - remaining_enemy_hp / total_enemy_hp


H15_DEFEAT_MARGIN_REWARD = "encounter_nonminion_total_hp_defeat_margin_beta050_v1"


def encounter_total_hp_defeat_margin_score(
    outcome: str, *, player_exit_hp: int, player_max_hp: int,
    total_enemy_hp: int | None, remaining_enemy_hp: int | None,
    beta: float = 0.5,
) -> float:
    """H15 M17: retain full victory HP reward, discount defeat progress.

    Every win is worth >=1; every loss is worth <=1-beta. This objective
    is deliberately separate from the historical H9 reward, so earlier
    checkpoints and labels remain interpretable under their original law.
    """
    import math

    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("Defeat margin beta must lie in [0,1)")
    base = encounter_total_hp_score(
        outcome, player_exit_hp=player_exit_hp, player_max_hp=player_max_hp,
        total_enemy_hp=total_enemy_hp, remaining_enemy_hp=remaining_enemy_hp,
    )
    return (1.0 - beta) * base if outcome == "defeat" else base
