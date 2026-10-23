"""Versioned single-combat terminal rewards.

H3's hp_preservation_v1 is left untouched in draw_only_episode.py.
H4's progress score is lexicographic: every victory beats every defeat,
then normalized survivor HP or damage to the original enemy formation.
"""
from __future__ import annotations

import math

H4_REWARD = "boss_progress_v2"


def boss_progress_score(
    outcome: str, *, player_exit_hp: int, player_max_hp: int,
    opening_enemy_hp: int, terminal_enemy_hp: int | None,
) -> float:
    """Return [0,1) for defeats, (1,2] for victories.

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


def opening_enemy_health(observation: dict) -> int:
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
