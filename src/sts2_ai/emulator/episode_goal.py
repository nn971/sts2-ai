"""Versioned public-observation episode goals, separate from emulator termination.

Never manufacture a terminal emulator state. Act-1 training may stop at the
certified boss-clear boundary while the same emulator run remains continuable.
"""
from __future__ import annotations

from typing import Any

from .run_environment import NATIVE_OVERGROWTH

PROTOTYPE_THREE_ACT_GOAL = "prototype-three-act-v0"
NATIVE_ACT1_BOSS_GOAL = "native-act1-boss-v1"
EPISODE_GOALS = (PROTOTYPE_THREE_ACT_GOAL, NATIVE_ACT1_BOSS_GOAL)

# The pinned emulator serializes C# RunPhase.ActTransition and
# PrototypeRoomType.Boss as their numeric enum values in CanonicalJson.
_ACT_TRANSITION_PHASE = 9
_BOSS_ROOM_TYPE = 5


def require_episode_goal(goal: str, environment: str) -> None:
    if goal not in EPISODE_GOALS:
        raise ValueError(f"Unknown episode goal: {goal!r}")
    if goal == NATIVE_ACT1_BOSS_GOAL and environment != NATIVE_OVERGROWTH:
        raise ValueError("Native Act-1 boss goal requires native-overgrowth environment")


def certified_native_act1_clear(frame: dict[str, Any]) -> bool:
    """True only after the emulator records a completed Act-1 floor-16 boss.

    The adapter checks public facts rather than guessing from current floor,
    boss HP, or a decision limit. ActTransition is reached only after the
    boss's end-of-act reward has been resolved. It is *not* a simulator terminal.
    """
    if (
        type(frame.get("act")) is not int or frame["act"] != 1
        or type(frame.get("floor")) is not int or frame["floor"] != 16
        or frame.get("phase") != _ACT_TRANSITION_PHASE
        or frame.get("combat") is not None
        or frame.get("terminal_outcome") is not None
    ):
        return False
    rooms = frame.get("completed_rooms")
    if not isinstance(rooms, list):
        return False
    return any(
        isinstance(room, dict)
        and type(room.get("act")) is int and room["act"] == 1
        and type(room.get("floor")) is int and room["floor"] == 16
        and room.get("room_type") == _BOSS_ROOM_TYPE
        for room in rooms
    )
