"""Native Act-1 progress shaping and generalized advantage estimation (GAE).

Macro boundaries may be separated by entire combats. Each reward belongs to
the preceding *legal macro action*, and is credited only when the next macro
decision (or a certified Act-1 outcome) is observed. The reward is based only
on player-visible floor advancement, never hidden combat consequences.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

MAX_ACT1_FLOOR = 16


@dataclass(frozen=True)
class H14MacroTarget:
    reward: float
    advantage: float
    value_target: float


def macro_gae(
    floors: Sequence[int],
    behavior_values: Sequence[float],
    *,
    final_floor: int,
    act1_cleared: bool,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    floor_reward: float = 0.02,
) -> tuple[H14MacroTarget, ...]:
    """Discounted TD(lambda) targets over *macro actions*, not combat frames.

    End states are terminal for the chosen Act-1 task. There is no bootstrap
    from a censored episode: callers exclude censored runs entirely.
    The small positive floor increment gives defeats a useful ordering.
    """
    if not floors or len(floors) != len(behavior_values):
        raise ValueError("Macro GAE requires aligned nonempty states/values")
    if (not all(type(x) is int and 0 <= x <= MAX_ACT1_FLOOR for x in (*floors, final_floor))
            or any(b < a for a, b in zip(floors, (*floors[1:], final_floor)))):
        raise ValueError("Public Act-1 floor progress must be monotone and valid")
    if not all(math.isfinite(v) for v in behavior_values):
        raise ValueError("Macro values must be finite")
    if (not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1
            or not math.isfinite(floor_reward) or floor_reward < 0):
        raise ValueError("Invalid macro return coefficients")
    if act1_cleared and final_floor != MAX_ACT1_FLOOR:
        raise ValueError("Act-1 clear must terminate at the complete Act-1 boundary")
    n = len(floors)
    increments = [
        floor_reward * (next_floor - floor)
        for floor, next_floor in zip(floors, (*floors[1:], final_floor))
    ]
    increments[-1] += 1.0 if act1_cleared else -1.0
    advantages = [0.0] * n
    trailing = 0.0
    for t in range(n - 1, -1, -1):
        future_value = behavior_values[t + 1] if t + 1 < n else 0.0
        delta = increments[t] + gamma * future_value - behavior_values[t]
        trailing = delta + gamma * gae_lambda * trailing
        advantages[t] = trailing
    return tuple(
        H14MacroTarget(
            reward=increments[t],
            advantage=advantages[t],
            value_target=advantages[t] + behavior_values[t],
        )
        for t in range(n)
    )
