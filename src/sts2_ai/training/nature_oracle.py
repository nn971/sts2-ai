"""Fully enumerated finite-horizon Nature oracle, independent of the emulator.

Two player actions and two *conditional* chance events are enough to expose
wrong trajectory ratios and score-function gradients. This is an exact
synthetic reference, not a model of actual Slay the Spire probabilities.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .chance_importance import ChanceLaw, log_chance_ratio


@dataclass(frozen=True, slots=True)
class ToyChanceTree:
    p_first: tuple[float, float]
    q_first: tuple[float, float]
    p_second: tuple[tuple[float, float], tuple[float, float]]
    q_second: tuple[tuple[float, float], tuple[float, float]]
    # [player action][first outcome][second outcome]
    rewards: tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ]

    def __post_init__(self) -> None:
        for x in (0, 1):
            ChanceLaw("toy-exact-v1", "toy-first", self.p_first, "SYNTHETIC_EXACT")
            ChanceLaw("toy-exact-v1", f"toy-second-{x}",
                      self.p_second[x], "SYNTHETIC_EXACT")
            if any(not math.isfinite(v) for action in self.rewards
                   for branch in action for v in branch):
                raise ValueError("Toy rewards must be finite")
            # Validate full proposal support before sampling.
            from .chance_importance import _check_probabilities
            _check_probabilities(self.q_first, full_support=True)
            _check_probabilities(self.q_second[x], full_support=True)

    @staticmethod
    def example() -> ToyChanceTree:
        return ToyChanceTree(
            p_first=(0.7, 0.3), q_first=(0.25, 0.75),
            p_second=((0.8, 0.2), (0.35, 0.65)),
            q_second=((0.5, 0.5), (0.8, 0.2)),
            rewards=(
                ((1.0, -3.0), (2.0, 0.0)),
                ((4.0, -1.0), (-2.0, 5.0)),
            ),
        )


def _sigmoid(theta: float) -> float:
    if not math.isfinite(theta):
        raise ValueError("Player logit must be finite")
    return (1 / (1 + math.exp(-theta)) if theta >= 0
            else math.exp(theta) / (1 + math.exp(theta)))


def exact_toy_reward_gradient(game: ToyChanceTree, theta: float) -> tuple[float, float]:
    """Return E_p[R] and on-policy d/dtheta E_p[R] by exact enumeration."""
    pi1 = _sigmoid(theta)
    expectation = gradient = 0.0
    for action, pi in enumerate((1 - pi1, pi1)):
        score = action - pi1
        for first in (0, 1):
            for second in (0, 1):
                mass = pi * game.p_first[first] * game.p_second[first][second]
                term = mass * game.rewards[action][first][second]
                expectation += term
                gradient += term * score
    return expectation, gradient


def exact_proposal_weighted_gradient(
    game: ToyChanceTree, theta: float,
) -> tuple[float, float]:
    """Enumerate q trajectories, retaining complete p/q correction."""
    pi1 = _sigmoid(theta)
    result = gradient = 0.0
    first_law = ChanceLaw("toy-exact-v1", "toy-first", game.p_first, "SYNTHETIC_EXACT")
    for action, pi in enumerate((1 - pi1, pi1)):
        for first in (0, 1):
            law = ChanceLaw("toy-exact-v1", f"toy-second-{first}",
                            game.p_second[first], "SYNTHETIC_EXACT")
            first_logw = log_chance_ratio(
                first_law, game.q_first, event_id="toy-first", selected_index=first,
            )
            for second in (0, 1):
                second_logw = log_chance_ratio(
                    law, game.q_second[first],
                    event_id=f"toy-second-{first}", selected_index=second,
                )
                proposal_mass = pi * game.q_first[first] * game.q_second[first][second]
                term = proposal_mass * math.exp(first_logw + second_logw)
                term *= game.rewards[action][first][second]
                result += term
                gradient += term * (action - pi1)
    return result, gradient


def sample_proposal_weighted_gradient(
    game: ToyChanceTree, theta: float, *, seed: int, episodes: int,
) -> tuple[float, float]:
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("Positive episode count required")
    rng = random.Random(seed)
    pi1 = _sigmoid(theta)
    first_law = ChanceLaw("toy-exact-v1", "toy-first", game.p_first, "SYNTHETIC_EXACT")
    reward_total = gradient_total = 0.0
    for _ in range(episodes):
        action = int(rng.random() < pi1)
        first = int(rng.random() >= game.q_first[0])
        second = int(rng.random() >= game.q_second[first][0])
        second_law = ChanceLaw(
            "toy-exact-v1", f"toy-second-{first}",
            game.p_second[first], "SYNTHETIC_EXACT",
        )
        logw = log_chance_ratio(
            first_law, game.q_first, event_id="toy-first", selected_index=first,
        ) + log_chance_ratio(
            second_law, game.q_second[first],
            event_id=f"toy-second-{first}", selected_index=second,
        )
        term = math.exp(logw) * game.rewards[action][first][second]
        reward_total += term
        gradient_total += term * (action - pi1)
    return reward_total / episodes, gradient_total / episodes
