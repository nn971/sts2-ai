from __future__ import annotations

import json
from typing import Any

from sts2_ai.agents import RandomAgent
from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.evaluation.run import play_run


class ScriptedBackend:
    """Tiny public-observation-only episode for goal and censoring regression."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = frames

    def reset(self, seed: str, ascension: int = 0) -> str:
        return "0"

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        return Observation(policy.policy_id, json.dumps(self.frames[int(state)]), state)

    def is_terminal(self, state: str) -> bool:
        return int(state) == len(self.frames) - 1

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        return (LegalAction(action_id="next", kind="advance"),)

    def step(self, state: str, action: LegalAction) -> Transition:
        child = str(int(state) + 1)
        return Transition(state, action, child, self.is_terminal(child))

    def release_many(self, states: list[str]) -> int:
        return len(states)


POLICY = InformationPolicy("prototype-fair-v0")


def test_act1_defeat_is_not_clear_or_full_victory() -> None:
    summary = play_run(
        ScriptedBackend([
            {"act": 1, "floor": 1, "hp": 70},
            {"act": 1, "floor": 1, "hp": 0, "terminal_outcome": "defeat"},
        ]),
        RandomAgent(0),
        seed="one",
        policy=POLICY,
    )
    assert summary.act1_cleared is False
    assert summary.full_game_victory is False
    assert summary.censored is False
    assert summary.episode_goal_version == "prototype-three-act-v0"


def test_known_act1_clear_with_censored_full_game() -> None:
    summary = play_run(
        ScriptedBackend([
            {"act": 1, "floor": 6, "hp": 50},
            {"act": 2, "floor": 1, "hp": 40},
            {"act": 2, "floor": 2, "hp": 0, "terminal_outcome": "defeat"},
        ]),
        RandomAgent(0),
        seed="two",
        policy=POLICY,
        max_decisions=1,
    )
    assert summary.outcome == "truncated"
    assert summary.act1_cleared is True
    assert summary.full_game_victory is None
    assert summary.censored is True


def test_act1_truncation_has_unknown_clear_label() -> None:
    summary = play_run(
        ScriptedBackend([
            {"act": 1, "floor": 2, "hp": 70},
            {"act": 1, "floor": 3, "hp": 55, "terminal_outcome": "defeat"},
        ]),
        RandomAgent(0),
        seed="three",
        policy=POLICY,
        max_decisions=0,
    )
    assert summary.act1_cleared is None
    assert summary.full_game_victory is None
    assert summary.censored is True


def test_full_game_victory_is_distinct_explicit_label() -> None:
    summary = play_run(
        ScriptedBackend([
            {"act": 3, "floor": 5, "hp": 15},
            {"act": 3, "floor": 6, "hp": 2, "terminal_outcome": "victory"},
        ]),
        RandomAgent(0),
        seed="four",
        policy=POLICY,
    )
    assert summary.act1_cleared is True
    assert summary.full_game_victory is True
    assert summary.censored is False
