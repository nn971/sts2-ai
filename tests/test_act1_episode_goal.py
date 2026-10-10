"""Act-1 termination is certified from public boss completion, never a floor guess."""
from __future__ import annotations

import json
import random
from typing import Any

import pytest

from sts2_ai.agents import RandomAgent
from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.emulator.episode_goal import (
    NATIVE_ACT1_BOSS_GOAL, PROTOTYPE_THREE_ACT_GOAL,
    certified_native_act1_clear,
)
from sts2_ai.emulator.run_environment import NATIVE_MAP_PROFILE, NATIVE_RESET_SCHEMA
from sts2_ai.evaluation.run import play_run
from sts2_ai.training.selfplay import collect_public_episode
from test_neural_selfplay import zero_model

POLICY = InformationPolicy("prototype-fair-v0")
BOSS_RECORD = {"act": 1, "floor": 16, "room_type": 5, "node_id": "boss"}


def _root() -> dict[str, Any]:
    return {
        "act": 1, "floor": 0, "phase": 7, "hp": 70, "max_hp": 70,
        "event_id": "proto.native.event.neow",
        "map_generation_profile_id": NATIVE_MAP_PROFILE,
        "map": [{"floor": 16, "node_id": "boss", "room_type": 5}],
        "deck": [
            {"card_id": "proto.common.restlessness"},
            *[{"card_id": "proto.silent.strike"} for _ in range(12)],
        ],
        "potions": [], "relics": [],
        "completed_rooms": [],
        "combat": None,
    }


def _boss() -> dict[str, Any]:
    return {
        "act": 1, "floor": 16, "phase": 3,
        "hp": 20, "max_hp": 70, "potions": [], "relics": [],
        "completed_rooms": [],
        "combat": {
            "turn": 3, "energy": 2, "player_block": 0,
            "enemies": [{"instance_id": 3, "enemy_id": "kin", "hp": 12}],
            "hand": [], "draw_pile_count": 0, "draw_pile": [],
            "discard_pile": [], "exhaust_pile": [], "relic_counters": [],
            "player_powers": [],
        },
    }


def _reward() -> dict[str, Any]:
    return {
        "act": 1, "floor": 16, "phase": 5,
        "hp": 20, "max_hp": 70, "potions": [], "relics": [],
        "completed_rooms": [], "combat": None,
    }


def _clear() -> dict[str, Any]:
    return {
        "act": 1, "floor": 16, "phase": 9,
        "hp": 20, "max_hp": 70, "potions": [], "relics": [],
        "completed_rooms": [BOSS_RECORD], "combat": None,
    }


class NativeScriptedBackend:
    """Public-only scripted backend that can model a nonterminal act boundary."""

    native_overgrowth_reset_schema = NATIVE_RESET_SCHEMA
    emulator_revision = "scripted-overgrowth-public-goal-v1"

    def __init__(self, frames: list[dict[str, Any]] | None = None) -> None:
        self.frames = frames if frames is not None else [
            _root(), _boss(), _reward(), _clear(),
            {"act": 2, "floor": 1, "hp": 20, "max_hp": 70, "combat": None},
        ]
        self.handles: set[str] = set()
        self.steps = 0

    def reset_native_overgrowth(self, seed: str, ascension: int = 0) -> str:
        assert ascension == 0
        self.handles.add("0")
        return "0"

    def reset(self, seed: str, ascension: int = 0) -> str:
        raise AssertionError("native mode must not use legacy reset")

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        payload = json.dumps(self.frames[int(state)], sort_keys=True)
        return Observation(policy.policy_id, payload, payload)

    def is_terminal(self, state: str) -> bool:
        return self.frames[int(state)].get("terminal_outcome") in ("victory", "defeat")

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        if self.is_terminal(state):
            return ()
        return (LegalAction("advance", "advance", "{}"),)

    def step(self, state: str, action: LegalAction) -> Transition:
        assert action.action_id == "advance"
        next_index = int(state) + 1
        if next_index >= len(self.frames):
            raise AssertionError("Act-1 goal should have stopped before this step")
        self.steps += 1
        child = str(next_index)
        self.handles.add(child)
        return Transition(state, action, child, self.is_terminal(child))

    def release_many(self, states: Any) -> int:
        count = 0
        for state in states:
            self.handles.remove(state)
            count += 1
        return count


def test_certification_requires_actual_completed_boss_and_transition() -> None:
    assert certified_native_act1_clear(_clear())
    fake = [
        {**_clear(), "phase": 5},
        {**_clear(), "floor": 15},
        {**_clear(), "completed_rooms": []},
        {**_clear(), "completed_rooms": [{**BOSS_RECORD, "room_type": 0}]},
        {**_clear(), "combat": _boss()["combat"]},
        {**_clear(), "terminal_outcome": "defeat"},
        {"act": 2, "floor": 1, "phase": 2, "completed_rooms": [BOSS_RECORD]},
    ]
    assert not any(certified_native_act1_clear(frame) for frame in fake)


def test_training_and_evaluation_stop_at_same_certified_boundary() -> None:
    train_backend = NativeScriptedBackend()
    episode = collect_public_episode(
        train_backend, zero_model(), seed="native-success",
        actor_rng=random.Random(1), max_decisions=3,
        environment="native-overgrowth",
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
    )
    assert episode.completed and episode.won
    assert (episode.act, episode.floor) == (1, 16)
    assert len(episode.decisions) == 3
    assert len(episode.combat_outcomes) == 1
    assert episode.combat_outcomes[0].result == "victory"
    assert episode.boss_progress is not None
    assert episode.boss_progress.remaining_hp == 0
    assert train_backend.steps == 3
    assert not train_backend.handles

    eval_backend = NativeScriptedBackend()
    result = play_run(
        eval_backend, RandomAgent(seed=5),
        seed="native-success", policy=POLICY, max_decisions=3,
        environment="native-overgrowth",
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
    )
    assert result.won and result.outcome == "victory"
    assert result.episode_goal_version == NATIVE_ACT1_BOSS_GOAL
    assert result.act1_cleared is True
    assert result.full_game_victory is None
    assert result.censored is False
    assert (result.terminal_act, result.terminal_floor) == (1, 16)
    assert eval_backend.steps == 3
    assert not eval_backend.handles


def test_act1_goal_never_labels_death_or_censoring_as_victory() -> None:
    death = NativeScriptedBackend([
        _root(), _boss(),
        {"act": 1, "floor": 16, "hp": 0, "max_hp": 70,
         "combat": None, "terminal_outcome": "defeat"},
    ])
    episode = collect_public_episode(
        death, zero_model(), seed="death", actor_rng=random.Random(0),
        environment="native-overgrowth",
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
    )
    assert episode.completed and episode.outcome == "defeat"
    assert not death.handles

    incomplete = NativeScriptedBackend()
    summary = play_run(
        incomplete, RandomAgent(seed=0), seed="cutoff", policy=POLICY,
        environment="native-overgrowth",
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        max_decisions=2,
    )
    assert summary.outcome == "truncated"
    assert summary.censored and summary.act1_cleared is None
    assert not incomplete.handles

    missing = NativeScriptedBackend([
        _root(), _boss(), _reward(), {**_clear(), "completed_rooms": []},
    ])
    episode = collect_public_episode(
        missing, zero_model(), seed="false-clear",
        actor_rng=random.Random(0), max_decisions=3,
        environment="native-overgrowth",
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
    )
    assert episode.outcome == "truncated" and not episode.completed
    assert not missing.handles


def test_goals_cannot_be_mixed_with_wrong_environment() -> None:
    with pytest.raises(ValueError, match="requires native-overgrowth"):
        play_run(
            NativeScriptedBackend(), RandomAgent(seed=2),
            seed="wrong", policy=POLICY,
            episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        )


def test_act1_ppo_checkpoint_refuses_old_full_game_objective(tmp_path) -> None:
    pytest.importorskip("torch")
    from sts2_ai.training.phase_split_selfplay import train_phase_split

    options = dict(
        rounds=1, episodes_per_round=2,
        dimension=64, hidden=4, max_decisions=8, workers=1,
        environment="native-overgrowth", seed=21,
        optimizer_method="ppo", ppo_epochs=1, ppo_batch_size=2,
        ppo_sample_limit=16, hp_monotonic_weight=0.0,
        tactical_state_encoding="public_resources",
        combat_objective="hp_preservation",
        checkpoint=tmp_path / "goal-checkpoint.pt",
    )
    model, rows = train_phase_split(
        NativeScriptedBackend(), episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        **options,
    )
    assert model.combat.format_id.endswith("public-resources-tactical")
    assert len(rows) == 1
    assert rows[0].wins == 2
    assert rows[0].censored == 0
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            NativeScriptedBackend(), episode_goal_version=PROTOTYPE_THREE_ACT_GOAL,
            resume=True, **{**options, "rounds": 2},
        )



def test_goal_rejects_uncertified_late_prototype_terminal() -> None:
    frames = [
        _root(),
        {"act": 3, "floor": 6, "hp": 20, "max_hp": 70,
         "terminal_outcome": "victory", "combat": None},
    ]
    for action in ("train", "eval"):
        backend = NativeScriptedBackend(frames)
        with pytest.raises(ValueError, match="without certified"):
            if action == "train":
                collect_public_episode(
                    backend, zero_model(), seed="late-terminal",
                    actor_rng=random.Random(1), environment="native-overgrowth",
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                )
            else:
                play_run(
                    backend, RandomAgent(seed=1), seed="late-terminal",
                    policy=POLICY, environment="native-overgrowth",
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                )
        assert not backend.handles


def test_longrun_summary_counts_certified_act1_floor16_victory() -> None:
    from tools.train_longrun import summarize
    report = {
        "rounds": [{"wins": 4, "optimization_steps": 8}],
        "heldout": [
            {"outcome": "victory", "act": 1, "floor": 16,
             "act1_cleared": True, "censored": False, "progress": 16.0,
             "boss": {"initial_hp": 200, "remaining_hp": 0}},
            {"outcome": "defeat", "act": 1, "floor": 16,
             "act1_cleared": False, "censored": False, "progress": 15.2,
             "boss": {"initial_hp": 200, "remaining_hp": 100}},
        ],
    }
    metrics = summarize(report)
    assert metrics["act1_clears"] == 1
    assert metrics["boss_entries"] == 2
    assert metrics["training_wins"] == 4
