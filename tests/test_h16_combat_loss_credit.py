"""H16 v5: directly recorded floor performance and bounded combat defeat credit."""
from __future__ import annotations

import json
import random

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition
from sts2_ai.search.combat_puct import CombatFrame
from sts2_ai.training.h16_episode_metrics import (
    combat_results,
    enemy_hp_progress,
    episode_floor_record,
    summarize_floor_progress,
)
from sts2_ai.training.h16_joint_act_pilot import (
    NativeRootConditionedSampler,
    combat_shaped_hp_return,
)


def enemy(ref: int, hp: int, max_hp: int, *, minion: bool = False,
          identity: str = "proto.enemy.main") -> dict:
    return {
        "instance_id": ref, "enemy_id": identity,
        "hp": hp, "max_hp": max_hp, "is_minion": minion,
    }


def frame(floor: int, hp: int, enemies=None, *, completed=()) -> dict:
    return {
        "floor": floor, "act": 1, "hp": hp, "max_hp": 70,
        "combat": {"turn": 1, "enemies": enemies}
        if enemies is not None else None,
        "completed_rooms": [
            {"act": 1, "floor": f} for f in completed
        ],
    }


def test_defeat_progress_excludes_summoned_minions_and_preserves_win_margin():
    history = [
        frame(10, 40, [enemy(1, 100, 100), enemy(2, 40, 40, minion=True)]),
        frame(10, 20, [enemy(1, 35, 100), enemy(2, 0, 40, minion=True)]),
        frame(10, 0, None),
    ]
    loss = combat_results(history, "defeat")
    assert len(loss) == 1
    assert loss[0].won is False
    assert loss[0].enemy_progress == pytest.approx(0.65)
    assert loss[0].quality() == pytest.approx(0.13)
    assert combat_shaped_hp_return(
        [0.0], 0.0, 0.0, combat_quality=loss[0].quality(),
    ) == pytest.approx(0.0325)

    win = combat_results([
        frame(10, 40, [enemy(1, 100, 100)]),
        frame(10, 1, None, completed=(10,)),
    ], "defeat")
    assert len(win) == 1 and win[0].won
    assert win[0].quality() == 1.0
    # Even at a subsequent Act defeat with lambda=1, this combat win gets
    # full combat-outcome credit, larger than any possible defeated run.
    assert combat_shaped_hp_return(
        [0.0], 0.0, 1.0, combat_quality=win[0].quality(),
    ) == pytest.approx(0.25)
    assert combat_shaped_hp_return(
        [0.0], 0.0, 1.0, combat_quality=0.2,
    ) == pytest.approx(0.05)


def test_defeated_combat_gets_more_credit_for_more_damage_without_healing_farm():
    low = enemy_hp_progress([
        frame(5, 20, [enemy(1, 100, 100)]),
        frame(5, 5, [enemy(1, 75, 100)]),
    ])
    high = enemy_hp_progress([
        frame(5, 20, [enemy(1, 100, 100)]),
        frame(5, 5, [enemy(1, 25, 100)]),
    ])
    healing = enemy_hp_progress([
        frame(5, 20, [enemy(1, 100, 100)]),
        frame(5, 10, [enemy(1, 20, 100)]),
        frame(5, 5, [enemy(1, 60, 100)]),
    ])
    assert (low, high, healing) == pytest.approx((0.25, 0.75, 0.4))


def test_multiphase_nonminion_progress_counts_completed_earlier_stage():
    progress = enemy_hp_progress([
        frame(16, 55, [enemy(1, 100, 100, identity="boss.phase.one")]),
        frame(16, 40, [enemy(1, 30, 100, identity="boss.phase.one")]),
        frame(16, 20, [enemy(1, 200, 200, identity="boss.phase.two")]),
        frame(16, 5, [enemy(1, 60, 200, identity="boss.phase.two")]),
    ])
    assert progress == pytest.approx(0.8)


def test_max_hp_change_within_one_identity_is_not_a_defeated_phase():
    # Buffing max HP within the same phase must not retire the old health bar.
    progress = enemy_hp_progress([
        frame(8, 50, [enemy(1, 100, 100)]),
        frame(8, 40, [enemy(1, 130, 150)]),
    ])
    assert progress == pytest.approx(20 / 150)
    # A reduced maximum also does not certify a phase transition.
    shrinking = enemy_hp_progress([
        frame(8, 50, [enemy(1, 100, 100)]),
        frame(8, 40, [enemy(1, 70, 80)]),
    ])
    assert shrinking == pytest.approx(0.3)


def test_act_clear_with_terminal_combat_frame_counts_as_boss_victory():
    result = combat_results([
        frame(16, 30, [enemy(1, 50, 50)]),
        frame(16, 20, [enemy(1, 0, 50)]),
    ], "act_clear")
    assert len(result) == 1
    assert result[0].won
    assert result[0].quality() == 1.0


def test_floor_progress_is_written_per_seed_with_death_histogram_and_hp():
    first = episode_floor_record([
        frame(1, 50),
        frame(1, 40, None, completed=(1,)),
        frame(2, 0),
    ], "run-1", "defeat", 2)
    second = episode_floor_record([
        frame(1, 60),
        frame(1, 55, None, completed=(1,)),
        frame(2, 50, None, completed=(1, 2)),
    ], "run-2", "act_clear", 2)
    record = summarize_floor_progress([first, second])
    assert first["death_floor"] == 2
    assert record["mean_max_floor_reached"] == 2
    assert record["death_floors"] == {"2": 1}
    assert record["reached_floors"]["2"] == 2
    assert record["completed_floors"]["2"] == 1
    assert record["per_floor_hp"]["1"]["completions"] == 2
    assert record["per_floor_hp"]["1"]["mean_hp_fraction"] == pytest.approx(
        (40 + 55) / 140,
    )
    assert len(record["runs"]) == 2


class TerminalBackend:
    def __init__(self, *, win: bool):
        self.win = win
        self.released = []

    def fork(self, state):
        return "simulation-root"

    def step(self, state, action):
        return Transition(
            state, action, "simulation-child", not self.win,
        )

    def observe(self, state, policy):
        payload = frame(
            7, 10 if self.win else 0, None,
            completed=(7,) if self.win else (),
        )
        return Observation(policy.policy_id, json.dumps(payload), "terminal-hash")

    def legal_actions(self, state):
        return (LegalAction("continue", "map_choice", "{}"),)

    def release_many(self, states):
        self.released.extend(states)


class TerminalPolicy:
    combat_credit_weight = 0.25
    loss_progress_cap = 0.2

    def value_at(self, observation, actions, *, terminal=False):
        return 0.0


@pytest.mark.parametrize("win", [False, True])
def test_puct_terminal_combat_win_strictly_dominates_loss(win: bool):
    backend = TerminalBackend(win=win)
    policy = TerminalPolicy()
    root_obs = Observation(
        FAIR_POLICY_ID,
        json.dumps(frame(7, 30, [enemy(1, 100, 100)])),
        "root-hash",
    )
    act = LegalAction("hit", "play_card", "{}")
    root = CombatFrame("root", root_obs, (act,))
    sampler = NativeRootConditionedSampler(
        backend, "real-root", policy, rollout_transport="legacy",
    )
    sampler.begin_simulation(root, random.Random(7))
    try:
        # Root is at 100 HP; insert a later observed 20-HP state before
        # the terminal no-combat state, modeling a branch that dealt 80.
        sampler._combat_frames.append(frame(7, 25, [enemy(1, 20, 100)]))
        result = sampler.step(root, act, random.Random(7))
        assert result.terminal_reward is not None
        assert result.terminal_reward / 2 == pytest.approx(
            0.25 if win else 0.25 * 0.2 * 0.8,
        )
    finally:
        sampler.end_simulation()
