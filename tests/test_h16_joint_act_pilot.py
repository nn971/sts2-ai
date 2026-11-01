"""Focused H16 joint run / delayed-HP / experimental local search tests."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition
from sts2_ai.emulator.run_environment import (
    EXPERIMENTAL_FIXED_ROUTE_PROFILE, validate_native_start,
)
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet, PilotDecision, PilotEpisode, floor_completed,
    hp_at_future_floor, local_combat_puct, targets_for_episode, train_joint_cohort,
)


def obs(frame: dict) -> Observation:
    payload = json.dumps(frame, sort_keys=True)
    return Observation(FAIR_POLICY_ID, payload, hashlib.sha256(payload.encode()).hexdigest())


ACTIONS = (LegalAction("pass", "end_turn"), LegalAction("play", "other_legal"))


def frame(floor: int, hp: int, completed: tuple[int, ...] = (), *,
          combat: bool = False) -> dict:
    return {
        "act": 1, "floor": floor, "hp": hp, "max_hp": 70,
        "completed_rooms": [
            {"act": 1, "floor": i, "room_type": 0} for i in completed
        ],
        "combat": {"turn": 1, "enemies": []} if combat else None,
        "deck": [], "relics": [], "potions": [], "map": [],
    }


def test_hp_target_follows_future_completed_floor_not_current_hp():
    states = (
        frame(4, 55, (1, 2, 3)),  # decision before combat / resources spent
        frame(5, 44, (1, 2, 3, 4)),
        frame(6, 37, (1, 2, 3, 4, 5)),
        frame(7, 41, (1, 2, 3, 4, 5, 6)),
    )
    assert hp_at_future_floor(states, 0, horizon=2) == pytest.approx(41 / 70)
    assert hp_at_future_floor(states, 0, horizon=1) == pytest.approx(37 / 70)
    assert hp_at_future_floor(states[:3], 0, horizon=2) == 0.0
    assert floor_completed(states[-1], 6)


def test_early_death_future_hp_is_zero_and_binary_label_is_act_level():
    d = PilotDecision(obs(frame(4, 55, combat=True)), ACTIONS, 0, (0., 0.))
    ep = PilotEpisode(
        "death", (d,), (frame(4, 55), frame(4, 0)),
        "defeat", 0, 0,
    )
    assert targets_for_episode(ep, 2) == ((0.0, 0.0),)
    with pytest.raises(ValueError, match="Censored"):
        targets_for_episode(PilotEpisode(
            "censored", (d,), ep.frames, "censored", 0, 0,
        ), 2)


def test_opt_in_fixed_map_does_not_weaken_native_default_validation():
    state = frame(0, 56)
    state.update(
        map_generation_profile_id=EXPERIMENTAL_FIXED_ROUTE_PROFILE,
        event_id="proto.native.event.neow",
        deck=[{"card_id": "proto.silent.strike"} for _ in range(12)] +
             [{"card_id": "proto.curse.ascenders_bane"}],
        map=[{"floor": n} for n in range(1, 17)],
    )
    with pytest.raises(Exception, match="Native Overgrowth reset failed"):
        validate_native_start(obs(state), ascension=10)
    validate_native_start(
        obs(state), ascension=10,
        expected_map_profile=EXPERIMENTAL_FIXED_ROUTE_PROFILE,
    )


def test_one_joint_optimizer_updates_both_phases_and_continuation_heads():
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    original = {name: v.detach().clone() for name, v in model.named_parameters()}
    combat_obs = obs(frame(1, 56, combat=True))
    macro_obs = obs(frame(2, 50))
    d0 = PilotDecision(combat_obs, ACTIONS, 0, (0., 0.))
    d1 = PilotDecision(
        macro_obs, ACTIONS, 1, (0., 0.), search_policy=None,
    )
    cleared = PilotEpisode(
        "clear", (d0, d1),
        (frame(1, 56), frame(2, 50, (1,)), frame(16, 32, tuple(range(1, 17)))),
        "act_clear", 0, 0,
    )
    lost = PilotEpisode(
        "defeat", (d0, d1),
        (frame(1, 56), frame(2, 48, (1,)), frame(2, 0, (1,))),
        "defeat", 0, 0,
    )
    torch.set_num_threads(1)
    result = train_joint_cohort(
        model, optimizer, [cleared, lost], alpha=0.35, horizon=2,
        temperature=1.0, minibatch=4, epochs=1,
    )
    assert result["optimizer_steps"] == 1
    assert result["combat_decisions"] == 2
    assert result["macro_decisions"] == 2
    assert any(not torch.equal(original[name], value)
               for name, value in model.named_parameters()
               if name.startswith("backbone.policy_heads.0"))
    assert any(not torch.equal(original[name], value)
               for name, value in model.named_parameters()
               if name.startswith("backbone.policy_heads.1"))
    assert any(not torch.equal(original[name], value)
               for name, value in model.named_parameters()
               if name.startswith("act_clear"))
    assert any(not torch.equal(original[name], value)
               for name, value in model.named_parameters()
               if name.startswith("future_hp"))


class ForkBackend:
    def __init__(self):
        self.counter = 0
        self.released = []
        self.step_count = 0

    def fork(self, root):
        assert root == "root"
        self.counter += 1
        return f"fork-{self.counter}"

    def step(self, source, action):
        assert source.startswith("fork-")
        self.step_count += 1
        return Transition(source, action, f"{source}/{action.action_id}", False)

    def observe(self, state, policy):
        assert policy.policy_id == FAIR_POLICY_ID
        return obs(frame(2, 30 if state.endswith("/pass") else 60, (1,)))

    def legal_actions(self, state):
        return (LegalAction("reward", "choose_reward"),)

    def release_many(self, handles):
        self.released.extend(handles)
        return len(handles)


class ValueStub:
    def infer(self, observation, actions):
        return (tuple(0.0 for _ in actions), 0.5,
                tuple(0.0 for _ in actions))

    def value_at(self, observation, actions, *, terminal=False):
        if terminal:
            return 0.0
        return json.loads(observation.payload_json)["hp"] / 70


def test_local_puct_is_short_root_conditioned_and_releases_all_forks():
    backend = ForkBackend()
    policy = ValueStub()
    root = obs(frame(1, 56, combat=True))
    visits, transitions = local_combat_puct(
        backend, "root", root, ACTIONS, policy,
        simulations=24, depth=2, seed=42,
    )
    assert len(visits) == 2
    assert sum(visits) == pytest.approx(1.0)
    assert visits[1] > visits[0]
    assert transitions == backend.step_count
    assert len(backend.released) == backend.counter + backend.step_count
