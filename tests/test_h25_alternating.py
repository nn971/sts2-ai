"""H25 split-policy invariants and fail-closed historical combat replay."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
)
from sts2_ai.training.h24_joint_act_model import H24JointActPilotNet
from sts2_ai.training.h25_alternating import (
    CombatSearchExample,
    PrefixReplayMismatch,
    SplitPilotPolicy,
    fit_combat_search,
    fit_macro_reinforce,
    macro_future_floor_targets,
    replay_to_combat_start,
    sample_fight_starts,
)


def _state(*, combat=None, hp=55):
    return {
        "hp": hp, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [], "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3,
        "map": [], "current_map_node_id": None, "combat": combat,
        "event": {
            "event_id": "native.ancient_temple",
            "offered_choice_ids": ["gold", "heal"],
            "visible_choice_values": [
                {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6},
                {"choice_id": "heal", "heal": 15},
            ],
        },
    }


def _actions():
    return (
        LegalAction("choice-a", "event_choice", '{"ChoiceId":"gold"}'),
        LegalAction("choice-b", "event_choice", '{"ChoiceId":"heal"}'),
    )


def _observation(state):
    return Observation("prototype-fair-v0", json.dumps(state), "h25-test")


def _models():
    torch.manual_seed(25)
    baseline = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, heads=4, layers=1, feedforward=64)
    )).eval()
    macro = H24JointActPilotNet.warmstart_h22(
        baseline, act_only=True, search_act_mix=1.0, macro_progress=True,
    )
    combat = copy.deepcopy(macro)
    combat.macro_progress_head = None
    return macro, combat


class _DummyPolicy:
    def __init__(self, value: float, score: float):
        self.value = value
        self.score = score
        self.temperature = 0.85
        self.combat_credit_weight = 0.25
        self.loss_progress_cap = 0.2
        self.inference_calls = 0
        self.inference_seconds = 0.0

    def infer(self, observation, actions):
        self.inference_calls += 1
        return tuple([self.score] * len(actions)), self.value, tuple([0.] * len(actions))


def test_h25_split_macro_value_untouched_combat_value_has_hp_prior():
    macro = _DummyPolicy(0.2, score=1.)
    combat = _DummyPolicy(0.9, score=3.)
    policy = SplitPilotPolicy(macro, combat, hp_bias=0.25)
    scores, value, _ = policy.infer(_observation(_state()), _actions())
    assert scores == (1., 1.) and value == pytest.approx(0.2)
    combat_state = _state(combat={"phase": "player"})
    scores, value, _ = policy.infer(_observation(combat_state), _actions())
    assert scores == (3., 3.)
    assert value == pytest.approx(0.75 * 0.2 + 0.25 * 55 / 70)
    # After combat, the continuation has the combat-only HP bonus.
    assert policy.value_at(_observation(_state()), _actions()) == pytest.approx(value)
    assert policy.inference_calls == 4
    policy.inference_calls = 0
    assert policy.inference_calls == 0


def test_h25_macro_reinforce_does_not_change_combat_parameters():
    macro, combat = _models()
    actions = _actions()
    state = _state()
    decision = PilotDecision(_observation(state), actions, 0, (0., 0.))
    ep = PilotEpisode(
        "fake-seed", (decision,), (state, {**state, "hp": 0}),
        outcome="defeat", searches=0, search_transitions=0,
    )
    before = [x.detach().clone() for x in combat.parameters()]
    optimizer = torch.optim.AdamW(macro.parameters(), lr=1e-4)
    result = fit_macro_reinforce(macro, optimizer, [ep], minibatch=1)
    assert result["steps"] == 1
    assert result["examples"] == 1
    for prior, post in zip(before, combat.parameters(), strict=True):
        torch.testing.assert_close(prior, post, rtol=0, atol=0)


def test_h25_combat_distillation_only_updates_combat():
    macro, combat = _models()
    actions = _actions()
    state = _state()
    before = [x.detach().clone() for x in macro.parameters()]
    examples = [CombatSearchExample(
        _observation(state), actions, (0.8, 0.2), "seed", 2,
    )]
    optimizer = torch.optim.AdamW(combat.parameters(), lr=1e-4)
    report = fit_combat_search(combat, optimizer, examples, minibatch=1)
    assert report["steps"] == 1
    for prior, post in zip(before, macro.parameters(), strict=True):
        torch.testing.assert_close(prior, post, rtol=0, atol=0)


def test_h25_progress_labels_are_dense_after_defeat():
    state = _state()
    actions = _actions()
    observation = _observation(state)
    episode = PilotEpisode(
        "defeat-floor10",
        (PilotDecision(observation, actions, 0, (0., 0.)),),
        (state, {**state, "floor": 10, "hp": 0}),
        outcome="defeat", searches=0, search_transitions=0,
    )
    assert macro_future_floor_targets(episode) == pytest.approx((0.4,))

    early = PilotEpisode(
        "defeat-floor6",
        (PilotDecision(observation, actions, 0, (0., 0.)),),
        (state, {**state, "hp": 0}),
        outcome="defeat", searches=0, search_transitions=0,
    )
    assert macro_future_floor_targets(early) == (0.0,)


def test_h25_macro_progress_head_trains_from_only_defeats(tmp_path):
    macro, combat = _models()
    assert macro.macro_progress_head is not None
    assert combat.macro_progress_head is None
    state = _state()
    actions = _actions()
    observation = _observation(state)
    episodes = [
        PilotEpisode(
            f"loss-{floor}",
            (PilotDecision(observation, actions, 0, (0., 0.)),),
            (state, {**state, "floor": floor, "hp": 0}),
            outcome="defeat", searches=0, search_transitions=0,
        )
        for floor in (6, 10)
    ]
    previous = macro.macro_progress_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(macro.parameters(), lr=1e-3)
    report = fit_macro_reinforce(
        macro, optimizer, episodes, minibatch=2,
        progress_weight=0.25, progress_loss_weight=0.5,
    )
    assert report["examples"] == 2
    assert report["mean_progress_target"] == pytest.approx(0.2)
    assert report["progress_loss"] > 0
    assert not torch.equal(previous, macro.macro_progress_head.weight)
    macro.eval()
    macro.save(tmp_path / "macro-with-progress.pt")
    restored = JointActPilotNet.load(tmp_path / "macro-with-progress.pt")
    assert restored.macro_progress_head is not None
    batch = macro.collate([macro.encode_decision(state, actions)])
    with torch.inference_mode():
        expected = macro.forward_macro_training(batch)
        received = restored.forward_macro_training(batch)
    for x, y in zip(expected, received, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_h25_fight_sampler_and_exact_prefix_replay(monkeypatch):
    from sts2_ai.training import h25_alternating as h25

    initial, inside, terminal = (
        _state(), _state(combat={"turn": 1}), _state(hp=0),
    )
    first, second = _actions()[0], LegalAction(
        "end-turn", "end_turn", "{}",
    )
    episode = PilotEpisode(
        "seed", (
            PilotDecision(_observation(initial), (first,), 0, (0.,)),
            PilotDecision(_observation(inside), (second,), 0, (0.,)),
        ), (initial, inside, terminal),
        outcome="defeat", searches=0, search_transitions=0,
    )
    assert sample_fight_starts(episode, count=2, seed=5) == (1,)
    assert sample_fight_starts(episode, count=0, seed=5) == ()

    class Stub:
        def __init__(self):
            self.released = []

        def observe(self, state, policy):
            return _observation((initial, inside, terminal)[state])

        def legal_actions(self, state):
            return ((first,), (second,), ())[state]

        def step(self, state, action):
            return SimpleNamespace(child=state + 1)

        def release_many(self, handles):
            self.released.extend(handles)

    monkeypatch.setattr(h25, "reset_training_run", lambda *a, **k: 0)
    backend = Stub()
    state, observed, legal = replay_to_combat_start(
        backend, episode, 1, ascension=0,
    )
    assert state == 1 and legal == (second,)
    assert json.loads(observed.payload_json) == inside
    assert backend.released == [0]
    backend.release_many([state])
    backend.observe = lambda state, policy: _observation(_state(hp=34))
    with pytest.raises(PrefixReplayMismatch, match="observation diverges"):
        replay_to_combat_start(backend, episode, 1, ascension=0)
