"""Focused H16 joint run / delayed-HP / experimental local search tests."""
from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch

from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition
from sts2_ai.emulator.run_environment import (
    EXPERIMENTAL_FIXED_ROUTE_PROFILE,
    validate_native_start,
)
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer, collate_h7
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
    completed_room_hp,
    floor_completed,
    geometric_hp_return,
    remaining_room_hp,
    hp_after_current_room,
    hp_at_future_floor,
    local_combat_puct,
    targets_for_episode,
    train_joint_cohort,
)
from sts2_ai.training.h16_public_codec import encode_h16_decision


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


def test_post_room_hp_depends_on_room_actions_not_predecision_hp():
    # Two possible outcomes from the same combat-start observation.
    start = frame(4, 55, (1, 2, 3), combat=True)
    for room_end_hp in (24, 48):
        states = (
            start,
            frame(4, 51, (1, 2, 3), combat=True),
            frame(4, room_end_hp, (1, 2, 3, 4)),
            frame(5, room_end_hp, (1, 2, 3, 4)),
        )
        assert hp_after_current_room(states, 0) == pytest.approx(room_end_hp / 70)
    # An already-completed room means the decision belongs to the next
    # room's outcome; repeating its initial HP would be action-independent.
    map_choice = (
        frame(4, 55, (1, 2, 3, 4)),
        frame(5, 55, (1, 2, 3, 4), combat=True),
        frame(5, 32, (1, 2, 3, 4, 5)),
    )
    assert hp_after_current_room(map_choice, 0) == pytest.approx(32 / 70)
    assert hp_after_current_room(map_choice, 1) == pytest.approx(32 / 70)


def test_post_room_hp_handles_neow_death_and_act_boundary():
    neow = (
        frame(0, 70),
        frame(0, 70),
        frame(1, 60, (1,)),
    )
    assert hp_after_current_room(neow, 0) == pytest.approx(60 / 70)
    died_before_room_end = (
        frame(7, 51, tuple(range(1, 7)), combat=True),
        frame(7, 0, tuple(range(1, 7))),
    )
    assert hp_after_current_room(died_before_room_end, 0) == 0.0
    completed_boss = (
        frame(16, 30, tuple(range(1, 17))),
        frame(16, 30, tuple(range(1, 17))),
    )
    assert hp_after_current_room(completed_boss, 0) == pytest.approx(30 / 70)
    with pytest.raises(ValueError, match="Invalid post-room HP"):
        hp_after_current_room(neow, len(neow) - 1)


def test_geometric_lambda_is_continuous_and_has_correct_limits():
    hp = (0.80, 0.65, 0.52)
    assert geometric_hp_return(hp, 1.0, 0.0) == pytest.approx(0.80)
    assert geometric_hp_return(hp, 1.0, 1.0) == pytest.approx(1.0)
    assert geometric_hp_return((), 1.0, 0.42) == pytest.approx(1.0)
    # At lambda=1/2, weights are 1/2, 1/4, 1/8, 1/8.
    expected = 0.5 * 0.80 + 0.25 * 0.65 + 0.125 * 0.52 + 0.125
    assert geometric_hp_return(hp, 1.0, 0.5) == pytest.approx(expected)
    assert geometric_hp_return(hp, 0.0, 0.5) == pytest.approx(expected - 0.125)
    assert geometric_hp_return((0.65,), 0.0, 0.5) == pytest.approx(0.325)
    for a, b in ((0.0, 0.001), (0.15, 0.151), (0.99, 0.991)):
        assert (abs(geometric_hp_return(hp, 1.0, a)
                    - geometric_hp_return(hp, 1.0, b)) < 0.01)
    with pytest.raises(ValueError, match="Invalid geometric"):
        geometric_hp_return(hp, 1.0, 1.01)


def test_geometric_targets_use_first_room_completion_and_skip_current_completed():
    states = (
        frame(0, 70),
        frame(1, 60, (), combat=True),
        frame(1, 51, (1,)),
        frame(1, 65, (1,)),  # Later healing must not overwrite floor-1 HP.
        frame(2, 44, (1, 2)),
        frame(3, 0, (1, 2)),  # Defeat before floor 3 completion.
    )
    completed = completed_room_hp(states)
    assert completed[1] == pytest.approx(51 / 70)
    assert completed[2] == pytest.approx(44 / 70)
    opening = remaining_room_hp(states[0], completed)
    in_first_room = remaining_room_hp(states[1], completed)
    after_first_room = remaining_room_hp(states[2], completed)
    assert opening[0] == pytest.approx(51 / 70)
    assert in_first_room[0] == pytest.approx(51 / 70)
    assert after_first_room[0] == pytest.approx(44 / 70)
    assert after_first_room[1] == 0.0
    assert geometric_hp_return(after_first_room, 0.0, 0.0) == pytest.approx(44 / 70)
    assert geometric_hp_return(after_first_room, 0.0, 1.0) == 0.0


def test_horizon_schedule_starts_at_one_room_and_increases_slowly():
    # Importing the runner is intentional: enforce CLI schedule provenance.
    from tools.train_h16_joint_act_pilot import horizon_lambda_for_round

    assert horizon_lambda_for_round(0, 200) == 0.0
    assert horizon_lambda_for_round(1, 200) < 0.01
    assert horizon_lambda_for_round(39, 200) < 0.18
    assert horizon_lambda_for_round(99, 200) < 0.40
    assert horizon_lambda_for_round(499, 200) > 0.91
    with pytest.raises(ValueError):
        horizon_lambda_for_round(-1, 200)


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


def test_legacy_immediate_hp_option_is_rejected():
    # Old "immediate" meant pre-decision HP, which provides no
    # action-dependent policy-gradient signal.
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    with pytest.raises(ValueError, match="Invalid joint training configuration"):
        train_joint_cohort(
            model, optimizer, (), horizon_lambda=0.0, hp_target="immediate",
        )


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
    searched = PilotDecision(
        combat_obs, ACTIONS, 1, (0., 0.), search_policy=(0.1, 0.9),
    )
    cleared = PilotEpisode(
        "clear", (searched, d1),
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
        model, optimizer, [cleared, lost], horizon_lambda=0.35,
        temperature=1.0, minibatch=4, epochs=1,
        hp_target="post_room",
    )
    assert result["optimizer_steps"] == 1
    assert result["combat_decisions"] == 2
    assert result["macro_decisions"] == 2
    assert result["search_roots"] == 1
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
               if name.startswith("horizon_head"))


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


def test_h16_macro_options_and_relic_counters_are_visible_without_prices():
    state = frame(7, 45)
    state["relics"] = [{
        "relic_id": "demo.counter",
        "persistent_state": {"charges": 2, "nested": {"triggered": True}},
    }]
    choices = (
        LegalAction("left", "event_choice", '{"ChoiceId":"left"}'),
        LegalAction("right", "event_choice", '{"ChoiceId":"right"}'),
    )
    encoded = encode_h16_decision(state, choices)
    assert encoded.phase == "macro"
    assert encoded.actions[0].semantic != encoded.actions[1].semantic
    # Macro reward card IDs are public, but they are not hand-card pointers.
    rewards = (
        LegalAction("reward-left", "choose_card", '{"CardInstanceId":401}'),
        LegalAction("reward-right", "choose_card", '{"CardInstanceId":402}'),
    )
    reward_codec = encode_h16_decision(state, rewards)
    assert reward_codec.actions[0].semantic != reward_codec.actions[1].semantic
    assert len(encoded.objects) > len(
        encode_h16_decision({**state, "relics": []}, choices).objects
    )
    with_persisted = encode_h16_decision(
        {**state, "relics": [{
            "relic_id": "demo.counter",
            "persistent_state": {"charges": 5, "nested": {"triggered": False}},
        }]}, choices,
    )
    assert encoded.objects[-2].numeric != with_persisted.objects[-2].numeric
    combat = encode_h16_decision(
        {**state, "combat": {"turn": 1, "enemies": []}}, ACTIONS,
    )
    assert combat.phase == "combat"
    assert combat.actions[0].semantic == 0


def test_joint_model_checkpoint_preserves_codec_and_default_backbone_output(tmp_path):
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    ))
    batch = collate_h7([
        encode_h16_decision(frame(1, 56, combat=True), ACTIONS),
    ])
    original = model.eval()(batch)
    assert len(model.backbone(batch)) == 2
    assert len(original) == 3
    path = tmp_path / "joint.pt"
    model.save(path)
    restored = JointActPilotNet.load(path).eval()
    for expected, actual in zip(original, restored(batch), strict=True):
        assert torch.allclose(expected, actual, atol=1e-6)



def test_mixed_length_multibatch_update_remains_finite():
    """Regression for CUDA BCE failures after one corrupted optimizer step.

    The padded H7 action logits must not contaminate entropy gradients or
    cause a subsequent minibatch's sigmoid heads to become NaN.
    """
    torch.manual_seed(1281)
    torch.set_num_threads(1)
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    episodes = []
    for i in range(96):
        combat = i % 3 != 0
        state = frame(1 + i % 5, 46 + i % 15, combat=combat)
        state["deck"] = [
            {"card_id": f"demo.card.{j}", "cost": j % 3}
            for j in range(i % 11)
        ]
        state["relics"] = [
            {"relic_id": "demo.counter",
             "persistent_state": {"charges": i % 8}}
        ] if i % 4 == 0 else []
        legal = tuple(
            LegalAction(f"action-{j}", f"do-{j}", "{}")
            for j in range(1 + i % 7)
        )
        decision = PilotDecision(
            obs(state), legal, i % len(legal), (0.0,) * len(legal),
            search_policy=(
                tuple(1.0 / len(legal) for _ in legal)
                if i % 8 == 0 else None
            ),
        )
        final = frame(state["floor"], 0)
        episodes.append(PilotEpisode(
            f"stress-{i}", (decision,), (state, final),
            "defeat", 0, 0,
        ))
    metrics = train_joint_cohort(
        model, optimizer, episodes, horizon_lambda=0.2,
        minibatch=32, epochs=2, entropy_weight=0.001,
    )
    assert metrics["optimizer_steps"] == 6
    assert metrics["macro_decisions"] == 32
    assert metrics["combat_decisions"] == 64
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert all(
        isinstance(metrics[name], float)
        and math.isfinite(metrics[name])
        for name in ("loss", "policy", "act", "hp")
    )


def test_joint_head_logit_output_matches_inference_probabilities():
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    batch = collate_h7([
        encode_h16_decision(frame(2, 50, combat=True), ACTIONS),
    ])
    _, act_prob, hp_prob = model(batch)
    _, act_logit, hp_logit = model(batch, value_logits=True)
    assert torch.allclose(act_prob, act_logit.sigmoid())
    assert torch.allclose(hp_prob, hp_logit.sigmoid())


def test_joint_critic_horizon_conditioning_uses_single_backbone_call():
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    batch = collate_h7([
        encode_h16_decision(frame(2, 50, combat=True), ACTIONS),
        encode_h16_decision(frame(2, 50), ACTIONS),
    ])
    lam = torch.tensor([0.0, 1.0])
    logits, act, hp, aux = model(
        batch, value_logits=True, horizon_lambda=0.3,
        auxiliary_lambda=lam,
    )
    assert logits.shape[0] == 2
    assert act.shape == hp.shape == aux.shape == (2,)
    assert all(torch.isfinite(v).all() for v in (logits, act, hp, aux))
    with pytest.raises(ValueError, match="shape"):
        model(batch, horizon_lambda=torch.tensor([0.4]))


