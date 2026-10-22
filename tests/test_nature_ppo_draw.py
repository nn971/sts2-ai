"""On-policy Nature PPO tests: genuine behavior q, critic, Adam and replay."""
from __future__ import annotations

from dataclasses import replace

import pytest
from test_nature_contextual_draw import event, visible
from test_nature_draw_only_law import pending

from sts2_ai.nature.contextual_draw import restore_nature
from sts2_ai.nature.draw_only_p0 import DrawOnlyRouter, SeededLawSampler
from sts2_ai.nature.ppo_draw import PPO_DRAW_SCHEMA, PpoDrawNature
from sts2_ai.nature.two_site_opponent import PilotEpisode

torch = pytest.importorskip("torch")


def make_batch(nature):
    choices = []
    for i in range(8):
        state = visible()
        state["hp"] = 30 + i
        state["combat"]["turn"] = 1 + i
        _, choice = nature.propose(
            event(), context_key="draw",
            candidate_features=("strike", "defend"), episode_key=f"epi:{i}",
            public_observation=state,
        )
        choices.append(choice)
    # A full combat normally contains many draws. Use at least two for GAE.
    return tuple(PilotEpisode(
        player_return=(0.3 if i % 2 == 0 else 0.9),
        choices=(choices[i], choices[(i + 1) % 8]),
    ) for i in range(8))


def test_ppo_uses_same_fair_router_and_enemy_intents_stay_reference():
    nature = PpoDrawNature(seed=11, learning_rate=0.001)
    router = DrawOnlyRouter(nature, SeededLawSampler(19))
    result = router.choose(
        pending(), fingerprint="parent", episode_key="world",
        ordinal=0, public_observation=visible(),
    )
    choice = result.nature_choice
    assert choice is not None
    assert choice.behavior_value is not None
    assert -1 <= choice.behavior_value <= 0
    assert choice.state_features
    assert result.log_q == pytest.approx(
        torch.log(torch.tensor(choice.probabilities[choice.selected_index])).item()
    )
    assert nature.policy_id == PPO_DRAW_SCHEMA
    intent = router.choose(
        pending("enemy_intent"), fingerprint="parent",
        episode_key="world", ordinal=1,
    )
    assert intent.controller == "engine-law-reference"
    assert intent.nature_choice is None


def test_ppo_updates_actor_and_critic_multiple_times_and_roundtrips_moments():
    nature = PpoDrawNature(
        seed=17, learning_rate=0.002, ppo_epochs=3,
        ppo_batch_size=3, ppo_sample_limit=16,
    )
    before = nature.dump()
    training = make_batch(nature)
    report = nature.update(training)
    assert report["nature_optimizer_steps"] >= 6
    assert report["nature_used_decisions"] == 16
    assert report["nature_ppo_value_mse"] >= 0
    assert report["nature_optimizer_steps_total"] >= 6
    assert report["nature_gradient_norm"] > 0
    assert nature.value_weight != before["value_weight"]
    assert nature.state_weight != before["state_weight"]
    assert nature.dump()["optimizer_moments"]
    assert any(state["step"] > 0 for state in nature.optimizer_moments.values())

    restored = restore_nature(nature.dump())
    assert type(restored) is PpoDrawNature
    assert restored.dump() == nature.dump()
    opts = dict(
        event=event(), context_key="draw",
        candidate_features=("strike", "defend"), episode_key="resume",
        public_observation=visible(),
    )
    assert restored.propose(**opts) == nature.propose(**opts)
    second_batch = make_batch(restored)
    report2 = restored.update(second_batch)
    assert report2["nature_optimizer_steps_total"] > report["nature_optimizer_steps_total"]
    assert restored.optimizer_steps == nature.optimizer_steps + int(
        report2["nature_optimizer_steps"]
    )


def test_ppo_rejects_stale_policy_and_critic_and_probability_tampering():
    nature = PpoDrawNature(seed=24, ppo_epochs=1, ppo_batch_size=8)
    original = make_batch(nature)
    corrupted = replace(
        original[0].choices[0],
        behavior_value=original[0].choices[0].behavior_value + 0.1,
    )
    with pytest.raises(ValueError, match="critic provenance"):
        nature.update((replace(original[0], choices=(corrupted,)), *original[1:]))
    corrupted = replace(original[0].choices[0], probabilities=(0.8, 0.2))
    with pytest.raises(ValueError, match="behavior q"):
        nature.update((replace(original[0], choices=(corrupted,)), *original[1:]))
    nature.update(original)
    with pytest.raises(ValueError, match="Stale"):
        nature.update(original)


def test_ppo_restoration_keeps_full_support_and_context_dependence():
    nature = PpoDrawNature(seed=5, mix=0.75)
    args = dict(
        event=event(), context_key="draw",
        candidate_features=("strike", "defend"),
        episode_key="fixed", public_observation=visible(),
    )
    original = nature.propose(**args)
    assert all(p >= 0.125 for p in original[1].probabilities)
    changed = visible()
    changed["combat"]["energy"] = 0
    assert original[1].state_features != nature.propose(
        **{**args, "public_observation": changed}
    )[1].state_features
    restored = restore_nature(nature.dump())
    assert restored.propose(**args) == original


def test_ppo_portable_actor_and_critic_match_torch_optimizer():
    nature = PpoDrawNature(seed=57)
    opts = dict(
        event=event(), context_key="draw", episode_key="parity",
        candidate_features=("strike", "defend"),
        public_observation=visible(),
    )
    _, choice = nature.propose(**opts)
    model = nature._parameters(torch)
    value, probs = nature._forward_torch(
        torch, model, choice.state_features, choice.features,
        choice.baseline, nature.mix,
    )
    assert value.item() == pytest.approx(choice.behavior_value, abs=1e-12)
    assert probs.detach().tolist() == pytest.approx(choice.probabilities, abs=1e-12)


def test_ppo_learns_adversarial_draw_in_controlled_contextual_bandit():
    # Validates the direction of the *actual clipped optimizer* with a
    # known terminal outcome. This is a synthetic learning sanity check,
    # not evidence of success against the frozen v8 player.
    nature = PpoDrawNature(
        seed=63, learning_rate=0.003, ppo_epochs=4,
        ppo_batch_size=32, ppo_sample_limit=128,
    )
    common = dict(
        event=event(), context_key="draw", candidate_features=("strike", "defend"),
        public_observation=visible(),
    )
    _, initial = nature.propose(**{**common, "episode_key": "before"})
    initial_bad = initial.probabilities[0]
    for round_index in range(5):
        episodes = []
        for i in range(128):
            _, choice = nature.propose(
                **{**common, "episode_key": f"bandit:{round_index}:{i}"}
            )
            # Strike (index 0) has zero player return, so Nature should
            # learn to draw it more frequently than Defend (index 1).
            episodes.append(PilotEpisode(
                player_return=float(choice.selected_index == 1),
                choices=(choice,),
            ))
        nature.update(tuple(episodes))
    _, final = nature.propose(**{**common, "episode_key": "after"})
    assert final.probabilities[0] > initial_bad + 0.08
