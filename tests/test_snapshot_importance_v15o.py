"""v15o: finite-corpus importance math and weighted combat PPO regression tests."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
from test_combat_snapshots_v14 import ToyBackend
from test_pooled_ppo_v15 import combat_episode, frozen_policy
from test_snapshot_curriculum_v15m import authentic_recipe

from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.models.pooled_objects import PooledObjectTorch
from sts2_ai.training.pooled_ppo_v15 import (
    batched_ppo_loss,
    prepare_pooled_samples,
)
from sts2_ai.training.snapshot_importance import (
    build_importance_distribution,
    importance_diagnostics,
    snapshot_identity,
)
from test_object_instances_v15 import frame


def corpus():
    backend = ToyBackend()
    template = authentic_recipe(backend)
    weak = [
        replace(template, seed=f"weak-{i}", exact_hash=f"weak-{i}")
        for i in range(12)
    ]
    elite = replace(
        template, seed="elite", exact_hash="elite", tier="elite",
        floor=8, progress="middle", hp=8,
    )
    boss = replace(
        template, seed="boss", exact_hash="boss", tier="boss",
        floor=16, progress="late", hp=2,
    )
    assert not backend.states
    return weak + [elite, boss]


def test_target_proposal_reweighting_is_exact_for_finite_corpus() -> None:
    recipes = corpus()
    d = build_importance_distribution(
        recipes, target_balance=0.5, priority_mix=0.7,
    )
    assert d == build_importance_distribution(
        list(reversed(recipes)), target_balance=0.5, priority_mix=0.7,
    )
    assert math.fsum(d.target_probabilities) == pytest.approx(1.0)
    assert math.fsum(d.proposal_probabilities) == pytest.approx(1.0)
    assert all(q >= 0.3 * p - 1e-14 for p, q in zip(
        d.target_probabilities, d.proposal_probabilities, strict=True,
    ))
    # For any fixed function f, E_q[(p/q) f] = E_p[f].
    f = [i ** 2 for i in range(len(d.recipes))]
    corrected = math.fsum(
        q * (p / q) * x for p, q, x in zip(
            d.target_probabilities, d.proposal_probabilities, f, strict=True,
        )
    )
    target = math.fsum(p * x for p, x in zip(d.target_probabilities, f, strict=True))
    assert corrected == pytest.approx(target)
    assert d.target_probabilities[-1] > d.target_probabilities[0]
    assert d.proposal_probabilities[-1] > d.target_probabilities[-1]


def test_draws_are_reproducible_positive_and_ess_bounded() -> None:
    d = build_importance_distribution(corpus())
    draws = d.draw(count=200, seed=13)
    assert draws == d.draw(count=200, seed=13)
    assert draws != d.draw(count=200, seed=14)
    assert all(0 < x.weight <= 1 / 0.3 + 1e-12 for x in draws)
    stats = importance_diagnostics(draws)
    assert 0 < stats["effective_sample_size"] <= 200 + 1e-10
    assert 0 < stats["effective_sample_fraction"] <= 1 + 1e-10
    with pytest.raises(ValueError, match="count"):
        d.draw(count=0, seed=13)
    with pytest.raises(ValueError, match="empty"):
        importance_diagnostics([])


def test_priority_override_is_separate_from_target() -> None:
    recipes = corpus()
    plain = build_importance_distribution(recipes)
    key = snapshot_identity(next(r for r in recipes if r.tier == "boss"))
    changed = build_importance_distribution(
        recipes, priority_scores={key: 99.0},
    )
    assert changed.target_probabilities == plain.target_probabilities
    assert changed.proposal_probabilities != plain.proposal_probabilities
    boss = next(i for i, r in enumerate(changed.recipes) if r.tier == "boss")
    assert changed.proposal_probabilities[boss] > plain.proposal_probabilities[boss]
    with pytest.raises(ValueError, match="positive"):
        build_importance_distribution(recipes, priority_scores={key: 0})
    with pytest.raises(ValueError, match="priority_mix"):
        build_importance_distribution(recipes, priority_mix=1.0)
    with pytest.raises(ValueError, match="target_balance"):
        build_importance_distribution(recipes, target_balance=-0.1)
    with pytest.raises(ValueError, match="Duplicated"):
        build_importance_distribution(recipes + [recipes[-1]])


def test_weighted_batched_ppo_matches_manual_loss_and_gradient() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.3, 0.1], requires_grad=True)
    logits = [torch.zeros(2, requires_grad=True) for _ in range(2)]
    result = batched_ppo_loss(
        torch, values, logits, chosen_indices=[0, 0],
        old_logprobs=[-math.log(2)] * 2,
        advantages=[1.0, 2.0], targets=[0.2, 0.8],
        temperature=1.0, sample_weights=[1.0, 3.0],
        entropy_weight=0.0, value_weight=0.5,
    )
    combined, policy, critic, entropy = result
    assert float(policy) == pytest.approx(-1.75)
    assert float(critic) == pytest.approx((0.01 + 3 * 0.49) / 4)
    assert float(entropy) == pytest.approx(math.log(2))
    assert float(combined) == pytest.approx(
        -1.75 + 0.5 * (0.01 + 3 * 0.49) / 4,
    )
    combined.backward()
    assert values.grad is not None
    assert values.grad.tolist() == pytest.approx([0.025, -0.525], abs=1e-6)
    unweighted = batched_ppo_loss(
        torch, values.detach(), [x.detach() for x in logits],
        chosen_indices=[0, 0], old_logprobs=[-math.log(2)] * 2,
        advantages=[1.0, 2.0], targets=[0.2, 0.8],
        temperature=1.0, entropy_weight=0.0, value_weight=0.5,
    )
    assert float(unweighted[1]) == pytest.approx(-1.5)
    with pytest.raises(ValueError, match="finite and positive"):
        batched_ppo_loss(
            torch, values.detach(), [x.detach() for x in logits],
            chosen_indices=[0, 0], old_logprobs=[0.0, 0.0],
            advantages=[1, 1], targets=[1, 1], temperature=1.0,
            sample_weights=[1, 0],
        )


def test_importance_weight_is_normalized_per_snapshot_not_per_decision() -> None:
    pytest.importorskip("torch")
    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    head = PooledObjectTorch(vocabulary, width=16)
    two = combat_episode()
    one = replace(
        two,
        decisions=two.decisions[:1],
        combat_outcomes=(
            replace(two.combat_outcomes[0], end_decision=1),
        ),
    )
    samples, _ = prepare_pooled_samples(
        [two, one], base.combat, head, temperature=0.85,
        gamma=0.995, gae_lambda=0.98, episode_weights=[3, 1],
    )
    assert [s.importance_weight for s in samples] == [1.5, 1.5, 1]
    assert sum(s.importance_weight for s in samples[:2]) == 3
    assert sum(s.importance_weight for s in samples[2:]) == 1
    default, _ = prepare_pooled_samples(
        [two, one], base.combat, head, temperature=0.85,
        gamma=0.995, gae_lambda=0.98,
    )
    assert all(s.importance_weight == 1.0 for s in default)
    with pytest.raises(ValueError, match="one resolved combat"):
        prepare_pooled_samples(
            [combat_episode(completed=False, resolved=True)],
            base.combat, head, temperature=0.85, gamma=0.995,
            gae_lambda=0.98, episode_weights=[1],
        )
