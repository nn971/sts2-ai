"""v15i cached public action/ownership topology preserves Torch gradients."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from test_batched_relational_v15 import _cohort, _head
from test_pooled_ppo_v15 import combat_episode, frozen_policy

from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.models.relational_objects import (
    prepare_relational_topology,
)
from sts2_ai.training.pooled_ppo_v15 import prepare_pooled_samples
from tools.profile_v15_real_minibatches import (
    build_corpus,
    profile_minibatches,
)


def _inputs():
    _, _, objects, menus = _cohort()
    baselines = [
        PolicyValueEstimate(
            tuple(0.2 * j - 0.05 * i for j in range(len(menu))),
            0.15 + i * 0.1,
        )
        for i, menu in enumerate(menus)
    ]
    return objects, menus, baselines


def test_compiled_relation_topology_matches_dynamic_outputs_and_gradients() -> None:
    torch, head = _head()
    objects, menus, baselines = _inputs()
    topologies = [
        prepare_relational_topology(obj, menu)
        for obj, menu in zip(objects, menus, strict=True)
    ]
    assert [len(t.relations) for t in topologies] == [
        len(obj.tokens) for obj in objects
    ]
    assert [len(t.action_edges) for t in topologies] == list(map(len, menus))
    values0, logits0 = head.forward_batch_prepared(objects, menus, baselines)
    values1, logits1 = head.forward_batch_prepared(
        objects, menus, baselines, topologies=topologies,
    )
    torch.testing.assert_close(values0, values1, atol=2e-5, rtol=2e-5)
    for old, new in zip(logits0, logits1, strict=True):
        torch.testing.assert_close(new, old, atol=2e-5, rtol=2e-5)
    old_loss = values0.square().sum() + sum(
        lg.square().sum() for lg in logits0
    )
    new_loss = values1.square().sum() + sum(
        lg.square().sum() for lg in logits1
    )
    params = [
        head.module.encoder.identities.weight,
        head.module.blocks[0].bias,
        head.module.blocks[1].q.weight,
        head.module.action_roles.weight,
        head.module.cross_k.weight,
        head.module.value_gate,
        head.module.policy_gate,
    ]
    old_grad = torch.autograd.grad(old_loss, params, retain_graph=True)
    new_grad = torch.autograd.grad(new_loss, params)
    for actual, expected in zip(new_grad, old_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_cached_topology_reuses_precomputed_relation_matrix() -> None:
    torch, head = _head()
    obj, menus, baselines = _inputs()
    topologies = [
        prepare_relational_topology(o, a)
        for o, a in zip(obj, menus, strict=True)
    ]
    with patch(
        "sts2_ai.models.relational_objects._relations",
        side_effect=AssertionError("Recomputed immutable ownership relations"),
    ):
        with torch.no_grad():
            values, logits = head.forward_batch_prepared(
                obj, menus, baselines, topologies=topologies,
            )
    assert values.shape == (len(obj),)
    assert [len(x) for x in logits] == [len(a) for a in menus]


def test_reject_wrong_cached_topology_batch_or_menu() -> None:
    _, head = _head()
    obj, menus, baselines = _inputs()
    cache = [
        prepare_relational_topology(o, a)
        for o, a in zip(obj, menus, strict=True)
    ]
    with pytest.raises(ValueError, match="batch size mismatch"):
        head.forward_batch_prepared(obj, menus, baselines, topologies=cache[:1])
    with pytest.raises(ValueError, match="no longer matches"):
        head.forward_batch_prepared(
            obj, [menus[0][:-1], *menus[1:]], baselines,
            topologies=cache,
        )


def test_ppo_samples_cache_topology_and_profile_both_modes() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(47)
    episode = combat_episode()
    base = frozen_policy()
    from sts2_ai.models.object_embeddings import ObjectVocabulary
    from sts2_ai.models.object_instances import extract_combat_objects
    from sts2_ai.models.relational_objects import RelationalObjectTorch
    from test_object_instances_v15 import frame

    vocab = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    head = RelationalObjectTorch(vocab, width=16)
    samples, _ = prepare_pooled_samples(
        [episode], base.combat, head,
        temperature=0.75, gamma=0.995, gae_lambda=0.98,
    )
    assert len(samples) == 2
    assert all(s.relational_topology is not None for s in samples)
    corpus = build_corpus(
        [episode], base, {
            "cardIds": [], "enemyIds": [], "potionIds": [],
            "relicIds": [], "powerIds": [],
        }, revision="test",
    )
    torch.set_num_threads(1)
    profile = profile_minibatches(
        corpus, devices=["cpu"], batch_sizes=[1, 2], width=16,
        warmups=0, trials=1, loss_modes=("batched",),
        topology_modes=("dynamic", "cached"),
    )
    assert {(r["batch_size"], r["topology_mode"]) for r in profile["rows"]} == {
        (1, "dynamic"), (1, "cached"), (2, "dynamic"), (2, "cached"),
    }
    assert profile["topology_preparation_seconds"] >= 0
