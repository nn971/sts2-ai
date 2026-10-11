"""v15j weight-independent compiled public features keep full gradient parity."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from test_batched_relational_v15 import _cohort, _head
from test_pooled_ppo_v15 import combat_episode, frozen_policy

from sts2_ai.models.object_embeddings import ObjectVocabulary, TypedObjectEncoder
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.models.relational_objects import prepare_relational_topology
from sts2_ai.training.pooled_ppo_v15 import prepare_pooled_samples
from tools.profile_v15_real_minibatches import build_corpus, profile_minibatches


def test_compiled_public_embedding_matches_dynamic_and_gradients() -> None:
    torch, head = _head()
    _, _, objects, _ = _cohort()
    encoder = head.embedding
    prepared = [encoder.prepare(obj) for obj in objects]
    reference = encoder.encode_many(objects)
    compiled = encoder.encode_many_compiled(prepared)
    assert sum(row.count for row in prepared) == sum(len(obj.tokens) for obj in objects)
    for x, y in zip(reference, compiled, strict=True):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
    loss0 = sum((x ** 2).sum() for x in reference)
    loss1 = sum((x ** 2).sum() for x in compiled)
    params = [
        head.module.encoder.kinds.weight,
        head.module.encoder.identities.weight,
        head.module.encoder.categories.weight,
        head.module.encoder.numeric_fields.weight,
        head.module.encoder.numeric_layers[
            encoder._id("kinds", "card")
        ][0].weight,
        head.module.encoder.norm.weight,
    ]
    old = torch.autograd.grad(loss0, params, retain_graph=True, allow_unused=True)
    new = torch.autograd.grad(loss1, params, allow_unused=True)
    for a, b in zip(old, new, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)


def test_compiled_full_relational_model_matches_dynamic_and_gradient() -> None:
    torch, head = _head()
    _, _, objects, actions = _cohort()
    baselines = [
        PolicyValueEstimate(
            tuple(j * 0.04 for j in range(len(menu))), 0.15,
        )
        for menu in actions
    ]
    topologies = [
        prepare_relational_topology(o, menu)
        for o, menu in zip(objects, actions, strict=True)
    ]
    features = [head.embedding.prepare(o) for o in objects]
    v0, log0 = head.forward_batch_prepared(
        objects, actions, baselines, topologies=topologies,
    )
    v1, log1 = head.forward_batch_prepared(
        objects, actions, baselines,
        topologies=topologies, features=features,
    )
    torch.testing.assert_close(v0, v1, atol=2e-5, rtol=2e-5)
    for a, b in zip(log0, log1, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    params = [
        head.module.encoder.identities.weight,
        head.module.encoder.numeric_fields.weight,
        head.module.blocks[0].q.weight,
        head.module.action_roles.weight,
        head.module.policy_gate,
    ]
    reference_loss = v0.square().sum() + sum(x.square().sum() for x in log0)
    compiled_loss = v1.square().sum() + sum(x.square().sum() for x in log1)
    reference_grad = torch.autograd.grad(reference_loss, params, retain_graph=True)
    compiled_grad = torch.autograd.grad(compiled_loss, params)
    for old, new in zip(reference_grad, compiled_grad, strict=True):
        torch.testing.assert_close(old, new, atol=2e-4, rtol=2e-4)


def test_cache_reuses_labels_but_not_old_embedding_weights() -> None:
    torch, head = _head()
    _, _, objects, _ = _cohort()
    prepared = [head.embedding.prepare(obj) for obj in objects]
    with patch.object(
        head.embedding, "_id", side_effect=AssertionError("Recomputed public labels"),
    ):
        with torch.no_grad():
            before = head.embedding.encode_many_compiled(prepared)[0].clone()
            head.module.encoder.identities.weight.add_(0.1 * torch.randn_like(
                head.module.encoder.identities.weight,
            ))
            after = head.embedding.encode_many_compiled(prepared)[0]
    assert not torch.allclose(before, after)


def test_empty_sample_encoder_and_malformed_feature_batch() -> None:
    torch = pytest.importorskip("torch")
    vocab = ObjectVocabulary.from_object_sets([])
    enc = TypedObjectEncoder(vocab, width=16)
    empty = enc.prepare(extract_combat_objects({
        "combat": {
            "turn": 1, "energy": 2, "player_block": 0,
            "hand": [], "draw_pile": [], "draw_pile_count": 0,
            "discard_pile": [], "exhaust_pile": [], "play_pile": [],
            "enemies": [], "player_powers": [],
        },
        "player": {"hp": 50, "max_hp": 60},
        "relics": [], "potions": [],
    }))
    assert empty.count > 0
    x = enc.encode_many_compiled([empty])
    assert x[0].shape[0] == empty.count
    torch, head = _head()
    _, _, objects, actions = _cohort()
    values = [PolicyValueEstimate(tuple(0.0 for _ in menu), 0.2)
              for menu in actions]
    with pytest.raises(ValueError, match="feature batch size"):
        head.forward_batch_prepared(
            objects, actions, values, features=[head.embedding.prepare(objects[0])],
        )
    with pytest.raises(ValueError, match="no longer match"):
        head.forward_batch_prepared(
            objects, actions, values,
            features=[head.embedding.prepare(objects[0])] * len(objects),
        )


def test_ppo_replay_compiles_features_once_and_profiles_both_modes() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    episode = combat_episode()
    base = frozen_policy()
    from test_object_instances_v15 import frame
    from sts2_ai.models.relational_objects import RelationalObjectTorch

    objects = extract_combat_objects(frame())
    vocab = ObjectVocabulary.from_object_sets([objects])
    head = RelationalObjectTorch(vocab, width=16)
    samples, _ = prepare_pooled_samples(
        [episode], base.combat, head,
        temperature=0.75, gamma=0.995, gae_lambda=0.98,
    )
    assert all(sample.compiled_features is not None for sample in samples)
    corpus = build_corpus(
        [episode], base, {
            "cardIds": [], "enemyIds": [], "potionIds": [],
            "relicIds": [], "powerIds": [],
        }, revision="synthetic-public",
    )
    torch.set_num_threads(1)
    profile = profile_minibatches(
        corpus, devices=["cpu"], batch_sizes=[1, 2], width=16,
        warmups=0, trials=1, loss_modes=("batched",),
        topology_modes=("cached",), feature_modes=("dynamic", "compiled"),
    )
    assert {(row["batch_size"], row["feature_mode"]) for row in profile["rows"]} == {
        (1, "dynamic"), (1, "compiled"), (2, "dynamic"), (2, "compiled"),
    }
    assert profile["feature_preparation_seconds"] >= 0
