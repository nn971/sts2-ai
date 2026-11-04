"""H21 eliminates known 4096-bucket aliases with stable multi-hash features."""
from __future__ import annotations

import pytest

from sts2_ai.models.shared_game_objects import IDENTITY_BUCKETS, _bucket
from sts2_ai.training.h21_stable_identity import (
    H21_EMBEDDING_MODULUS,
    H21StableIdentityHasher,
    stable_id,
)

CARD_A = "proto.silent.piercing_wail"
CARD_B = "proto.silent.dagger_spray"


def test_actual_silent_card_collision_disappears_without_context_rehashing():
    # This pair was found by the eight-seed native live-observation probe.
    assert _bucket(CARD_A, IDENTITY_BUCKETS) == _bucket(CARD_B, IDENTITY_BUCKETS)
    hasher = H21StableIdentityHasher()
    first, second = hasher(CARD_A), hasher(CARD_B)
    assert first != second
    assert first % H21_EMBEDDING_MODULUS != second % H21_EMBEDDING_MODULUS
    assert (first, second) == (
        H21StableIdentityHasher()(CARD_A),
        H21StableIdentityHasher()(CARD_B),
    )
    assert hasher(CARD_A) == first


def test_effective_neural_alias_still_fails_even_with_different_full_ids(monkeypatch):
    import sts2_ai.training.h21_stable_identity as module

    first = 7
    second = H21_EMBEDDING_MODULUS + first
    original = module.stable_id

    def synthetic(label):
        if label == "synthetic.one":
            return first
        if label == "synthetic.two":
            return second
        return original(label)

    monkeypatch.setattr(module, "stable_id", synthetic)
    hasher = H21StableIdentityHasher()
    hasher("synthetic.one")
    with pytest.raises(ValueError, match="stable neural identity collision"):
        hasher("synthetic.two")


def test_unchanged_public_names_retain_their_full_hash_across_instances():
    assert 0 < stable_id(CARD_A) < (1 << 63)
    assert stable_id(CARD_A) == stable_id(CARD_A)
    assert stable_id(CARD_A) != stable_id(CARD_B)
    assert stable_id("") == 0


def test_multi_hash_neural_embedding_separates_colliding_legacy_cards():
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_macro_transformer_v2 import H21StableEmbedding

    torch.manual_seed(42)
    embed = H21StableEmbedding(width=16)
    values = torch.tensor([[stable_id(CARD_A), stable_id(CARD_B)]])
    outputs = embed(values)
    assert outputs.shape == (1, 2, 16)
    assert not torch.allclose(outputs[0, 0], outputs[0, 1])
    outputs.sum().backward()
    assert all(t.weight.grad is not None for t in embed.tables)
