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



def test_stable_id_memoization_preserves_public_identity_and_collision_guards():
    # Repeated concept strings are common across consecutive public decisions.
    # Keep a bounded process-local SHA cache without disabling the per-frame
    # full-hash and effective-embedding collision checks.
    name = "h21.memoized.identity.unique.test.20261104"
    stable_id.cache_clear()
    before = stable_id.cache_info()
    first = stable_id(name)
    second = stable_id(name)
    after = stable_id.cache_info()
    assert first == second
    assert after.misses == before.misses + 1
    assert after.hits == before.hits + 1
    assert H21StableIdentityHasher()(name) == first
    assert H21StableIdentityHasher()(name) == second



def test_sparse_h21_numeric_projection_matches_original_all_field_shapes():
    import math

    from sts2_ai.models.shared_game_objects import (
        NUMERIC_FIELDS, _numbers, _numbers_h21,
    )

    examples = [
        {},
        {"identity": "non-numeric", "state": {}, "enabled": True},
        {"stacks": 0},
        {"floor": 16},
        {"energy": 3, "turn": 12},
        {"hp": 56, "max_hp": 80, "block": 0},
        {"gold": 0, "cost": float("nan")},
        {"cost": float("inf")},
        {"cost": -5.0},
        {"stacks": False},
        {"stacks": "two"},
        {"count": 1000000000},
        {"hp": None, "floor": 2},
    ]
    for field in NUMERIC_FIELDS:
        examples.extend([
            {field: 0},
            {field: -13.5},
            {field: True},
            {field: math.nan},
            {field: math.inf},
        ])
    for record in examples:
        assert _numbers_h21(record) == _numbers(record), record



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
