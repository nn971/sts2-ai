"""v15e: vectorized representation, catalog completeness and OOV semantics."""
from __future__ import annotations

import copy
import math

import pytest
from test_object_instances_v15 import frame

from sts2_ai.models.object_embeddings import ObjectVocabulary, TypedObjectEncoder
from sts2_ai.models.object_instances import (
    OBJECT_SCHEMA,
    ObjectSet,
    extract_combat_objects,
)


def _scalar_reference(encoder: TypedObjectEncoder, objects: ObjectSet):
    """Independent reference implementation of the pre-v15e token loop."""
    torch = pytest.importorskip("torch")
    device = encoder.module.kinds.weight.device
    rows = []
    for token in objects.tokens:
        kind = encoder._id("kinds", token.kind)

        def index(value: int):
            return torch.tensor(value, dtype=torch.long, device=device)

        x = (
            encoder.module.kinds(index(kind))
            + encoder.module.identities(
                index(encoder._id("identities", f"{token.kind}:{token.type_id}"))
            )
            + encoder.module.roles(
                index(encoder._id("roles", f"{token.kind}:{token.role}"))
            )
        )
        if token.categorical:
            c = [
                encoder._id("categories", f"{token.kind}:{label}")
                for label in token.categorical
            ]
            x = x + encoder.module.categories(
                torch.tensor(c, dtype=torch.long, device=device)
            ).mean(dim=0)
        if token.numeric:
            ids = [
                encoder._id("numeric_fields", f"{token.kind}:{field}")
                for field, _ in token.numeric
            ]
            vals = [
                math.copysign(math.log1p(abs(value)), value) / 4.0
                for _, value in token.numeric
            ]
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
            vals_tensor = torch.tensor(
                vals, dtype=encoder.module.kinds.weight.dtype, device=device,
            ).unsqueeze(-1)
            fields = encoder.module.numeric_fields(ids_tensor)
            x = x + encoder.module.numeric_layers[kind](
                torch.cat((fields, vals_tensor), dim=-1)
            ).mean(dim=0)
        rows.append(encoder.module.norm(x))
    if not rows:
        return encoder.module.kinds.weight.new_zeros((0, encoder.width))
    return torch.stack(rows, dim=0)


def _cases() -> tuple[ObjectSet, ...]:
    first = frame()
    second = copy.deepcopy(first)
    second["combat"]["hand"][0]["state"] = {
        "counter": -999.0,
        "status": "new-tag",
        "flags": [True, False],
        "nested": {"energy_cost": 0.5},
    }
    second["combat"]["enemies"][0]["powers"] = [
        {"power_id": "proto.power.strangle", "stacks": 1},
        {"power_id": "proto.power.strangle", "stacks": 9},
    ]
    second["combat"]["hand"].reverse()
    second["combat"]["draw_pile"].reverse()
    return (
        extract_combat_objects(first),
        extract_combat_objects(second),
        ObjectSet(OBJECT_SCHEMA, ()),
    )


def test_vectorized_embedding_matches_scalar_tokens_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(163)
    cases = _cases()
    vocab = ObjectVocabulary.from_object_sets(cases[:1])
    encoder = TypedObjectEncoder(vocab, width=16)
    reference = tuple(_scalar_reference(encoder, obj) for obj in cases)
    batched = encoder.encode_many(cases)
    assert len(batched) == len(cases)
    for before, after in zip(reference, batched, strict=True):
        torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)
    for obj, expected in zip(cases, reference, strict=True):
        torch.testing.assert_close(
            encoder.encode(obj), expected, atol=2e-5, rtol=2e-5,
        )
    reference_loss = sum(x.square().sum() for x in reference)
    vector_loss = sum(x.square().sum() for x in batched)
    parameters = [
        encoder.module.kinds.weight,
        encoder.module.identities.weight,
        encoder.module.categories.weight,
        encoder.module.numeric_fields.weight,
        encoder.module.numeric_layers[
            encoder._id("kinds", "card")
        ][0].weight,
    ]
    ref_grad = torch.autograd.grad(reference_loss, parameters, retain_graph=True)
    fast_grad = torch.autograd.grad(vector_loss, parameters)
    for left, right in zip(ref_grad, fast_grad, strict=True):
        torch.testing.assert_close(left, right, atol=3e-4, rtol=2e-4)


def test_vectorized_empty_batch_and_unknown_tokens() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(31)
    a, b, empty = _cases()
    vocab = ObjectVocabulary.from_object_sets((a,))
    encoder = TypedObjectEncoder(vocab, width=32)
    encoded = encoder.encode_many((empty, a, empty, b, empty))
    assert [x.shape[0] for x in encoded] == [0, len(a.tokens), 0, len(b.tokens), 0]
    assert all(torch.isfinite(x).all().item() for x in encoded)
    torch.testing.assert_close(
        encoded[1], encoder.encode(a), atol=2e-5, rtol=2e-5,
    )


def test_catalog_vocabulary_distinguishes_unseen_implemented_objects() -> None:
    observation = extract_combat_objects(frame())
    manifest = {
        "cardIds": ["proto.test.starter", "proto.test.rare_card"],
        "enemyIds": ["proto.enemy.alpha", "proto.enemy.beta"],
        "relicIds": ["proto.relic.one", "proto.relic.two"],
        "potionIds": ["proto.potion.alpha", "proto.potion.beta"],
        "powerIds": ["proto.power.strangle", "proto.power.strength"],
    }
    catalog = ObjectVocabulary.from_object_sets((observation,), manifest=manifest)
    sampled = ObjectVocabulary.from_object_sets((observation,))
    assert "card:proto.test.rare_card" in catalog.identities
    assert "enemy:proto.enemy.beta" in catalog.identities
    assert "relic:proto.relic.two" in catalog.identities
    assert "potion:proto.potion.beta" in catalog.identities
    assert "power:proto.power.strangle" in catalog.identities
    assert "card:proto.test.rare_card" not in sampled.identities
    assert catalog.to_dict() == ObjectVocabulary.from_dict(catalog.to_dict()).to_dict()
    assert catalog.roles[0] == "<UNK>"
    assert "status:active" in catalog.roles
    for field in manifest:
        bad = copy.deepcopy(manifest)
        bad[field] = ["duplicated", "duplicated"]
        with pytest.raises(ValueError, match="Invalid emulator manifest"):
            ObjectVocabulary.from_object_sets((), manifest=bad)


def test_oov_report_tracks_identity_and_mutable_attribute_separately() -> None:
    old, new, _ = _cases()
    vocab = ObjectVocabulary.from_object_sets((old,))
    report = vocab.oov_report((new,))
    assert report["identities"]["unknown"] >= 0
    assert report["categories"]["unknown"] > 0
    assert any("new-tag" in s for s in report["categories"]["examples"])
    assert all(
        not any("instance_id=" in value for value in entry["examples"])
        for entry in report.values()
    )
    assert vocab.oov_report((old,))["categories"]["unknown"] == 0
    for entry in report.values():
        assert entry["total"] >= entry["unknown"] >= 0
        assert 0 <= entry["unknown_fraction"] <= 1
