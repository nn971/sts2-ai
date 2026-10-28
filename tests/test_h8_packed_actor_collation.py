"""Contract tests for the packed H8 actor collation fast path."""
from __future__ import annotations

import pytest
import torch

from sts2_ai.models.packed_actor_collation import (
    H7_PACKED_METADATA_FIELDS,
    collate_h7_actor_packed,
    contiguous_actor_metadata,
)
from sts2_ai.models.shared_game_objects import (
    NUMERIC_FIELDS,
    SHARED_OBJECT_SCHEMA,
    ActionToken,
    EncodedDecision,
    ObjectToken,
)
from sts2_ai.models.shared_transformer import (
    H7Config,
    SharedObjectTransformer,
    collate_h7,
    vars_as_dict,
)


def sample_decisions() -> list[EncodedDecision]:
    features = len(NUMERIC_FIELDS)

    def obj(kind: int, identity: int, zone: int, owner: int = -1,
            present: int = 1) -> ObjectToken:
        return ObjectToken(
            kind=kind, identity=identity, zone=zone, owner=owner,
            numeric=tuple((i - 5) / 32.0 for i in range(features)),
            observed=tuple((i + present) % 2 for i in range(features)),
        )

    return [
        EncodedDecision(
            SHARED_OBJECT_SCHEMA, "combat",
            (obj(0, 1, 0), obj(1, 3, 6, 0), obj(2, 20, 1, 1)),
            (ActionToken(8, 20, 2, -1), ActionToken(4, 0, -1, -1)),
        ),
        EncodedDecision(
            SHARED_OBJECT_SCHEMA, "macro",
            (obj(0, 1, 0), obj(4, 100, 7, 0)),
            (ActionToken(4, 0, -1, -1),),
        ),
        EncodedDecision(
            SHARED_OBJECT_SCHEMA, "combat",
            (obj(0, 1, 0), obj(1, 4, 6, 0),
             obj(2, 21, 1, 1), obj(3, 90, 5, 0)),
            (ActionToken(8, 21, 2, 3),
             ActionToken(7, 0, -1, -1),
             ActionToken(15, 0, -1, 3)),
        ),
    ]


def test_packed_actor_tensors_are_bit_exact_across_all_h7_fields() -> None:
    decisions = sample_decisions()
    legacy = collate_h7(decisions)
    packed = collate_h7_actor_packed(decisions)
    assert set(vars_as_dict(legacy)) == set(vars_as_dict(packed))
    for name, expected in vars_as_dict(legacy).items():
        actual = getattr(packed, name)
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        assert actual.device == expected.device, name
        assert torch.equal(expected, actual), name

    # No aliasing back into mutable Python source structures.
    assert packed.kinds.shape == (3, 4)
    assert packed.action_kinds.shape == (3, 3)
    assert packed.owners[1, 3].item() == -1
    assert packed.action_mask[1].tolist() == [True, False, False]


def test_packed_contiguous_materializes_only_metadata_and_preserves_values() -> None:
    decisions = sample_decisions()
    packed = collate_h7_actor_packed(decisions)
    contiguous = collate_h7_actor_packed(
        decisions, contiguous_metadata=True)
    reference = collate_h7(decisions)
    assert set(H7_PACKED_METADATA_FIELDS) == {
        "kinds", "identities", "zones", "owners",
        "action_kinds", "action_semantics", "sources", "targets",
    }
    for name in vars_as_dict(reference):
        a, b, expected = (
            getattr(packed, name), getattr(contiguous, name),
            getattr(reference, name))
        assert torch.equal(a, expected), name
        assert torch.equal(b, expected), name
        if name in H7_PACKED_METADATA_FIELDS:
            assert b.is_contiguous(), name
            # The packed interleaved columns each have stride 4; the
            # contiguous candidate owns a new allocation for each column.
            assert not a.is_contiguous(), name
            assert b.data_ptr() != a.data_ptr(), name
        else:
            # Neither packed numeric features nor masks should be copied.
            assert b.data_ptr() == a.data_ptr() or (
                b.is_contiguous() and a.is_contiguous()), name

    # The helper acts only on metadata, and preserves the original batch.
    again = contiguous_actor_metadata(packed)
    for name in H7_PACKED_METADATA_FIELDS:
        assert torch.equal(getattr(again, name), getattr(contiguous, name))
    assert not packed.kinds.is_contiguous()


def test_packed_actor_transformer_logits_values_match_reference() -> None:
    decisions = sample_decisions()
    torch.manual_seed(27)
    model = SharedObjectTransformer(H7Config(
        width=32, heads=4, layers=1, feedforward=64)).eval()
    with torch.inference_mode():
        legacy_logits, legacy_values = model(collate_h7(decisions))
        packed_logits, packed_values = model(collate_h7_actor_packed(decisions))
        contiguous_logits, contiguous_values = model(
            collate_h7_actor_packed(decisions, contiguous_metadata=True))
    # CPU results must be identical for the same valid tokens and mask.
    for actual in (packed_logits, contiguous_logits):
        torch.testing.assert_close(legacy_logits, actual, atol=0, rtol=0)
    for actual in (packed_values, contiguous_values):
        torch.testing.assert_close(legacy_values, actual, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ("owner", "source", "target", "phase", "empty"))
def test_packed_actor_keeps_checked_pointer_phase_contract(kind: str) -> None:
    frame = sample_decisions()[0]
    if kind == "empty":
        malformed = []
    elif kind == "phase":
        malformed = [EncodedDecision(
            frame.schema, "unsupported", frame.objects, frame.actions)]
    elif kind == "owner":
        malformed = [EncodedDecision(
            frame.schema, frame.phase,
            (frame.objects[0], ObjectToken(
                frame.objects[1].kind, frame.objects[1].identity,
                frame.objects[1].zone, frame.objects[1].numeric,
                frame.objects[1].observed, owner=50)),
            frame.actions)]
    else:
        a = frame.actions[0]
        action = ActionToken(a.kind, a.semantic,
                             77 if kind == "source" else a.source,
                             77 if kind == "target" else a.target)
        malformed = [EncodedDecision(
            frame.schema, frame.phase, frame.objects, (action,))]
    with pytest.raises(ValueError):
        collate_h7_actor_packed(malformed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_actor_matches_gpu_reference_tensor_fields() -> None:
    decisions = sample_decisions()
    old = collate_h7(decisions).to("cuda")
    packed = collate_h7_actor_packed(decisions, device="cuda")
    contiguous = collate_h7_actor_packed(
        decisions, device="cuda", contiguous_metadata=True)
    for name, expected in vars_as_dict(old).items():
        assert torch.equal(expected, getattr(packed, name)), name
        assert torch.equal(expected, getattr(contiguous, name)), name
        if name in H7_PACKED_METADATA_FIELDS:
            assert not getattr(packed, name).is_contiguous()
            assert getattr(contiguous, name).is_contiguous()

    torch.manual_seed(27)
    model = SharedObjectTransformer(H7Config(
        width=32, heads=4, layers=1, feedforward=64)).eval().cuda()
    with torch.inference_mode():
        l_logit, l_value = model(old)
        p_logit, p_value = model(packed)
        c_logit, c_value = model(contiguous)
    for output, reference in (
        (p_logit, l_logit), (c_logit, l_logit),
        (p_value, l_value), (c_value, l_value),
    ):
        torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)
