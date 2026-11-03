"""H19 resource model can learn owners and cannot load legacy checkpoint."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sts2_ai.emulator.protocol import LegalAction  # noqa: E402
from sts2_ai.models.shared_transformer import (  # noqa: E402
    H7Config, SharedObjectTransformer,
)
from sts2_ai.models.shared_owner_transformer_v2 import (  # noqa: E402
    H19OwnerTransformer, collate_h19,
)
from sts2_ai.training.h19_resource_codec import (  # noqa: E402
    EMULATOR_PUBLIC_SCHEMA_V1, encode_h19_resource_decision,
)


def decision():
    state = {
        "hp": 35, "max_hp": 70, "gold": 99,
        "potion_slot_capacity": 2,
        "potion_slots_occupied": 0, "potion_slots_empty": 2,
        "deck": [{"card_id": "demo.attack", "instance_id": 71,
                  "upgrade_level": 0, "state": {"counter": 2}}],
        "relics": [{"relic_id": "demo.relic", "state": {"used": False}}],
        "potions": [], "map": [],
    }
    return encode_h19_resource_decision(
        state, [LegalAction("a", "skip", "{}")],
        public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1,
    )


def test_h19_batched_model_forward_and_checkpoint_separation(tmp_path):
    encoded = decision()
    batch = collate_h19([encoded])
    torch.manual_seed(29)
    model = H19OwnerTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64))
    logits, value = model(batch)
    assert logits.shape == (1, 1)
    assert value.shape == (1,)
    assert torch.isfinite(logits).all()
    (-torch.log_softmax(logits, dim=-1).mean() + value.square().mean()).backward()
    assert model.owner_projection.weight.grad.abs().sum() > 0
    h19_path = tmp_path / "owner.pt"
    model.save_checkpoint(h19_path)
    restored = H19OwnerTransformer.load_checkpoint(h19_path)
    with torch.no_grad():
        ref = model.eval()(batch)
        new = restored.eval()(batch)
    torch.testing.assert_close(ref[0], new[0])
    torch.testing.assert_close(ref[1], new[1])
    with pytest.raises(ValueError, match="Unsupported H7"):
        SharedObjectTransformer.load_checkpoint(h19_path)

    legacy_path = tmp_path / "legacy.pt"
    SharedObjectTransformer(H7Config(
        width=32, layers=1, heads=4, feedforward=64)).save_checkpoint(legacy_path)
    with pytest.raises(ValueError, match="Unsupported H19"):
        H19OwnerTransformer.load_checkpoint(legacy_path)
