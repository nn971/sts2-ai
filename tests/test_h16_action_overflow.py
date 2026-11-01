"""H16 regression: all legal choices survive large combat/macro menus."""
from __future__ import annotations

from itertools import combinations

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.packed_actor_collation import collate_h7_actor_packed
from sts2_ai.models.shared_game_objects import MAX_ACTIONS, encode_public_decision
from sts2_ai.models.shared_transformer import (
    H7Config, SharedObjectTransformer, collate_h7,
)
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet
from sts2_ai.training.h16_public_codec import encode_h16_decision


def state(*, combat: bool) -> dict:
    return {
        "act": 1, "floor": 7, "hp": 42, "max_hp": 70,
        "completed_rooms": [],
        "combat": {
            "turn": 1, "enemies": [], "hand": [],
            "discard_pile": [
                {"instance_id": j, "card_id": f"demo.card.{j}"}
                for j in range(23)
            ],
            "exhaust_pile": [],
        } if combat else None,
        "deck": [], "relics": [], "potions": [], "map": [],
    }


@pytest.mark.parametrize("size", [MAX_ACTIONS + 1, 257])
@pytest.mark.parametrize("combat", [False, True])
def test_large_menu_preserves_all_choices_and_logits(size: int, combat: bool):
    public = state(combat=combat)
    if combat:
        subsets = (
            subset for count in (0, 1, 2)
            for subset in combinations(range(23), count)
        )
        actions = tuple(
            LegalAction(f"select-{i}", "select_cards",
                        '{"CardInstanceIds":[' + ",".join(map(str, selected)) + "]}")
            for i, selected in zip(range(size), subsets)
        )
    else:
        actions = tuple(
            LegalAction(f"menu-{i}", "event_choice", f'{{"ChoiceId":{i}}}')
            for i in range(size)
        )
    assert len(actions) == size
    with pytest.raises(ValueError, match="Legal action cap exceeded"):
        encode_public_decision(public, actions)

    encoded = encode_h16_decision(public, actions)
    assert len(encoded.actions) == size
    assert len(set(encoded.actions)) == size
    control = encode_h16_decision(public, actions[:3])
    full = collate_h7([encoded, control])
    packed = collate_h7_actor_packed([encoded, control])
    assert full.action_mask.shape == (2, size)
    assert bool(full.action_mask[0].all())
    assert int(full.action_mask[1].sum()) == 3
    for name in full.__dataclass_fields__:
        assert torch.equal(getattr(full, name), getattr(packed, name))

    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    with torch.inference_mode():
        scores, act, hp = model(full, horizon_lambda=0.3)
    assert scores.shape == (2, size)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1, :3]).all()
    assert torch.isfinite(act).all()
    assert torch.isfinite(hp).all()
