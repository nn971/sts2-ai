"""H16 action subset encoding and collision auditor regressions."""
from __future__ import annotations

from itertools import combinations

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.packed_actor_collation import collate_h7_actor_packed
from sts2_ai.models.shared_transformer import (
    H7Config, SharedObjectTransformer, collate_h7,
)
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet
from sts2_ai.training.h16_public_codec import encode_h16_decision


def state() -> dict:
    return {
        "act": 1, "floor": 10, "hp": 46, "max_hp": 70,
        "completed_rooms": [],
        "combat": {
            "turn": 3,
            "hand": [],
            "discard_pile": [
                {"instance_id": i, "card_id": f"demo.card.{i}", "cost": i % 3}
                for i in range(17)
            ],
            "exhaust_pile": [],
            "enemies": [],
            "player_powers": [],
        },
        "deck": [], "relics": [], "potions": [], "map": [],
    }


def selection_actions() -> tuple[LegalAction, ...]:
    # Exactly the 154-menu size observed in the crashed H16 PUCT root.
    return tuple(
        LegalAction(
            f"select-{count}-{','.join(map(str, chosen))}",
            "select_cards",
            '{"CardInstanceIds":[' + ",".join(map(str, chosen)) + "]}",
        )
        for count in (0, 1, 2)
        for chosen in combinations(range(17), count)
    )


def test_h16_154_select_cards_choices_all_have_distinct_compositional_tokens():
    actions = selection_actions()
    assert len(actions) == 154
    encoded = encode_h16_decision(state(), actions)
    assert len(encoded.actions) == 154
    assert len(set(encoded.actions)) == 154
    assert encoded.actions[0].members == ()
    assert len(encoded.actions[1].members) == 1
    assert len(encoded.actions[-1].members) == 2
    assert len(set(action.semantic for action in encoded.actions)) == 3
    # Different candidate-card combinations refer to their public object
    # tokens, independent of the instance-ID numeric representation.
    batch = collate_h7([encoded])
    packed = collate_h7_actor_packed([encoded])
    assert batch.action_members.shape == (1, 154, 2)
    for name in batch.__dataclass_fields__:
        assert torch.equal(getattr(batch, name), getattr(packed, name))
    torch.manual_seed(77)
    torch.set_num_threads(1)
    net = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    with torch.inference_mode():
        logits, _, _ = net(batch)
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, 154)
    # Selected card properties are actually reflected in action predictions.
    assert not torch.allclose(logits[0, 1], logits[0, 2])


def test_h16_rejects_missing_public_subset_candidates():
    bad = LegalAction("select-missing", "select_cards", '{"CardInstanceIds":[999]}')
    with pytest.raises(ValueError, match="candidate missing from public"):
        encode_h16_decision(state(), [bad])


def test_h16_rejects_distinct_actions_that_share_the_same_structural_token():
    a = LegalAction("other-1", "other_choice", '{"ChoiceId":1}')
    b = LegalAction("other-2", "other_choice", '{"ChoiceId":2}')
    with pytest.raises(ValueError, match="action representation collision"):
        encode_h16_decision(state(), [a, b])
