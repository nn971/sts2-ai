"""H16 action subset encoding and collision auditor regressions."""
from __future__ import annotations

from itertools import combinations
import json

import pytest
import torch

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.search.combat_puct import (
    H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY, CombatFrame, CombatPuct,
)
from sts2_ai.models.packed_actor_collation import collate_h7_actor_packed
from sts2_ai.models.shared_transformer import (
    H7Config,
    SharedObjectTransformer,
    collate_h7,
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


def test_colliding_selection_tokens_feed_finite_distinct_policy_logits():
    # A PUCT leaf must be able to evaluate legal internal variants even when
    # they alias to the same publicly observed card class.
    public = state()
    public["combat"]["discard_pile"] = [
        {"instance_id": i, "card_id": "demo.same"} for i in (6, 10)
    ]
    actions = (
        LegalAction("6", "select_cards", '{"CardInstanceIds":[6]}'),
        LegalAction("10", "select_cards", '{"CardInstanceIds":[10]}'),
    )
    encoded = encode_h16_decision(public, actions)
    assert len(set(encoded.actions)) == 2
    batch = collate_h7([encoded])
    torch.manual_seed(19)
    net = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    with torch.inference_mode():
        scores, _, _ = net(batch)
    assert torch.isfinite(scores).all()
    assert scores.shape == (1, 2)
    assert not torch.allclose(scores[0, 0], scores[0, 1])


def test_puct_expands_alias_selection_node_instead_of_crashing():
    # Regression for the observed floor-5 PUCT leaf collision [6] versus [10].
    public = state()
    public["combat"]["discard_pile"] = [
        {"instance_id": i, "card_id": "demo.same"} for i in (6, 10)
    ]
    aliases = (
        LegalAction("6", "select_cards", '{"CardInstanceIds":[6]}'),
        LegalAction("10", "select_cards", '{"CardInstanceIds":[10]}'),
    )

    def observation(suffix):
        return Observation(
            FAIR_POLICY_ID, json.dumps(public), "alias-test-" + suffix,
        )

    root = CombatFrame(
        "root", observation("root"), (
            LegalAction("start-a", "end_turn", "{}"),
            LegalAction("start-b", "other_choice", '{"ChoiceId":1}'),
        ),
    )

    class Sampler:
        fair_combat_capability_id = H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY

        def begin_simulation(self, root_frame, rng):
            assert root_frame.information_key == "root"

        def step(self, current, action, rng):
            if current.information_key == "root":
                return CombatFrame(
                    "alias-" + action.action_id,
                    observation(action.action_id),
                    aliases,
                )
            return CombatFrame(
                current.information_key + "/" + action.action_id,
                observation("terminal"),
                (),
                terminal_reward=1.0,
            )

        def end_simulation(self):
            pass

    torch.manual_seed(27)
    net = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()

    def evaluator(obs, menu):
        enc = encode_h16_decision(json.loads(obs.payload_json), menu)
        with torch.inference_mode():
            logits, _, values = net(collate_h7([enc]))
        return (tuple(float(v) for v in logits[0]), float(values[0]))

    result = CombatPuct(
        Sampler(), evaluator, max_depth=2, tree_expansion_mode="full_depth",
        free_card_selection_steps=True,
    ).search(root, simulations=4)
    assert result.transitions >= 4
    assert result.maximum_expanded_depth >= 1


def test_h16_rejects_missing_public_subset_candidates():
    bad = LegalAction("select-missing", "select_cards", '{"CardInstanceIds":[999]}')
    with pytest.raises(ValueError, match="candidate missing from public"):
        encode_h16_decision(state(), [bad])


def test_h16_rejects_distinct_actions_that_share_the_same_structural_token():
    a = LegalAction("other-1", "other_choice", '{"ChoiceId":1}')
    b = LegalAction("other-2", "other_choice", '{"ChoiceId":2}')
    with pytest.raises(ValueError, match="action representation collision"):
        encode_h16_decision(state(), [a, b])


def test_h16_choice_pool_candidates_are_visible_as_selected_objects():
    public = state()
    public["combat"]["discard_pile"] = []
    public["combat"]["selection_candidates"] = [
        {"instance_id": 11, "card_id": "demo.choose.one", "cost": 1},
        {"instance_id": 12, "card_id": "demo.choose.two", "cost": 2},
    ]
    choices = (
        LegalAction("none", "select_cards", '{"CardInstanceIds":[]}'),
        LegalAction("first", "select_cards", '{"CardInstanceIds":[11]}'),
        LegalAction("second", "select_cards", '{"CardInstanceIds":[12]}'),
    )
    encoded = encode_h16_decision(public, choices)
    assert len(set(encoded.actions)) == 3
    assert encoded.actions[1].members != encoded.actions[2].members


def test_h16_v2_checkpoint_migrates_without_changing_existing_weights(tmp_path):
    from sts2_ai.training.h16_joint_act_pilot import LEGACY_SCHEMA
    from sts2_ai.training.h16_public_codec import LEGACY_H16_CODEC
    torch.manual_seed(101)
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    new_path = tmp_path / "v3.pt"
    old_path = tmp_path / "v2.pt"
    model.save(new_path)
    contents = torch.load(new_path, map_location="cpu", weights_only=True)
    contents["schema"] = LEGACY_SCHEMA
    contents["codec"] = LEGACY_H16_CODEC
    torch.save(contents, old_path)
    recovered = JointActPilotNet.load(old_path).eval()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, recovered.state_dict()[name])
    # The next checkpoint is written with the new representation schema.
    migrated = tmp_path / "migrated.pt"
    recovered.save(migrated)
    written = torch.load(migrated, map_location="cpu", weights_only=True)
    assert written["schema"] != LEGACY_SCHEMA
    assert written["codec"] != LEGACY_H16_CODEC
