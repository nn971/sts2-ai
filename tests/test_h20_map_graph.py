"""H20 generated-map adjacency, grounded node actions and equivariance."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h20_map_graph_codec import (
    CURRENT_EDGE,
    H20_OBJECT_SCHEMA,
    NEXT_EDGE,
    REACHABLE_EDGE,
    encode_h20_map_decision,
)


def state():
    return {
        "hp": 60, "max_hp": 70, "gold": 120, "act": 1, "floor": 1,
        "deck": [{"instance_id": 1, "card_id": "demo.strike",
                  "upgrade_level": 0, "state": {}}],
        "relics": [], "potions": [],
        "potion_slot_capacity": 2,
        "potion_slots_occupied": 0, "potion_slots_empty": 2,
        "current_map_node_id": "root",
        "map": [
            {"node_id": "root", "room_type": "Start", "act": 1, "floor": 0,
             "next_node_ids": ["a", "b"]},
            {"node_id": "a", "room_type": "Combat", "act": 1, "floor": 1,
             "next_node_ids": ["c"]},
            {"node_id": "b", "room_type": "Combat", "act": 1, "floor": 1,
             "next_node_ids": ["d"]},
            {"node_id": "c", "room_type": "Shop", "act": 1, "floor": 2,
             "next_node_ids": []},
            {"node_id": "d", "room_type": "Elite", "act": 1, "floor": 2,
             "next_node_ids": []},
        ],
    }


def actions(*names):
    return tuple(LegalAction(
        name, "choose_map_node",
        json.dumps({"NodeId": name}),
    ) for name in (names or ("a", "b")))


def encode(obs, opts=None):
    return encode_h20_map_decision(
        obs, actions() if opts is None else opts,
        public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1,
    )


def test_equal_room_types_are_distinct_targets_with_real_graph_edges():
    decision = encode(state())
    assert decision.schema == H20_OBJECT_SCHEMA
    assert len(decision.actions) == 2
    left, right = decision.actions
    assert left.semantic == right.semantic
    assert left.source == right.source >= 0
    assert left.target != right.target
    assert decision.objects[left.target].kind == KINDS.index("room")
    assert decision.objects[left.target].identity == decision.objects[right.target].identity
    assert decision.objects[left.target].numeric == decision.objects[right.target].numeric
    assert sum(e.relation_type == NEXT_EDGE for e in decision.relations) == 4
    assert sum(e.relation_type == CURRENT_EDGE for e in decision.relations) == 1
    assert sum(e.relation_type == REACHABLE_EDGE for e in decision.relations) == 2
    assert any(e.source == left.target and e.target != right.target
               for e in decision.relations if e.relation_type == NEXT_EDGE)


def test_reference_key_renaming_preserves_every_learned_token():
    original = state()
    replacement = copy.deepcopy(original)
    names = {"root": "mapnode987", "a": "node6", "b": "node7",
             "c": "node14", "d": "node15"}
    replacement["current_map_node_id"] = names["root"]
    for node in replacement["map"]:
        node["node_id"] = names[node["node_id"]]
        node["next_node_ids"] = [names[n] for n in node["next_node_ids"]]
    assert encode(original) == encode(replacement, actions("node6", "node7"))


def test_topology_edits_change_edges_even_when_node_multiset_is_identical():
    source = state()
    old = encode(source)
    changed = copy.deepcopy(source)
    changed["map"][1]["next_node_ids"] = ["d"]
    changed["map"][2]["next_node_ids"] = ["c"]
    other = encode(changed)
    assert other.objects == old.objects
    assert other.actions == old.actions
    assert other.relations != old.relations


def test_unknown_map_targets_and_edges_fail_closed():
    obs = state()
    with pytest.raises(ValueError, match="Unbound choose_map_node"):
        encode(obs, actions("unknown"))
    obs = state()
    obs["map"][1]["next_node_ids"] = ["missing"]
    with pytest.raises(ValueError, match="missing visible node"):
        encode(obs)
    obs = state()
    obs["map"][0]["next_node_ids"] = ["a", "a"]
    with pytest.raises(ValueError, match="Duplicate NextNodeIds"):
        encode(obs)


def test_h20_batch_and_model_are_permutation_equivariant(tmp_path):
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_map_transformer_v2 import (
        H20MapGraphTransformer,
        collate_h20,
    )
    from sts2_ai.models.shared_owner_transformer_v2 import H19OwnerTransformer
    from sts2_ai.models.shared_transformer import H7Config

    before = encode(state())
    changed = state()
    changed["map"] = [
        changed["map"][i] for i in (4, 2, 0, 3, 1)
    ]
    after = encode(changed, actions("b", "a"))
    model = H20MapGraphTransformer(
        H7Config(width=32, layers=2, heads=4, feedforward=64)).eval()
    with torch.no_grad():
        logits0, value0 = model(collate_h20([before]))
        logits1, value1 = model(collate_h20([after]))
    torch.testing.assert_close(logits0[0], logits1[0, [1, 0]],
                               atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(value0, value1, atol=2e-5, rtol=2e-5)

    path = tmp_path / "map-v1.pt"
    model.save_checkpoint(path)
    loaded = H20MapGraphTransformer.load_checkpoint(path).eval()
    with torch.no_grad():
        x, y = loaded(collate_h20([before]))
    torch.testing.assert_close(x, logits0)
    torch.testing.assert_close(y, value0)
    with pytest.raises(ValueError, match="Unsupported H19"):
        H19OwnerTransformer.load_checkpoint(path)
