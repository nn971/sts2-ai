"""v15a/b fixture tests: mutable object identity, schema boundaries, gradients."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.models.object_embeddings import ObjectVocabulary, TypedObjectEncoder
from sts2_ai.models.object_instances import (
    ObjectRef,
    extract_combat_objects,
    legal_action_object_refs,
)


def frame() -> dict:
    def card(i: int, *, enchantment: object = None) -> dict:
        return {
            "instance_id": i, "card_id": "proto.silent.strike", "upgrade_level": 0,
            "is_temporary": False, "state": {"times_played": 0},
            "affliction": None, "enchantment": enchantment,
        }
    return {
        "character_id": "proto.silent", "hp": 40, "max_hp": 70, "act": 1,
        "floor": 10, "gold": 20, "deck": [card(77)],
        "relics": [{"relic_id": "proto.relic.demo", "state": {"charged": True}}],
        "potions": [
            {"slot": 0, "potion_id": "proto.potion.demo", "state": {}},
            {"slot": 2, "potion_id": "proto.potion.demo", "state": {}},
        ],
        "combat": {
            "turn": 3, "energy": 2, "player_block": 0,
            "hand": [card(11), card(12, enchantment={"kind": "spiral"})],
            "draw_pile": [card(13, enchantment={"kind": "sown"}), card(14)],
            "draw_pile_count": 2, "discard_pile": [], "exhaust_pile": [],
            "play_pile": [],
            "relic_counters": [{"relic_id": "proto.relic.demo", "trigger_counts": [2, 1]}],
            "player_powers": [{"power_id": "strength", "stacks": 2}],
            "enemies": [
                {"instance_id": 1, "enemy_id": "proto.enemy.nibbit", "hp": 22,
                 "block": 0, "move_id": "bite", "formation_position": 0,
                 "powers": [], "statuses": {"vulnerable": 2}},
                {"instance_id": 2, "enemy_id": "proto.enemy.nibbit", "hp": 15,
                 "block": 9, "move_id": "bite", "formation_position": 1,
                 "powers": [], "statuses": {"vulnerable": 1}},
            ],
        },
    }


def test_instances_are_never_merged_and_actions_link_exact_instances() -> None:
    objects = extract_combat_objects(frame())
    assert len([t for t in objects.tokens if t.kind == "card"]) == 4
    assert len([t for t in objects.tokens if t.kind == "enemy"]) == 2
    assert len([t for t in objects.tokens if t.kind == "potion"]) == 2
    assert len([t for t in objects.tokens if t.kind == "relic"]) == 1
    assert len([t for t in objects.tokens if t.kind == "status"]) == 2
    assert objects.index()[ObjectRef("card", 12)] != objects.index()[ObjectRef("card", 11)]
    assert legal_action_object_refs("play_card", json.dumps({
        "CardInstanceId": 12, "TargetEnemyId": 2,
    }), objects) == (ObjectRef("card", 12), ObjectRef("enemy", 2))
    assert legal_action_object_refs("use_potion", '{"Slot":2,"TargetEnemyId":1}', objects) == (
        ObjectRef("potion", 2), ObjectRef("enemy", 1),
    )
    with pytest.raises(ValueError, match="absent object"):
        legal_action_object_refs("play_card", '{"CardInstanceId":999}', objects)


def test_pending_combat_card_selection_preserves_chosen_instance_set() -> None:
    objects = extract_combat_objects(frame())
    assert legal_action_object_refs(
        "select_cards", '{"CardInstanceIds":[11,13]}', objects,
    ) == (ObjectRef("card", 11), ObjectRef("card", 13))
    assert legal_action_object_refs(
        "select_cards", '{"CardInstanceIds":[12,14]}', objects,
    ) == (ObjectRef("card", 12), ObjectRef("card", 14))
    assert legal_action_object_refs(
        "select_cards", '{"CardInstanceIds":[]}', objects,
    ) == ()
    assert legal_action_object_refs(
        "select_cards", '{"card_instance_ids":[14]}', objects,
    ) == (ObjectRef("card", 14),)
    with pytest.raises(ValueError, match="Duplicate card"):
        legal_action_object_refs(
            "select_cards", '{"CardInstanceIds":[11,11]}', objects,
        )
    with pytest.raises(ValueError, match="absent object"):
        legal_action_object_refs(
            "select_cards", '{"CardInstanceIds":[999]}', objects,
        )
    with pytest.raises(ValueError, match="Missing CardInstanceIds"):
        legal_action_object_refs("select_cards", "{}", objects)
    with pytest.raises(ValueError, match="Missing CardInstanceId"):
        legal_action_object_refs("play_card", "{}", objects)


def test_multiple_instanced_powers_on_one_owner_keep_distinct_references() -> None:
    state = frame()
    strangle = "proto.power.strangle"
    first = state["combat"]["enemies"][0]
    second = state["combat"]["enemies"][1]
    first["powers"] = [
        {"power_id": strangle, "stacks": 1},
        {"power_id": strangle, "stacks": 3},
        {"power_id": "proto.power.strength", "stacks": 2},
    ]
    second["powers"] = [{"power_id": strangle, "stacks": 4}]
    state["combat"]["player_powers"].extend([
        {"power_id": "strength", "stacks": 5},
        {"power_id": strangle, "stacks": 6},
        {"power_id": strangle, "stacks": 7},
    ])
    objects = extract_combat_objects(state)
    assert len(objects.index()) == len(objects.tokens)
    powers = [token for token in objects.tokens if token.kind == "power"]
    assert len(powers) == 8
    first_owner = ObjectRef("enemy", 1)
    same_owner = [
        token for token in powers
        if token.owner == first_owner and token.type_id == strangle
    ]
    assert len(same_owner) == 2
    assert len({token.ref for token in same_owner}) == 2
    assert [dict(token.numeric)["stacks"] for token in same_owner] == [1, 3]
    assert all(token.owner is not None for token in powers)
    # Power occurrences are lookup identities, not embeddings. Neither
    # relabeling enemy IDs nor shuffling same-type powers changes features.
    changed = copy.deepcopy(state)
    changed["combat"]["enemies"][0]["powers"].reverse()
    changed["combat"]["enemies"][0]["instance_id"] += 100
    changed["combat"]["enemies"][1]["instance_id"] += 200
    original_features = sorted(
        (t.kind, t.type_id, t.role, t.numeric, t.categorical)
        for t in objects.tokens
    )
    new_objects = extract_combat_objects(changed)
    assert len(new_objects.index()) == len(new_objects.tokens)
    changed_features = sorted(
        (t.kind, t.type_id, t.role, t.numeric, t.categorical)
        for t in new_objects.tokens
    )
    assert changed_features == original_features


def test_mutations_enchantments_and_transforms_reembed_one_instance() -> None:
    state = frame()
    base = extract_combat_objects(state).tokens
    card = next(t for t in base if t.ref == ObjectRef("card", 11))
    changed = copy.deepcopy(state)
    changed["combat"]["hand"][0]["enchantment"] = {"kind": "sown"}
    enchanted = next(t for t in extract_combat_objects(changed).tokens if t.ref == card.ref)
    assert enchanted.type_id == card.type_id and enchanted.categorical != card.categorical
    changed["combat"]["hand"][0]["state"]["times_played"] = 2
    evolved = next(t for t in extract_combat_objects(changed).tokens if t.ref == card.ref)
    assert evolved.numeric != enchanted.numeric
    changed["combat"]["hand"][0]["card_id"] = "proto.native.event.peck"
    transformed = next(t for t in extract_combat_objects(changed).tokens if t.ref == card.ref)
    assert transformed.type_id != card.type_id
    # Never accidentally count persistent deck as a second combat object.
    assert sum(t.kind == "card" for t in extract_combat_objects(changed).tokens) == 4


def test_relabeling_and_reordering_only_permutes_tokens() -> None:
    original = frame()
    changed = copy.deepcopy(original)
    changed["combat"]["hand"].reverse()
    changed["combat"]["draw_pile"].reverse()
    changed["combat"]["enemies"].reverse()
    # Renumber *pointers*, not any features of the instance itself.
    for zone in ("hand", "draw_pile"):
        for card in changed["combat"][zone]:
            card["instance_id"] += 500
    for enemy in changed["combat"]["enemies"]:
        enemy["instance_id"] += 200
    a = extract_combat_objects(original)
    b = extract_combat_objects(changed)
    def features(obj):
        return (obj.kind, obj.type_id, obj.role, obj.numeric, obj.categorical)
    assert sorted(map(repr, map(features, a.tokens))) == sorted(map(repr, map(features, b.tokens)))
    assert a.tokens[0] == b.tokens[0]
    assert len(a.index()) == len(b.index())
    assert b.index()[ObjectRef("card", 512)] >= 0


def test_fails_closed_on_invalid_public_objects_and_unobserved_enchantment() -> None:
    bad = frame()
    bad["combat"]["discard_pile"] = [bad["combat"]["hand"][0]]
    with pytest.raises(ValueError, match="multiple combat zones"):
        extract_combat_objects(bad)
    bad = frame()
    bad["combat"]["draw_pile_count"] = 3
    with pytest.raises(ValueError, match="draw_pile count"):
        extract_combat_objects(bad)
    bad = frame()
    bad["combat"]["enemies"][0]["intent_damage"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        extract_combat_objects(bad)
    # Pinned emulator currently does not expose enchantment; absence stays
    # ABSENCE, not fabricated 'no enchantment' or a private state lookup.
    bad = frame()
    del bad["combat"]["hand"][0]["enchantment"]
    token = next(t for t in extract_combat_objects(bad).tokens if t.ref == ObjectRef("card", 11))
    assert all(not t.startswith("enchantment") for t in token.categorical)



def test_affliction_source_instance_ids_remain_pointers_not_features() -> None:
    state = frame()
    card = state["combat"]["hand"][0]
    card["affliction"] = {
        "kind": "wounded",
        "source_enemy_id": "proto.enemy.nibbit",
        "source_enemy_instance_id": 1,
        "related_instance_ids": [1, 2],
        "sourceEnemyInstanceId": 1,
    }
    before = extract_combat_objects(state)
    altered = copy.deepcopy(state)
    altered_card = altered["combat"]["hand"][0]
    altered_card["affliction"]["source_enemy_instance_id"] = 2001
    altered_card["affliction"]["related_instance_ids"] = [201, 202]
    altered_card["affliction"]["sourceEnemyInstanceId"] = 2001
    after = extract_combat_objects(altered)
    i = before.index()[ObjectRef("card", 11)]
    assert before.tokens[i] == after.tokens[after.index()[ObjectRef("card", 11)]]
    assert all("instance_id" not in key for key, _ in before.tokens[i].numeric)
    assert all("InstanceId" not in tag for tag in before.tokens[i].categorical)


def test_typed_embeddings_gradients_and_frozen_vocabulary_roundtrip() -> None:
    torch = pytest.importorskip("torch")
    state = frame()
    objects = extract_combat_objects(state)
    vocabulary = ObjectVocabulary.from_object_sets([objects])
    assert ObjectVocabulary.from_dict(vocabulary.to_dict()) == vocabulary
    torch.manual_seed(17)
    encoder = TypedObjectEncoder(vocabulary, width=64)
    initial = encoder.encode(objects)
    assert initial.shape == (len(objects.tokens), 64)
    assert torch.isfinite(initial).all().item()
    different = objects.index()[ObjectRef("card", 12)]
    same_type = objects.index()[ObjectRef("card", 11)]
    assert not torch.allclose(initial[different], initial[same_type])
    initial.square().sum().backward()
    assert encoder.module.identities.weight.grad is not None
    assert encoder.module.numeric_fields.weight.grad is not None
    # ID renumbering cannot change the complete embedding multiset.
    changed = copy.deepcopy(state)
    changed["combat"]["hand"].reverse()
    changed["combat"]["enemies"].reverse()
    for card in changed["combat"]["hand"]:
        card["instance_id"] += 100
    for enemy in changed["combat"]["enemies"]:
        enemy["instance_id"] += 100
    reordered = encoder.encode(extract_combat_objects(changed))
    assert torch.allclose(initial.mean(dim=0), reordered.mean(dim=0), atol=1e-6)
    # Vocabulary frozen: unknown variants map to reserved OOV entries.
    alien = copy.deepcopy(state)
    alien["combat"]["hand"][0]["card_id"] = "previously-unseen-card"
    assert torch.isfinite(encoder.encode(extract_combat_objects(alien))).all().item()
