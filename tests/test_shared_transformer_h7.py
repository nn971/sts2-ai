"""Opt-in H7 codec/backbone gates; independent of the emulator runtime."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from sts2_ai.models.shared_game_objects import (  # noqa: E402
    NUMERIC_FIELDS,
    SHARED_OBJECT_SCHEMA,
    encode_public_decision,
)
from sts2_ai.models.shared_transformer import (  # noqa: E402
    H7Config,
    SharedObjectTransformer,
    collate_h7,
)


def actions(*rows):
    return [SimpleNamespace(kind=kind, payload_json=json.dumps(payload)) for kind, payload in rows]


def frame():
    return {
        "hp": 44, "max_hp": 70, "floor": 8,
        "relics": [{"relic_id": "Lantern"}],
        "potions": [{"slot": 0, "potion_id": "PoisonPotion"}],
        "deck": [{"card_id": "Neutralize"}, {"card_id": "Strike"}],
        "combat": {
            "energy": 3, "player_block": 2, "draw_pile_count": 6, "turn": 2,
            "hand": [
                {"instance_id": 51, "card_id": "Strike", "cost": 1},
                {"instance_id": 52, "card_id": "Neutralize", "cost": 0},
            ],
            "enemies": [
                {"instance_id": 201, "enemy_id": "Mawler", "hp": 35, "max_hp": 42,
                 "block": 0, "move_id": "mawler_attack", "intent_damage": 9,
                 "statuses": {"weak": 1}},
                {"instance_id": 202, "enemy_id": "Mawler", "hp": 18, "max_hp": 42,
                 "block": 5, "move_id": "mawler_defend", "intent_damage": 0,
                 "statuses": {"poison": 2}},
            ],
        },
    }


def legal():
    return actions(
        ("play_card", {"CardInstanceId": 51, "TargetEnemyId": 201}),
        ("play_card", {"CardInstanceId": 51, "TargetEnemyId": 202}),
        ("play_card", {"CardInstanceId": 52, "TargetEnemyId": 202}),
        ("end_turn", {}),
    )


def model():
    torch.manual_seed(41)
    return SharedObjectTransformer(H7Config(width=64, layers=2, heads=4, feedforward=128))


def test_public_identity_reuse_and_zero_vs_missing():
    encoded = encode_public_decision(frame(), legal())
    assert encoded.schema == SHARED_OBJECT_SCHEMA
    assert encoded.phase == "combat"
    hand = [x for x in encoded.objects if x.zone == 1]
    deck = [x for x in encoded.objects if x.zone == 2]
    assert hand[0].identity == deck[1].identity  # same card semantic ID, new context
    enemy = [x for x in encoded.objects if x.zone == 5]
    hp = NUMERIC_FIELDS.index("hp")
    block = NUMERIC_FIELDS.index("block")
    assert enemy[0].numeric[hp] != enemy[1].numeric[hp]
    assert enemy[0].numeric[block] == 0 and enemy[0].observed[block] == 1
    assert encoded.actions[0].source == encoded.actions[1].source
    assert encoded.actions[0].target != encoded.actions[1].target


def test_id_renaming_enemy_permutation_and_action_equivariance():
    old = encode_public_decision(frame(), legal())
    state = copy.deepcopy(frame())
    state["combat"]["enemies"].reverse()
    state["combat"]["enemies"][0]["instance_id"] = 902
    state["combat"]["enemies"][1]["instance_id"] = 901
    state["combat"]["hand"][0]["instance_id"] = 551
    state["combat"]["hand"][1]["instance_id"] = 552
    changed_actions = actions(
        ("end_turn", {}),
        ("play_card", {"CardInstanceId": 552, "TargetEnemyId": 902}),
        ("play_card", {"CardInstanceId": 551, "TargetEnemyId": 902}),
        ("play_card", {"CardInstanceId": 551, "TargetEnemyId": 901}),
    )
    changed = encode_public_decision(state, changed_actions)
    net = model().eval()
    with torch.no_grad():
        logits0, value0 = net(collate_h7([old]))
        logits1, value1 = net(collate_h7([changed]))
    torch.testing.assert_close(logits0[0], logits1[0, [3, 2, 1, 0]], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(value0, value1, atol=2e-5, rtol=2e-5)


def test_padding_masks_and_unequal_legal_menu():
    first = encode_public_decision(frame(), legal())
    second = encode_public_decision(frame(), legal()[:1])
    net = model().eval()
    with torch.no_grad():
        logits, values = net(collate_h7([first, second]))
        alone, val_alone = net(collate_h7([second]))
    torch.testing.assert_close(logits[1, 0], alone[0, 0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(values[1], val_alone[0], atol=2e-5, rtol=2e-5)
    assert (logits[1, 1:] < -1e20).all()


def test_all_blocks_trainable_and_checkpoint_roundtrip(tmp_path: Path):
    net = model()
    batch = collate_h7([encode_public_decision(frame(), legal())])
    logits, value = net(batch)
    assert logits.shape == (1, 4) and value.shape == (1,)
    loss = -torch.log_softmax(logits, dim=1)[0, 0] + (value[0] - 0.3).square()
    loss.backward()
    assert net.encoder.layers[0].linear1.weight.grad.abs().sum() > 0
    assert net.encoder.layers[-1].linear2.weight.grad.abs().sum() > 0
    assert net.action_attention.in_proj_weight.grad.abs().sum() > 0
    assert net.policy_heads[0][-1].weight.grad.abs().sum() > 0
    assert net.value_heads[0][-1].weight.grad.abs().sum() > 0
    net.eval()
    path = tmp_path / "h7.pt"
    net.save_checkpoint(path)
    restored = SharedObjectTransformer.load_checkpoint(path).eval()
    with torch.no_grad():
        new_logits, new_value = restored(batch)
        ref_logits, ref_value = net(batch)
    torch.testing.assert_close(new_logits, ref_logits, rtol=0, atol=0)
    torch.testing.assert_close(new_value, ref_value, rtol=0, atol=0)


def test_missing_targets_fail_instead_of_collapsing():
    with pytest.raises(ValueError, match="invisible enemy"):
        encode_public_decision(frame(), actions(("play_card", {
            "CardInstanceId": 51, "TargetEnemyId": 999,
        })))
    with pytest.raises(ValueError, match="visible hand"):
        encode_public_decision(frame(), actions(("play_card", {"CardInstanceId": 999})))


def test_macro_can_share_embedding_vocabulary_but_uses_independent_heads():
    macro = {
        "hp": 40, "max_hp": 70, "floor": 5, "gold": 66,
        "deck": [{"card_id": "Strike"}],
        "relics": [{"relic_id": "Lantern"}],
    }
    decision = encode_public_decision(macro, actions(("skip_reward", {})))
    combat = encode_public_decision(frame(), legal())
    assert decision.phase == "macro"
    assert {x.identity for x in decision.objects} & {x.identity for x in combat.objects}
    net = model()
    logits, values = net(collate_h7([combat, decision]))
    assert logits.shape == (2, 4) and values.shape == (2,)
    assert torch.isfinite(logits[1, 0])
    assert (logits[1, 1:] < -1e20).all()


def test_h7_public_actor_uses_existing_p0_menu_without_teacher(tmp_path: Path):
    from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
    from sts2_ai.models.h7_public_policy import H7PlayerSpec

    checkpoint = tmp_path / "fresh-random-h7.pt"
    model().save_checkpoint(checkpoint)
    spec = H7PlayerSpec.from_file(checkpoint, seed=19, temperature=0.85, greedy=False)
    actor = FrozenP0Player(
        spec.load_pinned(), actor_seed=spec.episode_seed(
            fixture_seed="h7-test-world", fixture_id="h7-p0",
        ), temperature=spec.temperature, greedy=False,
        action_scope="all-legal-v1", capture_training=True,
    )
    legal_source = [
        {"actionId": str(i), "kind": action.kind,
         "payloadJson": action.payload_json}
        for i, action in enumerate(legal())
    ]
    selected = actor.choose(frame(), legal_source)
    assert selected in {str(i) for i in range(len(legal_source))}
    assert actor.decisions == 1
    row = actor.training_actions[0]
    assert row["selected_id"] == selected
    assert row["behavior_model_id"].startswith("h7-")
    assert row["source_legal_actions"] == legal_source
    assert row["source_observation"] == frame()


def test_large_combat_legal_menu_is_encoded_without_truncation():
    # Native combat can expose more than 128 distinct card/target actions.
    state = frame()
    hand = [
        {"instance_id": i + 1000, "card_id": f"card-{i}", "cost": 1}
        for i in range(48)
    ]
    enemies = [
        {"instance_id": i + 2000, "enemy_id": f"enemy-{i}",
         "hp": 25, "max_hp": 25}
        for i in range(4)
    ]
    state["combat"]["hand"] = hand
    state["combat"]["enemies"] = enemies
    menu = actions(*[
        ("play_card", {"CardInstanceId": card["instance_id"],
                       "TargetEnemyId": enemy["instance_id"]})
        for card in hand for enemy in enemies
    ])
    assert len(menu) == 192
    wide = encode_public_decision(state, menu)
    narrow = encode_public_decision(state, menu[:1])
    assert len(wide.actions) == len(menu)
    assert len(set(wide.actions)) == len(menu)
    batch = collate_h7([wide, narrow])
    assert batch.action_mask.shape == (2, 192)
    assert batch.action_mask[0].sum().item() == 192
    assert batch.action_mask[1].sum().item() == 1

    net = model().eval()
    with torch.no_grad():
        logits, values = net(batch)
        single_logits, single_values = net(collate_h7([narrow]))
    assert logits.shape == (2, 192)
    assert torch.isfinite(logits[0]).all()
    assert (logits[1, 1:] < -1e20).all()
    torch.testing.assert_close(logits[1, 0], single_logits[0, 0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(values[1], single_values[0], atol=2e-5, rtol=2e-5)
