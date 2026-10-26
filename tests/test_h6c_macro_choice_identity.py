"""H6-C regression: public shop/event choice collapse and v18 parity."""
from __future__ import annotations

import json

import pytest
from test_h6c_macro_attention import sample
from test_phase_split_selfplay import TwoPhaseToy, small_model

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.macro_choice_identity import (
    MACRO_CHOICE_FORMAT,
    choice_tokens,
    macro_choice_action_features,
    public_choice_label,
)
from sts2_ai.models.macro_objects import MACRO_ATTENTION_FORMAT
from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.phase_split_selfplay import _forward_decision, train_phase_split
from sts2_ai.training.selfplay import PublicDecision


def shop_event():
    state, _ = sample()
    state.update({
        "shop": {
            "card_offers": [
                {"offer_id": 11, "item_id": "silent.poison", "price": 53},
                {"offer_id": 12, "item_id": "silent.backflip", "price": 75},
            ],
            "potion_offers": [
                {"offer_id": 13, "item_id": "swift_potion", "price": 44},
                {"offer_id": 14, "item_id": "strength_potion", "price": 58},
            ],
            "relic_offers": [
                {"offer_id": 15, "item_id": "kunai", "price": 134},
                {"offer_id": 16, "item_id": "paper_krane", "price": 147},
            ],
        },
        "event_id": "proto.native.event.trial",
        "event": {
            "event_id": "proto.native.event.trial",
            "offered_choice_ids": ["rest", "sacrifice"],
            "visible_choice_values": [
                {"choice_id": "rest", "heal": 15},
                {"choice_id": "sacrifice", "hp_loss": 7, "gold_gain": 100},
            ],
        },
        "reward": {
            "relic_options": ["kunai", "paper_krane"],
            "relic_resolved": False, "independent_selection": True,
        },
    })
    actions = (
        LegalAction("buy11", "buy_card", '{"OfferId":11}'),
        LegalAction("buy12", "buy_card", '{"OfferId":12}'),
        LegalAction("buy13", "buy_potion", '{"OfferId":13}'),
        LegalAction("buy14", "buy_potion", '{"OfferId":14}'),
        LegalAction("buy15", "buy_relic", '{"OfferId":15}'),
        LegalAction("buy16", "buy_relic", '{"OfferId":16}'),
        LegalAction("rest", "event_choice", '{"ChoiceId":"rest"}'),
        LegalAction("sacrifice", "event_choice", '{"ChoiceId":"sacrifice"}'),
        LegalAction("relic1", "take_reward_relic", '{"Index":0}'),
        LegalAction("relic2", "take_reward_relic", '{"Index":1}'),
        LegalAction("skip", "skip_reward_relic", "{}"),
    )
    return state, actions


def test_public_choice_tokens_distinguish_every_visible_option():
    state, actions = shop_event()
    feats = [macro_choice_action_features(state, a.kind, a.payload_json, 1024)
             for a in actions]
    for one, two in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)):
        assert feats[one] != feats[two]
    assert feats[8] != feats[10]
    assert "silent.poison" in public_choice_label(state, "buy_card", '{"OfferId":11}')
    assert "heal=15" in public_choice_label(state, "event_choice",
                                            '{"ChoiceId":"rest"}')
    assert "paper_krane" in public_choice_label(
        state, "take_reward_relic", '{"Index":1}'
    )
    assert "skip" in public_choice_label(state, "skip_reward_relic", "{}")


def test_opaque_offer_ids_are_lookup_only_and_unrevealed_effects_are_omitted():
    state, _ = shop_event()
    original = choice_tokens(state, "buy_card", '{"OfferId":11}')
    state["shop"]["card_offers"][0]["offer_id"] = 999
    assert original == choice_tokens(state, "buy_card", '{"OfferId":999}')
    assert original != choice_tokens(state, "buy_card", '{"OfferId":11}')
    event = choice_tokens(state, "event_choice", '{"ChoiceId":"rest"}')
    assert not any("future_random" in key for key, _ in event)


def test_v17_tied_logits_are_expected_but_v18_distinguishes():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.enemy_attention import add_attention_parameters
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.neural import _export, _new_params

    torch.manual_seed(27)
    params = _new_params(128, 16, torch)
    add_instance_parameters(params, 128, 16, torch)
    add_attention_parameters(params, 16, torch)
    state, actions = shop_event()
    obs = Observation("prototype-fair-v0", json.dumps(state), "test")
    v17 = _export(params, 128, 16, model_id="v17",
                  value_head_trained=True, format_id=MACRO_ATTENTION_FORMAT)
    v18 = _export(params, 128, 16, model_id="v18",
                  value_head_trained=True, format_id=MACRO_CHOICE_FORMAT)
    before = v17.evaluate(obs, actions).action_logits
    after = v18.evaluate(obs, actions).action_logits
    for one, two in ((0, 1), (2, 3), (4, 5), (6, 7)):
        assert before[one] == before[two], "v17 collision must be reproducible"
        assert after[one] != pytest.approx(after[two], abs=1e-9)

    decision = PublicDecision(obs, actions, 0, "strategy")
    value, logits = _forward_decision(
        decision, params, 128, torch, strategy_encoding="macro_attention_v2"
    )
    assert v18.evaluate(obs, actions).value == pytest.approx(value.item(), abs=3e-5)
    assert v18.evaluate(obs, actions).action_logits == pytest.approx(
        logits.detach().tolist(), abs=3e-5,
    )

    from sts2_ai.training.phase_split_ppo_batch import (
        collate,
        encode_decision,
        forward_batch,
    )
    rec = encode_decision(decision, 128, "enemy_instances", torch,
                          strategy_encoding="macro_attention_v2")
    batch = collate([rec], torch)
    bval, blogits = forward_batch(batch, params, torch)
    torch.testing.assert_close(bval[0], value, atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(
        blogits[0, :len(actions)], logits, atol=4e-5, rtol=4e-5,
    )


def test_v18_checkpoint_remains_separate_and_combat_frozen(tmp_path):
    pytest.importorskip("torch")
    original = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT),
        small_model(TACTICAL_FORMAT),
        "h6c-v18-toy",
    )
    warm = tmp_path / "warm.json"
    original.save(warm)
    settings = {
        "backend": TwoPhaseToy(), "episodes_per_round": 6,
        "dimension": 32, "hidden": 4, "max_decisions": 8,
        "workers": 1, "optimizer_method": "ppo",
        "ppo_backend": "batched", "hp_monotonic_weight": 0,
        "tactical_state_encoding": "legacy",
        "combat_objective": "continuation",
        "warm_start": warm, "freeze_combat": True,
        "strategy_encoding": "macro_attention_v2",
        "temperature_start": 1.0, "temperature_end": 1.0,
        "checkpoint": tmp_path / "v18.pt",
    }
    model, rows = train_phase_split(**settings, rounds=1)
    assert rows[0].strategic_decisions > 0
    assert model.strategy.format_id == MACRO_CHOICE_FORMAT
    assert model.combat.to_dict() == original.combat.to_dict()
    reloaded = PhaseSplitNeuralModel.from_dict(model.to_dict())
    assert reloaded.strategy.format_id == MACRO_CHOICE_FORMAT



def test_replay_identifies_legacy_feature_aliases_not_merely_logit_ties():
    from types import SimpleNamespace

    from sts2_ai.evaluation.macro_trace import feature_alias_groups

    state, actions = shop_event()
    old = SimpleNamespace(strategy=SimpleNamespace(
        format_id=MACRO_ATTENTION_FORMAT, dimension=128,
    ))
    new = SimpleNamespace(strategy=SimpleNamespace(
        format_id=MACRO_CHOICE_FORMAT, dimension=128,
    ))
    old_aliases = feature_alias_groups(old, state, actions)
    assert any({0, 1}.issubset(group) for group in map(set, old_aliases))
    assert any({2, 3}.issubset(group) for group in map(set, old_aliases))
    assert any({6, 7}.issubset(group) for group in map(set, old_aliases))
    new_aliases = feature_alias_groups(new, state, actions)
    assert not any({0, 1}.issubset(group) for group in map(set, new_aliases))
    assert not any({2, 3}.issubset(group) for group in map(set, new_aliases))
    assert not any({6, 7}.issubset(group) for group in map(set, new_aliases))
