"""Numerical parity and checkpoint coverage for the opt-in batched PPO path."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.training.instance_combat import add_instance_parameters
from sts2_ai.training.neural import _new_params
from sts2_ai.training.phase_split_ppo_batch import (
    collate,
    encode_decision,
    forward_batch,
)
from sts2_ai.training.phase_split_selfplay import _forward_decision, train_phase_split
from sts2_ai.training.selfplay import PublicDecision
from test_enemy_instances_v8 import combat_frame, targeted
from test_phase_split_selfplay import TwoPhaseToy


def decision(state: dict, actions: tuple[LegalAction, ...], *, phase: str) -> PublicDecision:
    payload = json.dumps(state, sort_keys=True)
    return PublicDecision(
        Observation("prototype-fair-v0", payload, "fixture-observation"),
        actions, 0, phase,
    )


@pytest.mark.parametrize("encoding", [
    "legacy", "relational_damage", "public_resources", "enemy_instances",
])
def test_batched_forward_and_gradients_match_reference(encoding: str) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    dimension, hidden = 128, 8
    combat = combat_frame()
    changed = copy.deepcopy(combat)
    changed["combat"]["enemies"][2]["statuses"]["poison"] = 40
    combat_actions = (
        targeted(3), targeted(1), targeted(2),
        LegalAction("end", "end_turn", "{}"),
    )
    cases = [
        decision(combat, combat_actions, phase="combat"),
        decision(changed, combat_actions[:2], phase="combat"),
    ]
    params = _new_params(dimension, hidden, torch)
    if encoding == "enemy_instances":
        add_instance_parameters(params, dimension, hidden, torch)

    old_values, old_logits = [], []
    for dec in cases:
        value, logits = _forward_decision(
            dec, params, dimension, torch, tactical_state_encoding=encoding,
        )
        old_values.append(value)
        old_logits.append(logits)
    old_loss = sum(
        value + logits[0] + 0.1 * torch.nn.functional.log_softmax(
            logits / 0.85, dim=0,
        )[0]
        for value, logits in zip(old_values, old_logits, strict=True)
    )
    old_grads = torch.autograd.grad(old_loss, tuple(params.values()))

    encoded = [encode_decision(dec, dimension, encoding, torch) for dec in cases]
    batched = collate(encoded, torch)
    values, logits = forward_batch(batched, params, torch)
    for i, (v, single_logits) in enumerate(
        zip(old_values, old_logits, strict=True)
    ):
        assert values[i].detach().item() == pytest.approx(v.detach().item(), abs=2e-5)
        assert logits[i, :len(single_logits)].detach().tolist() == pytest.approx(
            single_logits.detach().tolist(), abs=2e-5,
        )
        assert (logits[i, len(single_logits):] < -1e8).all()
    selected = torch.nn.functional.log_softmax(logits / 0.85, dim=1)[:, 0]
    new_loss = (values + logits[:, 0] + 0.1 * selected).sum()
    new_grads = torch.autograd.grad(new_loss, tuple(params.values()))
    for old, new in zip(old_grads, new_grads, strict=True):
        torch.testing.assert_close(old, new, rtol=4e-4, atol=4e-5)


def test_strategy_legal_menu_padding_and_gradient_parity() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    state = {
        "hp": 30, "max_hp": 70, "gold": 90,
        "act": 1, "floor": 7, "deck": [{"card_id": "proto.silent.strike"}],
    }
    choices = tuple(
        LegalAction(f"map-{i}", "choose_map_node",
                    json.dumps({"NodeId": f"node-{i}"}))
        for i in range(24)
    )
    cases = [
        decision(state, choices, phase="strategy"),
        decision(state, choices[:1], phase="strategy"),
        decision(state, choices[:7], phase="strategy"),
    ]
    dimension, hidden = 128, 8
    params = _new_params(dimension, hidden, torch)
    individual = [
        _forward_decision(dec, params, dimension, torch)
        for dec in cases
    ]
    batch = collate([encode_decision(dec, dimension, "enemy_instances", torch)
                     for dec in cases], torch)
    values, logits = forward_batch(batch, params, torch)
    for index, (value, choices_logits) in enumerate(individual):
        assert values[index].detach().item() == pytest.approx(value.detach().item(), abs=2e-5)
        assert logits[index, :len(choices_logits)].detach().tolist() == pytest.approx(
            choices_logits.detach().tolist(), abs=2e-5,
        )


def test_v8_padded_enemy_and_action_masks_do_not_cross_instances() -> None:
    torch = pytest.importorskip("torch")
    state = combat_frame()
    altered = copy.deepcopy(state)
    altered["combat"]["enemies"] = altered["combat"]["enemies"][:1]
    first = decision(state, (targeted(3), targeted(2), targeted(1)), phase="combat")
    second = decision(altered, (targeted(1),), phase="combat")
    packed = collate([
        encode_decision(first, 128, "enemy_instances", torch),
        encode_decision(second, 128, "enemy_instances", torch),
    ], torch)
    assert packed.enemies.shape == (2, 3, 128)
    assert packed.enemy_mask.tolist() == [[True, True, True], [True, False, False]]
    assert packed.targets.tolist() == [[2, 1, 0], [0, -1, -1]]
    assert packed.mask.tolist() == [[True, True, True], [True, False, False]]
    torch.manual_seed(8)
    params = _new_params(128, 8, torch)
    add_instance_parameters(params, 128, 8, torch)
    values, logits = forward_batch(packed, params, torch)
    assert values.shape == (2,)
    assert (logits[1, 1:] < -1e8).all()


def test_batched_ppo_runs_and_checkpoint_fingerprint_is_distinct(tmp_path) -> None:
    pytest.importorskip("torch")
    options = dict(
        episodes_per_round=4, dimension=128, hidden=8,
        max_decisions=8, workers=1, seed=23,
        optimizer_method="ppo", ppo_epochs=2,
        ppo_batch_size=2, ppo_sample_limit=16,
        hp_monotonic_weight=0.0, combat_objective="hp_preservation",
        tactical_state_encoding="relational_damage",
        checkpoint=tmp_path / "batched.pt",
        ppo_backend="batched",
    )
    model, rows = train_phase_split(TwoPhaseToy(), rounds=1, **options)
    assert rows[0].completed == 4
    assert rows[0].optimization_steps > 0
    assert rows[0].phase_loss_diagnostics is not None
    assert model.combat.format_id.endswith("relational-damage-tactical")
    resumed, history = train_phase_split(
        TwoPhaseToy(), rounds=2, resume=True, **options,
    )
    assert len(history) == 2 and history[0] == rows[0]
    assert resumed.model_id != model.model_id
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            TwoPhaseToy(), rounds=2, resume=True,
            **{**options, "ppo_backend": "reference"},
        )


def test_batched_ppo_rejects_bad_backend_choices() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="PPO backend"):
        train_phase_split(TwoPhaseToy(), rounds=1, episodes_per_round=2,
                          ppo_backend="nonsense")
    with pytest.raises(ValueError, match="requires PPO"):
        train_phase_split(TwoPhaseToy(), rounds=1, episodes_per_round=2,
                          ppo_backend="batched", optimizer_method="reinforce")
