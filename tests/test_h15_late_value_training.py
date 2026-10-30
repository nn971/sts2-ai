"""Contract checks for M16's H15 late-combat realized-value supervision."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from sts2_ai.emulator.chance import PublicHistoryStep  # noqa: E402
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID  # noqa: E402
from sts2_ai.emulator.protocol import LegalAction, Observation  # noqa: E402
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer  # noqa: E402
from sts2_ai.training.h15_native_search_collector import (  # noqa: E402
    select_late_value_indices,
)
from sts2_ai.training.h15_search_supervision import (  # noqa: E402
    RealizedValueSample,
    SearchTrainingSample,
    VALUE_SAMPLE_SCHEMA,
    decode_target,
    search_policy_value_loss,
    value_target_record,
)
from sts2_ai.models.h15_combat_codec import encode_h15_combat  # noqa: E402

ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


def observed(turn: int, hp: int, enemy_hp: int) -> Observation:
    state = {
        "act": 1, "floor": 6, "hp": hp, "max_hp": 70,
        "deck": [], "relics": [], "potions": [], "map": [],
        "combat": {
            "turn": turn, "energy": 4, "player_block": 0,
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "enemies": [{"instance_id": 1, "enemy_id": "boss",
                         "hp": enemy_hp, "move_id": "attack"}],
        },
    }
    return Observation(
        FAIR_POLICY_ID, json.dumps(state), f"public:{turn}:{hp}:{enemy_hp}",
    )


def test_selects_observable_late_turns_and_near_lethal_changes_only():
    pairs = [
        (5, 45, 100), (7, 40, 80),
        (8, 23, 70), (8, 23, 55), (8, 23, 11),
        (9, 20, 9),
    ]
    history = [
        PublicHistoryStep(observed(*row), ACTIONS[0], ACTIONS)
        for row in pairs
    ]
    assert select_late_value_indices(
        history, min_turn=7, max_player_hp=25, max_enemy_hp=40,
        max_samples=2, exclude=set(),
    ) == [2, 4]
    assert select_late_value_indices(
        history, min_turn=7, max_player_hp=25, max_enemy_hp=40,
        max_samples=2, exclude={2},
    ) == [4, 5]
    assert select_late_value_indices(
        history, min_turn=7, max_player_hp=0, max_enemy_hp=0,
        max_samples=4, exclude=set(),
    ) == []


def test_value_only_target_has_realized_reward_but_no_search_visit_claim():
    obs = observed(11, 3, 8)
    row = value_target_record(
        obs, ACTIONS, terminal_reward=1.1,
        search_model_sha256="a" * 64, source_seed="boss-seed-1",
        emulator_revision="revision", decision_index=63,
    )
    assert row["schema"] == VALUE_SAMPLE_SCHEMA
    assert row["realized_reward"] == pytest.approx(1.1)
    assert row["source_decision_index"] == 63
    assert "visits" not in row and "search_simulations" not in row
    sample = decode_target(row)
    assert isinstance(sample, RealizedValueSample)
    assert sample.realized_reward == pytest.approx(1.1)
    for forged in ({"visits": [1, 0]}, {"search_simulations": 32}):
        with pytest.raises(ValueError, match="must not claim"):
            decode_target({**row, **forged})


def test_value_only_gradient_updates_critic_but_cannot_imprint_policy():
    value = decode_target(value_target_record(
        observed(11, 3, 8), ACTIONS, terminal_reward=0.8,
        search_model_sha256="a" * 64, source_seed="case-1",
        emulator_revision="revision", decision_index=62,
    ))
    policy = SearchTrainingSample(
        decision=encode_h15_combat(
            json.loads(observed(1, 70, 262).payload_json), ACTIONS,
        ),
        visits=(6, 2), realized_reward=1.5,
        search_model_sha256="a" * 64, source_seed="case-1",
    )
    model = SharedObjectTransformer(H7Config(
        width=32, layers=1, heads=4, feedforward=64,
    ))
    loss, info = search_policy_value_loss(model, [policy, value])
    assert torch.isfinite(loss)
    assert info["policy_samples"] == 1
    assert info["value_only_samples"] == 1
    loss.backward()
    assert model.value_heads[0][-1].weight.grad is not None
    assert model.policy_heads[0][-1].weight.grad is not None

    model.zero_grad(set_to_none=True)
    only_value_loss, only_value_info = search_policy_value_loss(model, [value])
    assert only_value_info["policy_samples"] == 0
    assert only_value_info["policy_loss"] == 0
    assert only_value_info["policy_kl_to_search"] == 0
    only_value_loss.backward()
    assert model.value_heads[0][-1].weight.grad is not None
    policy_grad = model.policy_heads[0][-1].weight.grad
    assert policy_grad is None or torch.count_nonzero(policy_grad) == 0


def test_completed_fight_without_public_danger_target_is_accounted():
    from tools.train_h15_reference_rounds import aggregate

    row = {
        "case_id": "short-boss-combat",
        "encounter": "ceremonial-beast-boss-a10",
        "deck": "silent-boss-cycle-20-v1",
        "outcome": "defeat", "reward": 0.7, "exit_hp": 0,
        "decisions": 3, "searched_roots": 0, "search_overrides": 0,
        "searched_root_diagnostics": [], "late_value_targets": 0,
        "targets": [],
    }
    report = aggregate([row])
    assert report["combats"] == 1
    assert report["combats_without_training_targets"] == 1
    assert report["searched_roots"] == 0
