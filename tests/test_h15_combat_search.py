from __future__ import annotations

import json
import random

import pytest

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.h15_combat_codec import (
    H15_COMBAT_SCHEMA, collate_h15, encode_h15_combat,
)
from sts2_ai.models.shared_game_objects import SHARED_OBJECT_SCHEMA, encode_public_decision
from sts2_ai.search.combat_puct import (
    FAIR_COMBAT_CAPABILITY, CombatFrame, CombatPuct,
)
from sts2_ai.training.h15_search_supervision import (
    decode_target, search_policy_value_loss, target_record,
)


def state(move: str = "attack") -> dict:
    return {
        "act": 1, "floor": 2, "hp": 40, "max_hp": 70,
        "deck": [], "relics": [], "potions": [], "map": [],
        "combat": {
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "enemies": [
                {"instance_id": 1, "enemy_id": "proto.enemy.foo",
                 "hp": 20, "move_id": move},
                {"instance_id": 2, "enemy_id": "proto.enemy.bar",
                 "hp": 12, "move_id": "guard"},
            ],
        },
    }


ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


def frame() -> CombatFrame:
    return CombatFrame(
        "root:observed-public-history", Observation("fair", json.dumps(state()), "sha"),
        ACTIONS,
    )


class FairToyCombat:
    fair_combat_capability_id = FAIR_COMBAT_CAPABILITY

    def __init__(self):
        self.starts = 0
        self.ends = 0

    def begin_simulation(self, root: CombatFrame, search_rng: random.Random) -> None:
        assert root.information_key.startswith("root:")
        self.starts += 1

    def step(self, frame: CombatFrame, action: LegalAction, rng: random.Random) -> CombatFrame:
        assert frame.information_key.startswith("root:")
        reward = 1.5 if action.action_id == "safe" else 2.0 if rng.random() < .25 else 0.0
        return CombatFrame(
            f"{frame.information_key}:action:{action.action_id}:"
            f"reward:{reward}", frame.observation, (), reward,
        )

    def end_simulation(self) -> None:
        self.ends += 1


def test_move_aware_schema_is_additive_and_owned():
    first = encode_h15_combat(state("inhale"), ACTIONS)
    second = encode_h15_combat(state("attack"), ACTIONS)
    old = encode_public_decision(state("inhale"), ACTIONS)
    assert old.schema == SHARED_OBJECT_SCHEMA
    assert first.schema == H15_COMBAT_SCHEMA
    assert first.actions == old.actions
    assert first.objects[:len(old.objects)] == old.objects
    assert len(first.objects) == len(old.objects) + 2
    assert first.objects[-2].owner == 2
    assert first.objects[-2].identity != second.objects[-2].identity
    assert collate_h15([first]).action_mask.sum().item() == len(ACTIONS)
    with pytest.raises(ValueError):
        collate_h15([old])


def test_combat_puct_averages_chance_and_normalizes_visits():
    sampler = FairToyCombat()
    root = frame()
    puct = CombatPuct(sampler, lambda obs, actions: ((0.0, 0.0), 0.5),
                      c_puct=1.5, seed=129)
    result = puct.search(root, simulations=200)
    assert sum(result.visits) == 200
    assert result.policy[0] > result.policy[1]
    assert result.terminal_evaluations == 200
    assert result.neural_cutoffs == 0
    assert sampler.starts == sampler.ends == 200
    assert result.root_mean_value < 0.8  # chance reward, /2


def test_search_requires_certified_sampler_and_legal_terminal_labels():
    sampler = FairToyCombat()
    sampler.fair_combat_capability_id = "oracle-exact"
    with pytest.raises(ValueError, match="certified"):
        CombatPuct(sampler, lambda *_: ((0.0, 0.0), 0.5))
    with pytest.raises(ValueError):
        CombatFrame("bad", frame().observation, (), None)
    with pytest.raises(ValueError):
        CombatFrame("bad", frame().observation, ACTIONS, 0)


def test_search_target_roundtrip_and_rejects_unfair_provenance():
    root = frame()
    result = CombatPuct(FairToyCombat(), lambda *_: ((0.0, 0.0), 0.5)).search(
        root, simulations=20,
    )
    row = target_record(root, result, terminal_reward=1.4,
                        search_model_sha256="a" * 64,
                        source_seed="native-run-1", emulator_revision="pinned-sha")
    sample = decode_target(row)
    assert sample.visits == result.visits
    assert sample.realized_reward == pytest.approx(1.4)
    row["chance_capability"] = "oracle-state"
    with pytest.raises(ValueError):
        decode_target(row)


def test_policy_value_loss_backpropagates_through_both_heads():
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer

    root = frame()
    result = CombatPuct(FairToyCombat(), lambda *_: ((0.0, 0.0), 0.5)).search(
        root, simulations=12,
    )
    row = target_record(root, result, terminal_reward=1.6,
                        search_model_sha256="a" * 64,
                        source_seed="native-run-2", emulator_revision="revision")
    network = SharedObjectTransformer(H7Config(
        width=32, layers=1, heads=4, feedforward=64,
    ))
    loss, log = search_policy_value_loss(network, [decode_target(row)])
    assert torch.isfinite(loss)
    loss.backward()
    assert network.policy_heads[0][-1].weight.grad is not None
    assert network.value_heads[0][-1].weight.grad is not None
    assert log["value_loss"] >= 0


def test_early_search_cutoffs_are_not_fabricated_terminal_returns():
    class DeepSampler(FairToyCombat):
        def step(self, frame, action, rng):
            return CombatFrame(frame.information_key + ":" + action.action_id,
                               frame.observation, ACTIONS)
    sampler = DeepSampler()
    result = CombatPuct(sampler, lambda *_: ((0.0, 0.0), .7),
                        max_depth=2).search(frame(), simulations=20)
    assert result.neural_cutoffs > 0
    assert result.terminal_evaluations == 0
    assert sum(result.visits) == 20
