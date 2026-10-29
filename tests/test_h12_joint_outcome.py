from __future__ import annotations

import math

import torch

from sts2_ai.models.h12_joint_outcome import (
    FEATURE_DIM, H12JointOutcomeMixture, H12MacroContinuation,
    public_features, training_example,
)
from sts2_ai.training.h12_transition_data import (
    PublicJointRunTrace, public_resources,
)


def _state(*, hp: int = 50, combat: bool = False, potions=None):
    frame = {
        "act": 1, "floor": 3, "hp": hp, "max_hp": 72, "gold": 100,
        "deck": [{"card_id": "strike", "upgrade_level": 1}],
        "relics": [{"relic_id": "ring"}],
        "potions": potions if potions is not None else [
            {"slot": 0, "potion_id": "poison"},
            {"slot": 1, "potion_id": "block"},
        ],
        "map": [
            {"floor": 4, "room_type": "combat", "node_id": "ephemeral1"},
            {"floor": 5, "room_type": "rest", "node_id": "ephemeral2"},
        ],
        "terminal_outcome": None,
    }
    if combat:
        frame["combat"] = {
            "turn": 1, "energy": 3, "player_block": 0,
            "draw_pile_count": 9,
            "hand": [{"card_id": "strike", "upgrade_level": 1}],
            "enemies": [
                {"enemy_id": "beetle", "hp": 30, "block": 0,
                 "move_id": "bite", "intent_damage": 8, "intent_hits": 1}
            ],
        }
    return frame


def test_public_joint_transition_survival_and_named_inventory():
    trace = PublicJointRunTrace("train-seed-0")
    trace.observe(_state())
    trace.observe(_state(combat=True))
    trace.observe(_state(hp=31, potions=[
        {"slot": 0, "potion_id": "poison"},
    ]))
    result = trace.finish(
        outcome="defeat", censored=False, act1_clear=False,
        frontier_progress=7.0, decisions=20,
        policy_identity={"combat": "hash-c", "macro": "hash-m"},
    )
    assert len(result["combats"]) == 1
    fight = result["combats"][0]
    assert fight["survived"] is True
    assert fight["hp_delta"] == -19
    assert fight["potion_transition"] == {
        "retained": ["poison"], "consumed": ["block"], "gained": [],
    }
    assert fight["entry"]["combat"]["enemies"][0]["enemy_id"] == "beetle"
    assert fight["entry"]["public_map_future_room_counts"] == {
        "d1:combat": 1, "d2:rest": 1,
    }
    assert len(result["macro_states"]) == 2
    assert all(s["continuation_target"] > 0 for s in result["macro_states"])


def test_no_private_fields_reach_joint_training_frames():
    raw = _state(combat=True)
    raw["hidden_rng_cursor"] = "SECRET"
    raw["exact_hash"] = "PRIVATE"
    raw["run_seed"] = "PRIVATE"
    frame = public_resources(raw, include_combat=True)
    assert "hidden_rng_cursor" not in repr(frame)
    assert "SECRET" not in repr(frame)
    assert "PRIVATE" not in repr(frame)
    assert "run_seed" not in repr(frame)
    raw["map"][0]["node_id"] = "renamed"
    assert public_features(frame).equal(
        public_features(public_resources(raw, include_combat=True))
    )


def test_dead_combat_and_censored_combat_have_distinct_labels():
    dead = _state(combat=True, hp=0)
    dead["terminal_outcome"] = "defeat"
    d = PublicJointRunTrace("death")
    d.observe(_state(combat=True))
    d.observe(dead)
    defeated = d.finish(
        outcome="defeat", censored=False, act1_clear=False,
        frontier_progress=3.0, decisions=8, policy_identity={},
    )
    assert len(defeated["combats"]) == 1
    assert defeated["combats"][0]["survived"] is False

    c = PublicJointRunTrace("censored")
    c.observe(_state(combat=True))
    censored = c.finish(
        outcome="truncated", censored=True, act1_clear=False,
        frontier_progress=3.0, decisions=8, policy_identity={},
    )
    assert censored["combats"] == []
    assert censored["macro_states"] == []


def test_potion_multiplicities_and_joint_mixture_backpropagate():
    trace = PublicJointRunTrace("two-potions")
    before = _state(combat=True, potions=[
        {"slot": 0, "potion_id": "poison"},
        {"slot": 1, "potion_id": "poison"},
    ])
    after = _state(hp=35, potions=[
        {"slot": 1, "potion_id": "poison"},
    ])
    trace.observe(before)
    trace.observe(after)
    fight = trace.combats[0]
    assert fight["potion_transition"]["retained"] == ["poison"]
    assert fight["potion_transition"]["consumed"] == ["poison"]
    features, pid, mask, retained, target = training_example(fight)
    assert features.shape == (FEATURE_DIM,)
    assert mask.sum().item() == 2
    assert retained.sum().item() == 1

    torch.manual_seed(11)
    model = H12JointOutcomeMixture(components=4)
    logp = model.log_probability(
        features[None, :], pid[None, :], mask[None, :],
        retained[None, :], target[None, :],
    )
    assert logp.shape == (1,) and torch.isfinite(logp).all()
    (-logp.mean()).backward()
    assert model.encoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad.abs().sum().item() > 0
    prediction = model.predict(fight["entry"])
    assert 0 <= prediction["survival_probability"] <= 1
    assert len(prediction["named_potion_retention"]) == 2
    assert math.isclose(
        sum(prediction["mixture_weights"]), 1.0, abs_tol=1e-5,
    )


def test_macro_continuation_uses_same_public_features():
    model = H12MacroContinuation()
    states = torch.stack([
        public_features(public_resources(_state())),
        public_features(public_resources(_state(hp=23))),
    ])
    v = model(states)
    assert v.shape == (2,)
    assert torch.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
    v.sum().backward()
    assert model.network[0].weight.grad.abs().sum().item() > 0
