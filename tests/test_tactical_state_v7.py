"""v7 public-only combat resource features: exact stacks, piles, relic progress."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator import Observation
from sts2_ai.models.neural import (
    TACTICAL_DAMAGE_FORMAT,
    TACTICAL_RESOURCES_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.tactical_state import (
    PUBLIC_RESOURCES_TACTICAL_STATE_SCHEMA,
    damage_tactical_state_features,
    public_resources_tactical_state_features,
)


def frame() -> dict:
    return {
        "hp": 40, "max_hp": 70, "act": 1, "floor": 8, "gold": 50,
        "deck": [{"card_id": "strike", "upgrade_level": 0}],
        "relics": [{"relic_id": "kunai", "state": {"charged": False}}],
        "combat": {
            "turn": 2, "energy": 2, "player_block": 0,
            "draw_pile_count": 2,
            "draw_pile": [
                {"instance_id": 101, "card_id": "strike", "upgrade_level": 0},
                {"instance_id": 102, "card_id": "defend", "upgrade_level": 1},
            ],
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "relic_counters": [{"relic_id": "kunai", "trigger_counts": [1]}],
            "player_powers": [{"power_id": "strength", "stacks": 2}],
            "enemies": [{
                "enemy_id": "slime", "instance_id": 999,
                "hp": 20, "block": 0, "move_id": "attack",
                "intent_base_damage": 6, "intent_damage": 9, "intent_hits": 1,
                "statuses": {"poison": 3},
                "powers": [{"power_id": "vulnerable", "stacks": 2}],
            }],
        },
    }


def v7(state: dict) -> dict[int, float]:
    return public_resources_tactical_state_features(state, 512)


def test_v7_distinguishes_resources_v6_drops() -> None:
    base = frame()
    assert PUBLIC_RESOURCES_TACTICAL_STATE_SCHEMA.endswith("resources")
    mutations = [
        lambda x: x["combat"]["draw_pile"][0].update(card_id="another"),
        lambda x: x["combat"]["draw_pile"][0].update(upgrade_level=2),
        lambda x: x["combat"]["relic_counters"][0].update(trigger_counts=[2]),
        lambda x: x["combat"]["player_powers"][0].update(stacks=5),
        lambda x: x["combat"]["enemies"][0]["statuses"].update(poison=22),
        lambda x: x["combat"]["enemies"][0]["powers"][0].update(stacks=4),
        lambda x: x["relics"][0]["state"].update(charged=True),
    ]
    baseline = v7(base)
    for mutate in mutations:
        changed = copy.deepcopy(base)
        mutate(changed)
        assert v7(changed) != baseline
        # The old encoder cannot distinguish any of these changes.
        assert damage_tactical_state_features(changed, 512) == (
            damage_tactical_state_features(base, 512)
        )


def test_v7_is_order_invariant_and_ignores_instance_ids() -> None:
    base = frame()
    reordered = copy.deepcopy(base)
    reordered["combat"]["draw_pile"].reverse()
    reordered["combat"]["draw_pile"][0]["instance_id"] = 2026
    reordered["combat"]["draw_pile"][1]["instance_id"] = 3099
    reordered["combat"]["enemies"][0]["instance_id"] = 142
    assert v7(reordered) == v7(base)


def test_v7_fails_closed_when_draw_contents_or_counters_unavailable() -> None:
    for name in ("draw_pile", "relic_counters"):
        bad = frame()
        del bad["combat"][name]
        with pytest.raises(ValueError, match=name):
            v7(bad)
    invalid = frame()
    invalid["combat"]["draw_pile_count"] += 1
    with pytest.raises(ValueError, match="Draw-pile count"):
        v7(invalid)


def test_v7_inference_uses_exact_relic_counter() -> None:
    state = frame()
    upgraded = copy.deepcopy(state)
    upgraded["combat"]["relic_counters"][0]["trigger_counts"] = [2]
    x, y = v7(state), v7(upgraded)
    different = next(index for index in x.keys() | y.keys() if x.get(index) != y.get(index))
    dim, hidden = 512, 4
    model = NeuralPolicyValueModel.from_dict({
        "format": TACTICAL_RESOURCES_FORMAT, "dimension": dim,
        "hidden": hidden, "model_id": "v7-counter-test",
        "state_weight": [
            [1.0 if i == different else 0.0 for i in range(dim)],
            *[[0.0] * dim for _ in range(hidden - 1)],
        ],
        "state_bias": [0.0] * hidden,
        "action_weight": [[0.0] * dim for _ in range(hidden)],
        "action_bias": [0.0] * hidden,
        "policy_weight": [0.0] * hidden,
        "policy_bias": 0.0,
        "value_weight": [1.0, 0.0, 0.0, 0.0],
        "value_bias": 0.0,
    })
    low = Observation("prototype-fair-v0", json.dumps(state), "")
    high = Observation("prototype-fair-v0", json.dumps(upgraded), "")
    assert model.evaluate(low, ()).value != model.evaluate(high, ()).value
    assert NeuralPolicyValueModel.from_dict(model.to_dict()).to_dict() == model.to_dict()
    # Old model formats continue to load, preserving the previous experiment.
    old = model.to_dict()
    old["format"] = TACTICAL_DAMAGE_FORMAT
    assert NeuralPolicyValueModel.from_dict(old).format_id == TACTICAL_DAMAGE_FORMAT


def test_v7_training_and_warm_start_migration(tmp_path) -> None:
    pytest.importorskip("torch")
    from test_phase_split_selfplay import TwoPhaseToy
    from sts2_ai.models.phase_split import PhaseSplitNeuralModel
    from sts2_ai.models.neural import NEURAL_FORMAT
    from sts2_ai.training.phase_split_selfplay import train_phase_split

    class ResourceToy(TwoPhaseToy):
        def observe(self, state, policy):
            original = super().observe(state, policy)
            obj = json.loads(original.payload_json)
            if isinstance(obj.get("combat"), dict):
                obj["combat"].update({
                    "draw_pile_count": 0, "draw_pile": [],
                    "relic_counters": [],
                })
            return Observation("prototype-fair-v0", json.dumps(obj), "")

    def zero(fmt: str) -> NeuralPolicyValueModel:
        dimension, hidden = 64, 4
        return NeuralPolicyValueModel.from_dict({
            "format": fmt, "dimension": dimension, "hidden": hidden,
            "model_id": fmt, "state_weight": [[0.0] * dimension for _ in range(hidden)],
            "state_bias": [0.0] * hidden,
            "action_weight": [[0.0] * dimension for _ in range(hidden)],
            "action_bias": [0.0] * hidden,
            "policy_weight": [0.0] * hidden, "policy_bias": 0.0,
            "value_weight": [0.0] * hidden, "value_bias": 0.0,
        })

    warm = tmp_path / "warm-v6.json"
    PhaseSplitNeuralModel(
        zero(NEURAL_FORMAT), zero(TACTICAL_DAMAGE_FORMAT),
        model_id="v6-start", combat_value_objective="hp_preservation",
    ).save(warm)
    kwargs = dict(
        episodes_per_round=2, dimension=64, hidden=4,
        max_decisions=8, seed=9, workers=1,
        tactical_state_encoding="public_resources",
        optimizer_method="ppo", ppo_epochs=1, ppo_batch_size=2,
        ppo_sample_limit=16, combat_objective="hp_preservation",
        combat_advantage_baseline="critic",
        hp_monotonic_weight=0.0, warm_start=warm,
        checkpoint=tmp_path / "v7-checkpoint.pt",
    )
    first, rows = train_phase_split(ResourceToy(), rounds=1, **kwargs)
    assert first.combat.format_id == TACTICAL_RESOURCES_FORMAT
    assert rows[0].optimization_steps > 0
    saved = PhaseSplitNeuralModel.from_dict(first.to_dict())
    assert saved.to_dict() == first.to_dict()
    second, continuation = train_phase_split(
        ResourceToy(), rounds=2, resume=True, **kwargs
    )
    assert second.combat.format_id == TACTICAL_RESOURCES_FORMAT
    assert len(continuation) == 2
    assert continuation[0] == rows[0]
