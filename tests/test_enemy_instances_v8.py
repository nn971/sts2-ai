"""Enemy-instance identity, species knowledge, target pointers and Torch parity."""
from __future__ import annotations

import copy
import json
import math
import random

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.enemy_instances import (
    ENEMY_INSTANCE_SCHEMA,
    action_target_instance,
    enemy_instance_vectors,
    instance_action_features,
    instance_global_features,
)
from sts2_ai.models.neural import (
    NEURAL_FORMAT, TACTICAL_INSTANCES_FORMAT, NeuralPolicyValueModel,
)
from sts2_ai.training.instance_combat import instance_forward
from test_tactical_state_v7 import frame

DIM = 128
HIDDEN = 4


def combat_frame() -> dict:
    state = frame()
    follower = state["combat"]["enemies"][0]
    follower.update(
        instance_id=1, enemy_id="proto.enemy.kin_follower",
        formation_position=0, last_move_id="quick_slash",
        move_id="boomerang", hp=30, block=4,
        intent_damage=3, intent_base_damage=2, intent_hits=2,
    )
    other = copy.deepcopy(follower)
    other.update(instance_id=3, formation_position=2, intent_damage=8)
    other["statuses"] = {"poison": 12}
    priest = copy.deepcopy(follower)
    priest.update(
        instance_id=2, enemy_id="proto.enemy.kin_priest",
        formation_position=1, hp=140, block=0, intent_damage=7,
        move_id="orb_of_frailty", last_move_id="priest_heal",
    )
    priest["statuses"] = {}
    state["combat"]["enemies"] = [follower, priest, other]
    state["act_one_boss_encounter_id"] = "proto.encounter.the_kin_boss"
    return state


def targeted(instance_id: int) -> LegalAction:
    return LegalAction(
        f"strike-{instance_id}", "play_card",
        json.dumps({"CardInstanceId": 101, "TargetEnemyId": instance_id}),
    )


def build_model() -> NeuralPolicyValueModel:
    state_weight = [[0.0] * DIM for _ in range(HIDDEN)]
    enemy_weight = [[0.0] * DIM for _ in range(HIDDEN)]
    enemy_weight[0][4] = 15.0  # Target-local displayed per-hit incoming damage
    enemy_weight[1][3] = 5.0   # Target-local visible position
    # Learnable species features are categorically embedded, not numeric IDs.
    species_probe = enemy_instance_vectors(combat_frame(), DIM)[2]
    new_species = enemy_instance_vectors(combat_frame(), DIM)[1]
    different = next(i for i in species_probe if i >= 16 and i not in new_species)
    enemy_weight[2][different] = 1.0
    model = NeuralPolicyValueModel.from_dict({
        "format": TACTICAL_INSTANCES_FORMAT,
        "dimension": DIM, "hidden": HIDDEN, "model_id": "v8-fixture",
        "state_weight": state_weight,
        "state_bias": [1.0] * HIDDEN,
        "action_weight": [[0.0] * DIM for _ in range(HIDDEN)],
        "action_bias": [0.0] * HIDDEN,
        "policy_weight": [1.0] * HIDDEN,
        "policy_bias": 0.0,
        "value_weight": [0.25] * HIDDEN,
        "value_bias": 0.0,
        "enemy_weight": enemy_weight,
        "enemy_bias": [0.0] * HIDDEN,
        "enemy_context_weight": [0.25] * HIDDEN,
        "enemy_target_weight": [1.0] * HIDDEN,
    })
    return model


def observation(state: dict) -> Observation:
    payload = json.dumps(state, sort_keys=True)
    return Observation("prototype-fair-v0", payload, payload)


def test_species_and_effects_are_owned_by_individual_instances() -> None:
    state = combat_frame()
    assert ENEMY_INSTANCE_SCHEMA.endswith("v8")
    instances = enemy_instance_vectors(state, DIM)
    assert set(instances) == {1, 2, 3}
    assert instances[1] != instances[3]
    assert instances[1] != instances[2]
    previous = copy.deepcopy(state)
    previous["combat"]["enemies"][2]["statuses"]["poison"] = 90
    new = enemy_instance_vectors(previous, DIM)
    assert new[1] == instances[1]
    assert new[2] == instances[2]
    assert new[3] != instances[3]
    assert instance_global_features(state, DIM) == instance_global_features(
        previous, DIM
    )


def test_target_pointer_changes_policy_scores_and_is_renumbering_equivariant() -> None:
    state = combat_frame()
    model = build_model()
    actions = (targeted(1), targeted(2), targeted(3))
    original = model.evaluate(observation(state), actions)
    assert original.action_logits[0] != original.action_logits[2]
    assert original.action_logits[1] != original.action_logits[2]

    renamed = copy.deepcopy(state)
    for enemy in renamed["combat"]["enemies"]:
        enemy["instance_id"] += 1000
    renamed["combat"]["enemies"].reverse()
    renamed_actions = tuple(targeted(i + 1000) for i in (3, 1, 2))
    changed = model.evaluate(observation(renamed), renamed_actions)
    for from_index, to_index in [(2, 0), (0, 1), (1, 2)]:
        assert changed.action_logits[to_index] == pytest.approx(
            original.action_logits[from_index], abs=1e-6,
        )
    assert changed.value == pytest.approx(original.value, abs=1e-6)

    with pytest.raises(ValueError, match="visible enemy instance"):
        model.evaluate(observation(state), [targeted(42)])
    with pytest.raises(ValueError, match="Duplicate enemy instance"):
        invalid = copy.deepcopy(state)
        invalid["combat"]["enemies"][2]["instance_id"] = 1
        model.evaluate(observation(invalid), actions)


def test_species_and_last_move_inform_potential_future_intents() -> None:
    state = combat_frame()
    original = enemy_instance_vectors(state, DIM)
    new = copy.deepcopy(state)
    new["combat"]["enemies"][0]["enemy_id"] = "proto.enemy.some_other_species"
    new["combat"]["enemies"][0]["last_move_id"] = "different_prior_move"
    changed = enemy_instance_vectors(new, DIM)
    assert changed[1] != original[1]
    assert changed[2] == original[2]
    assert changed[3] == original[3]


def test_card_identity_and_target_are_separate_model_inputs() -> None:
    state = combat_frame()
    assert instance_action_features(
        state, "play_card", targeted(1).payload_json, DIM
    ) == instance_action_features(
        state, "play_card", targeted(3).payload_json, DIM
    )
    identities = enemy_instance_vectors(state, DIM)
    assert action_target_instance(targeted(1).payload_json, identities) == 1
    assert action_target_instance(targeted(3).payload_json, identities) == 3


def test_pure_inference_equals_torch_training_forward() -> None:
    torch = pytest.importorskip("torch")
    state = combat_frame()
    model = build_model()
    actions = (targeted(1), targeted(3), targeted(2))
    params = {}
    for key in (
        "state_weight", "state_bias", "action_weight", "action_bias",
        "policy_weight", "policy_bias", "value_weight", "value_bias",
        "enemy_weight", "enemy_bias", "enemy_context_weight", "enemy_target_weight",
    ):
        params[key] = torch.tensor(getattr(model, key), dtype=torch.float32)
    target, logits = instance_forward(state, actions, params, DIM, torch)
    actual = model.evaluate(observation(state), actions)
    assert actual.value == pytest.approx(float(target), abs=1e-5)
    assert actual.action_logits == pytest.approx(
        tuple(float(v) for v in logits), abs=1e-5,
    )
    assert NeuralPolicyValueModel.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_v8_ppo_smoke_and_checkpoint_goal(tmp_path) -> None:
    pytest.importorskip("torch")
    from test_act1_episode_goal import NativeScriptedBackend, _boss
    from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
    from sts2_ai.training.phase_split_selfplay import train_phase_split

    class WithVisiblePositions(NativeScriptedBackend):
        def __init__(self) -> None:
            super().__init__()
            for frame in self.frames:
                if isinstance(frame.get("combat"), dict):
                    for i, enemy in enumerate(frame["combat"]["enemies"]):
                        enemy["formation_position"] = i
                        enemy["last_move_id"] = None

    options = dict(
        rounds=1, episodes_per_round=2, dimension=DIM, hidden=HIDDEN,
        max_decisions=8, workers=1, environment="native-overgrowth",
        seed=21, optimizer_method="ppo", ppo_epochs=1,
        ppo_batch_size=2, ppo_sample_limit=16,
        hp_monotonic_weight=0.0, combat_objective="hp_preservation",
        tactical_state_encoding="enemy_instances",
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        checkpoint=tmp_path / "v8.pt",
    )
    model, rows = train_phase_split(WithVisiblePositions(), **options)
    assert rows[0].wins == 2 and rows[0].optimization_steps > 0
    assert model.strategy.format_id == NEURAL_FORMAT
    assert model.combat.format_id == TACTICAL_INSTANCES_FORMAT
    assert model.combat.enemy_weight is not None

    resumed, history = train_phase_split(
        WithVisiblePositions(), **{**options, "resume": True, "rounds": 2},
    )
    assert len(history) == 2
    assert history[0] == rows[0]
    assert resumed.combat.format_id == TACTICAL_INSTANCES_FORMAT
