"""Real Nature player PPO probability/observation and freshness contract tests."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

import pytest

from sts2_ai.emulator.semantic_p0_frozen_player import (
    FrozenP0Player,
    FrozenPolicyInferenceError,
)
from sts2_ai.emulator.semantic_p0_public_projection import (
    project_public_p0_observation,
)
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.training.nature_real_combat_ppo import (
    RealPlayerPpoError,
    validate_recorded_action,
)


@dataclass(frozen=True)
class Estimate:
    action_logits: tuple[float, ...]
    value: float


class SimplePublicPolicy:
    model_id = "source-public-policy-v1"

    def evaluate(self, observation, actions):
        assert observation.policy_id == "prototype-fair-v0"
        assert len(actions) == 2
        return Estimate((2.0, 0.5), 0.3)


OBS = {
    "hp": 58, "max_hp": 66,
    "combat": {
        "hand": [{"instance_id": 3, "card_id": "proto.silent.strike"}],
        "enemies": [],
    },
}
MENU = [
    {"actionId": "strike:3", "kind": "play_card",
     "payloadJson": '{"CardInstanceId":3}'},
    {"actionId": "end_turn", "kind": "end_turn", "payloadJson": "{}"},
]


def sample():
    actor = FrozenP0Player(
        SimplePublicPolicy(), actor_seed=17, greedy=False,
        temperature=0.85, action_scope="silent-basic-v1",
        capture_training=True,
    )
    chosen = actor.choose(OBS, MENU)
    assert actor.decisions == len(actor.training_actions) == 1
    return actor.training_actions[0], chosen


def test_sampled_behavior_exact_probability_and_public_menu():
    raw, selected = sample()
    decision, logprob, value = validate_recorded_action(
        raw, expected_model_id="source-public-policy-v1",
        expected_temperature=0.85, expected_scope="silent-basic-v1",
    )
    assert decision.legal_actions[decision.chosen_index].action_id == selected
    assert len(decision.legal_actions) == 2
    weights = [math.exp(x / 0.85) for x in (2.0, 0.5)]
    expected = math.log(weights[decision.chosen_index] / sum(weights))
    assert logprob == pytest.approx(expected)
    assert value == pytest.approx(0.3)
    assert "pending" not in decision.observation.payload_json
    assert "semanticDeck" not in decision.observation.payload_json


@pytest.mark.parametrize("corrupt,match", [
    ({"behavior_model_id": "stale"}, "Stale"),
    ({"temperature": 1.2}, "temperature"),
    ({"action_scope": "all-legal-v1"}, "scope"),
    ({"selected_id": "not-issued"}, "identity drift"),
    ({"chosen_index": 111}, "Chosen action index"),
    ({"behavior_logprob": float("nan")}, "nonfinite"),
    ({"behavior_logprob": 2.0}, "exceeds zero"),
])
def test_wrong_behavior_or_forged_action_is_rejected(corrupt, match):
    raw, _ = sample()
    with pytest.raises(RealPlayerPpoError, match=match):
        validate_recorded_action(
            {**raw, **corrupt}, expected_model_id="source-public-policy-v1",
            expected_temperature=0.85, expected_scope="silent-basic-v1",
        )


def test_public_observation_tampering_rejected():
    raw, _ = sample()
    tampered = copy.deepcopy(raw)
    tampered["observation"]["payload_json"] = '{"pending":{"secret":42}}'
    with pytest.raises(RealPlayerPpoError, match="private state"):
        validate_recorded_action(
            tampered, expected_model_id="source-public-policy-v1",
            expected_temperature=0.85, expected_scope="silent-basic-v1",
        )


@pytest.mark.parametrize("mutation,match", [
    ("source_digest", "provenance digest"),
    ("source_menu", "provenance digest"),
    ("restricted_menu", "differs from issued source menu"),
    ("public_observation", "differs from issued source menu"),
    ("fully_rehashed_source", "differs from issued source menu"),
])
def test_complete_source_legal_menu_and_observation_attestation(mutation, match):
    from sts2_ai.nature.protocol import canonical_sha256

    raw, _ = sample()
    corrupted = copy.deepcopy(raw)
    if mutation == "source_digest":
        corrupted["source_support_identity"] = "incorrect"
    elif mutation == "source_menu":
        # Change an *unchosen* source action, preserving the chosen ID.
        corrupted["source_legal_actions"][1]["kind"] = "use_potion"
    elif mutation == "restricted_menu":
        corrupted["legal_actions"][1]["kind"] = "play_card"
    elif mutation == "public_observation":
        visible = json.loads(corrupted["observation"]["payload_json"])
        visible["hp"] += 1
        corrupted["observation"]["payload_json"] = json.dumps(
            visible, sort_keys=True, separators=(",", ":")
        )
        corrupted["observation"]["observation_hash"] = canonical_sha256(visible)
    elif mutation == "fully_rehashed_source":
        # A self-consistently rehashed forged source is still compared
        # with the separately recorded projected public observation.
        corrupted["source_legal_actions"][1]["kind"] = "use_potion"
        corrupted["source_support_identity"] = canonical_sha256([
            corrupted["source_observation"], corrupted["source_legal_actions"]
        ])
    with pytest.raises(RealPlayerPpoError, match=match):
        validate_recorded_action(
            corrupted, expected_model_id="source-public-policy-v1",
            expected_temperature=0.85, expected_scope="silent-basic-v1",
        )


def test_greedy_rollout_not_used_as_on_policy_ppo():
    actor = FrozenP0Player(
        SimplePublicPolicy(), actor_seed=17, greedy=True,
        action_scope="silent-basic-v1", capture_training=True,
    )
    with pytest.raises(FrozenPolicyInferenceError, match="stochastic"):
        actor.choose(OBS, MENU)


def test_source_action_scope_masks_unimplemented_card():
    actor = FrozenP0Player(
        SimplePublicPolicy(), actor_seed=19, greedy=False,
        action_scope="silent-basic-v1", capture_training=True,
    )
    menu = MENU + [
        {"actionId": "survivor:7", "kind": "play_card",
         "payloadJson": '{"CardInstanceId":7}'},
    ]
    chosen = actor.choose(OBS, menu)
    assert chosen in {"strike:3", "end_turn"}
    assert len(actor.training_actions[0]["legal_actions"]) == 2


def test_public_count_only_pile_projection_has_no_fabricated_cards():
    source = {
        "hp": 58, "max_hp": 66,
        "combat": {
            "hand": [{"instance_id": 3, "card_id": "proto.silent.strike"}],
            "draw_pile": 7, "draw_pile_count": 7,
            "discard_pile": 3, "exhaust_pile": 0, "play_pile": 0,
            "enemies": [{
                "instance_id": 2, "enemy_id": "proto.enemy.test",
                "hp": 10, "statuses": {},
            }],
        },
    }
    projected = project_public_p0_observation(source)
    assert source["combat"]["draw_pile"] == 7  # Source remains untouched.
    assert projected["combat"]["draw_pile"] == []
    assert projected["combat"]["draw_pile_count"] == 7
    assert projected["combat"]["draw_pile_opaque"] is True
    assert projected["combat"]["discard_pile_count"] == 3
    assert "combat.draw_pile" in projected["p0_opaque_public_fields"]
    objects = extract_combat_objects(projected)
    global_object = next(x for x in objects.tokens if x.kind == "global")
    global_features = dict(global_object.numeric)
    assert global_features["draw_pile_count"] == 7
    assert global_features["discard_pile_count"] == 3
    assert global_features["exhaust_pile_count"] == 0
    assert len([x for x in objects.tokens if x.kind == "card"]) == 1
    assert any(
        name == "p0_opaque:combat.draw_pile"
        for x in objects.tokens if x.kind == "global"
        for name in x.categorical
    )
    actor = FrozenP0Player(
        SimplePublicPolicy(), actor_seed=2, greedy=False,
        action_scope="silent-basic-v1", capture_training=True,
    )
    chosen = actor.choose(source, MENU)
    assert chosen in {"strike:3", "end_turn"}
    assert "p0_public_projection" in actor.training_actions[0][
        "observation"
    ]["payload_json"]


def test_count_only_public_pile_conflict_fails_closed():
    fair = {
        "hp": 20, "max_hp": 66,
        "combat": {
            "hand": [], "enemies": [],
            "draw_pile": 4, "draw_pile_count": 4,
            "discard_pile": 3, "discard_pile_count": 2,
        },
    }
    with pytest.raises(ValueError, match="Inconsistent source P0 count"):
        project_public_p0_observation(fair)


def test_legacy_wrong_card_count_still_rejected_and_opaque_guarded():
    legacy = {
        "hp": 22, "combat": {
            "hand": [], "draw_pile": [], "discard_pile": [],
            "exhaust_pile": [], "play_pile": [], "choice_pool": [],
            "draw_pile_count": 2, "relic_counters": [],
            "player_powers": [], "enemies": [],
        },
        "relics": [], "potions": [],
    }
    with pytest.raises(ValueError, match="does not match"):
        extract_combat_objects(legacy)
    legacy["combat"]["draw_pile_opaque"] = True
    with pytest.raises(ValueError, match="Invalid opaque"):
        extract_combat_objects(legacy)
