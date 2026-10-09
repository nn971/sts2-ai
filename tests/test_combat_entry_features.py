"""Public combat entry features and versioned predictor compatibility."""
from __future__ import annotations

import json

import pytest

from sts2_ai.training.combat_distributional_fit import (
    fit_distributional_predictor,
)
from sts2_ai.training.combat_distributional_predictor import (
    DISTRIBUTIONAL_FORMAT,
    DISTRIBUTIONAL_FORMAT_V2,
    HP_BINS,
    DistributionalCombatOutcomePredictor,
)
from sts2_ai.training.combat_entry_features import (
    FEATURE_SCHEMA,
    NUMERIC_FEATURES,
    combat_entry_features,
)
from sts2_ai.training.combat_predictor import OutcomeExample


def state(hp: int = 60) -> dict:
    return {
        "hp": hp, "max_hp": 70, "floor": 5, "gold": 100,
        "act": 1, "act_one_boss_encounter_id": "proto.encounter.vantom_boss",
        "map": [
            {"node_id": str(i), "next_node_ids": [str(i + 1)]}
            for i in range(70)
        ],
        "deck": [{"card_id": "silent.strike"}, {"card_id": "silent.defend"}],
        "relics": [{"relic_id": "silent_ring"}],
        "potions": [{"potion_id": "fire", "slot": 0}],
        "combat": {
            "energy": 3, "turn": 1, "player_block": 0,
            "draw_pile_count": 7,
            "enemies": [{"enemy_id": "slime", "hp": 20, "block": 0,
                         "move_id": "attack"}],
            "hand": [{"card_id": "silent.strike"}],
        },
    }


def model(feature_schema: str = "legacy-hashed-v1"):
    return DistributionalCombatOutcomePredictor(
        dimension=64, hidden=4,
        state_weight=[[0.0] * 64 for _ in range(4)],
        state_bias=[0.0] * 4,
        survival_weight=[0.0] * 4, survival_bias=0.0,
        hp_bin_weight=[[0.0] * 4 for _ in range(HP_BINS)],
        hp_bin_bias=[0.0] * HP_BINS,
        model_id="version-check", feature_schema=feature_schema,
    )


def test_encoder_keeps_critical_hp_coordinates_and_excludes_map_ids() -> None:
    frame = state()
    features = combat_entry_features(frame, 256)
    assert features[1] == pytest.approx(60 / 70)
    assert features[6] == pytest.approx(20 / 200)
    assert len(features) < 150  # no one-feature-per-map-node explosion
    changed_map = state()
    changed_map["map"] = [{"node_id": "totally-unrelated-map-node"}]
    assert combat_entry_features(changed_map, 256) == features
    low_hp = state(30)
    assert combat_entry_features(low_hp, 256)[1] == pytest.approx(30 / 70)
    assert combat_entry_features(low_hp, 256) != features
    different_enemy = state()
    different_enemy["combat"]["enemies"][0]["enemy_id"] = "other"
    assert combat_entry_features(different_enemy, 256) != features
    assert NUMERIC_FEATURES == 24
    with pytest.raises(ValueError):
        combat_entry_features(frame, 24)


def test_potion_identity_is_preserved_and_no_instance_id_hashed() -> None:
    original = state()
    with_potion = state()
    with_potion["potions"][0]["potion_id"] = "stronger"
    assert combat_entry_features(original, 256) != combat_entry_features(with_potion, 256)
    with_instances = state()
    with_instances["combat"]["enemies"][0]["instance_id"] = 1234
    with_instances["potions"][0]["slot"] = 42
    assert combat_entry_features(original, 256) == combat_entry_features(with_instances, 256)


def test_both_format_versions_roundtrip_and_reject_schema_confusion() -> None:
    for schema, expected_format in (
        ("legacy-hashed-v1", DISTRIBUTIONAL_FORMAT),
        (FEATURE_SCHEMA, DISTRIBUTIONAL_FORMAT_V2),
    ):
        obj = model(schema)
        raw = obj.to_dict()
        assert raw["format"] == expected_format
        loaded = DistributionalCombatOutcomePredictor.from_dict(raw)
        assert loaded.to_dict() == raw
        assert sum(loaded.predict(json.dumps(state())).joint_hp_probabilities) == pytest.approx(1)
    old = model().to_dict()
    del old["feature_schema"]
    assert DistributionalCombatOutcomePredictor.from_dict(old).feature_schema == "legacy-hashed-v1"
    incorrect = model(FEATURE_SCHEMA).to_dict()
    incorrect["feature_schema"] = "legacy-hashed-v1"
    with pytest.raises(ValueError, match="feature schema"):
        DistributionalCombatOutcomePredictor.from_dict(incorrect)


def test_structured_encoder_trains_from_public_samples() -> None:
    pytest.importorskip("torch")
    samples = [
        OutcomeExample(
            f"seed-{i}", json.dumps(state(65 if i % 2 else 12)),
            float(i % 2), 0.75 if i % 2 else 0.0,
        )
        for i in range(8)
    ]
    trained = fit_distributional_predictor(
        samples, dimension=64, hidden=8, epochs=3,
        feature_schema=FEATURE_SCHEMA,
    )
    assert trained.feature_schema == FEATURE_SCHEMA
    assert trained.to_dict()["format"] == DISTRIBUTIONAL_FORMAT_V2
