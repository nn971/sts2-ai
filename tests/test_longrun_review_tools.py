"""Deterministic review summaries and strict fixed-baseline cache contracts."""
from __future__ import annotations

import json

import pytest

from sts2_ai.evaluation.baseline_cache import (
    baseline_identity,
    load_cache,
    save_cache,
)
from tools.replay_review_runs import markdown_review, public_trace_entry


def test_baseline_cache_reuses_only_exact_model_emulator_and_seed_set(tmp_path) -> None:
    model = tmp_path / "model.json"
    model.write_text('{"format":"fixture-v1"}')
    identity = baseline_identity(
        warm_start=model, emulator_revision="emulator-v1",
        seed_prefix="heldout", count=2,
        episode_goal="native-act1-boss-v1", max_decisions=4096,
        environment="native-overgrowth", policy_id="prototype-fair-v0",
    )
    rows = [
        {
            "seed": f"heldout-{i}", "episode_goal_version": "native-act1-boss-v1",
            "censored": False, "act1_cleared": i == 0,
        }
        for i in range(2)
    ]
    cache = tmp_path / "cache.json"
    assert load_cache(cache, identity) is None
    save_cache(cache, identity, rows)
    assert load_cache(cache, identity) == rows

    model.write_text('{"format":"fixture-v2"}')
    modified = baseline_identity(
        warm_start=model, emulator_revision="emulator-v1",
        seed_prefix="heldout", count=2,
        episode_goal="native-act1-boss-v1", max_decisions=4096,
        environment="native-overgrowth", policy_id="prototype-fair-v0",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_cache(cache, modified)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_cache(cache, {**identity, "emulator_revision": "emulator-v2"})
    with pytest.raises(ValueError, match="seed mismatch"):
        save_cache(tmp_path / "bad.json", identity, rows[::-1])


def test_trace_records_only_public_fields_and_target_instance() -> None:
    state = {
        "act": 1, "floor": 16, "phase": 5, "hp": 17, "max_hp": 70,
        "gold": 113, "hidden_rng_seed": "SHOULD_NOT_LEAK",
        "combat": {
            "turn": 4, "energy": 2,
            "hand": [{"instance_id": 1, "card_id": "proto.silent.snakebite"}],
            "enemies": [
                {
                    "instance_id": 4, "enemy_id": "proto.enemy.kin_priest",
                    "formation_position": 1, "hp": 20, "block": 0,
                    "move_id": "dark_prayer", "last_move_id": "weakness",
                    "intent_damage": 0, "intent_hits": 0,
                    "statuses": {"proto.status.poison": 5},
                    "powers": [],
                    "hidden_future_intent": "SHOULD_NOT_LEAK",
                }
            ],
        },
    }
    entry = public_trace_entry(
        json.dumps(state), "play_card",
        json.dumps({"CardInstanceId": 1, "TargetEnemyId": 4}),
        decision_index=6, legal_count=13,
    )
    assert entry["action"]["payload"]["TargetEnemyId"] == 4
    assert entry["enemies"][0]["enemy_id"] == "proto.enemy.kin_priest"
    assert "hidden_future_intent" not in entry["enemies"][0]
    assert "SHOULD_NOT_LEAK" not in json.dumps(entry)

    document = markdown_review(
        "heldout-44", "Kin recovery", {"round40": [entry]},
        {"round40": {
            "outcome": "victory", "act1_cleared": True, "terminal_floor": 16,
            "decisions": 7, "boss_progress": {
                "encounter_id": "proto.encounter.the_kin_boss",
                "remaining_hp": 0,
            },
        }},
    )
    assert "Kin recovery" in document
    assert "kin_priest" in document
    assert "TargetEnemyId" in document
