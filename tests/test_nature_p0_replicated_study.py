"""Replicated Nature P0 research: independent seeds and two-axis uncertainty."""
from __future__ import annotations

import copy

import pytest

from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.replicated_study import (
    REPLICATED_SCHEMA,
    TRAIN_SCHEMA,
    analyze_replicated_study,
)
from sts2_ai.nature.research_p0 import EVAL_SCHEMA


def history(seed: int, arm: str, *, steps: int = 12):
    config = {
        "schema": TRAIN_SCHEMA, "arm": arm, "seed": seed,
        "nature_update": arm == "nature",
        "source_law": "UNVERIFIED",
        "source_fixtures_sha256": "t" * 64,
        "emulator_revision": "pinned-csharp-source",
        "emulator_dll_sha256": "d" * 64,
        "emulator_bindings_sha256": "b" * 64,
        "initial_model_sha256": "i" * 64,
        "episodes_per_round": 4, "scope": "silent-basic-v1",
        "temperature": 0.85, "learning_rate": 0.0003,
        "epochs": 3, "sample_limit": 2048, "batch_size": 32,
    }
    return {
        "schema": TRAIN_SCHEMA, "config": config,
        "metrics": [{
            "round": 0, "training_arm": arm,
            "completed": 4, "source_run_count": 4,
            "sample_count": 44, "player_ppo_steps": steps,
            "nature_update": {"nature_revision": 1} if arm == "nature" else None,
            "rollout_seconds": 2.0, "optimizer_seconds": 1.1 if arm == "nature" else 1.,
            "worker_processes_observed": 4,
        }],
    }


def replica(seed: int, *, steps_nature: int = 12):
    n_sha = f"{seed:064x}"
    o_sha = f"{seed + 1000:064x}"
    rows = {"nature_trained": [], "ordinary_trained": []}
    for j, (name, source) in enumerate(
        (("Overgrowth", "holdout-a"), ("Underdocks", "holdout-b"))
    ):
        for label in rows:
            # Different improvements across training seeds and source runs:
            # preserve full opening equality and source pairing.
            if label == "nature_trained":
                score = 0.7 + 0.01 * seed + j * 0.02
            else:
                score = 0.6 - j * 0.01
            rows[label].append({
                "arm": "ordinary", "region": name, "source_seed": source,
                "fixture_id": f"eval-{j}",
                "source_opening_fingerprint": f"source-opening-{j}",
                "initial_public_fingerprint": f"public-opening-{j}",
                "score": score, "victory": True,
            })
    experiment = {
        "schema": EVAL_SCHEMA, "reference_law": "UNVERIFIED",
        "train_sha256": "t" * 64, "holdout_sha256": "h" * 64,
        "holdout_source_group_sha256": "g" * 64,
        "emulator_revision": "pinned-csharp-source",
        "emulator_dll_sha256": "d" * 64,
        "emulator_bindings_sha256": "b" * 64,
        "scope": "silent-basic-v1", "temperature": 0.85,
        "model_sha256": {
            "nature_trained": n_sha,
            "ordinary_trained": o_sha,
        },
        "actor_seed": 9000 + seed,
    }
    experiment["fingerprint"] = canonical_sha256(experiment)
    return {
        "nature_history": history(seed, "nature", steps=steps_nature),
        "ordinary_history": history(seed, "ordinary"),
        "nature_model_sha256": n_sha,
        "ordinary_model_sha256": o_sha,
        "holdout": {
            "schema": EVAL_SCHEMA, "decision_grade": False,
            "research_verdict": "NOT_ESTABLISHED",
            "source_chance_law": "UNVERIFIED",
            "source_run_count": 2,
            "ordinary_score_delta": 0.115 + 0.01 * seed,
            "experiment": experiment, "rows": rows,
        },
    }


def analyze(items):
    return analyze_replicated_study(
        items, bootstrap_seed=5, bootstrap_replicates=200
    )


def reseal_experiment(item):
    exp = item["holdout"]["experiment"]
    exp["fingerprint"] = canonical_sha256({
        key: value for key, value in exp.items() if key != "fingerprint"
    })


def test_two_axis_bootstrap_multiple_training_restarts_and_budget_ledger():
    items = [replica(x) for x in (1, 3, 5, 7, 9)]
    result = analyze(items)
    assert result["schema"] == REPLICATED_SCHEMA
    assert result["replica_count"] == 5
    assert result["heldout_source_run_count"] == 2
    assert result["research_replication_floor_met"] is True
    assert result["between_training_seed_sd"] > 0
    assert result["training_seed_uncertainty"] == "TWO_AXIS_BOOTSTRAP"
    assert result["ordinary_mean_score_delta"] == pytest.approx(0.165)
    assert result["paired_training_seed_and_source_bootstrap_95pct"][0] <= 0.165
    assert result["paired_training_seed_and_source_bootstrap_95pct"][1] >= 0.165
    assert result["budget_status"] == "EQUAL_PLAYER_OPTIMIZER_STEPS_NOT_TOTAL_COMPUTE"
    assert result["total_compute_matching"] == "NOT_ESTABLISHED"
    assert result["decision_grade"] is False
    assert result["research_verdict"] == "NOT_ESTABLISHED"
    assert result["source_chance_law"] == "UNVERIFIED"
    assert analyze(items) == result


def test_unequal_actual_optimizer_steps_are_audited_not_hidden():
    result = analyze([replica(1), replica(2, steps_nature=9)])
    assert result["budget_status"] == "UNEQUAL_PLAYER_OPTIMIZER_STEPS_AND_TOTAL_COMPUTE"
    assert result["replicas"][1]["equal_player_optimizer_steps"] is False
    assert result["research_replication_floor_met"] is False


def test_cross_retraining_source_or_public_opening_drift_rejected():
    items = [replica(1), replica(2)]
    items[1]["holdout"]["rows"]["nature_trained"][0][
        "source_opening_fingerprint"
    ] = "different"
    with pytest.raises(ValueError, match="opening"):
        analyze(items)


def test_duplicate_training_or_actor_seed_is_rejected():
    with pytest.raises(ValueError, match="duplicated training"):
        analyze([replica(1), replica(1)])
    items = [replica(1), replica(2)]
    items[1]["holdout"]["experiment"]["actor_seed"] = 9001
    reseal_experiment(items[1])
    with pytest.raises(ValueError, match="actor RNG seed"):
        analyze(items)


def test_mismatched_checkpoint_config_and_source_partitions_rejected():
    items = [replica(1), replica(2)]
    mismatch = copy.deepcopy(items)
    mismatch[1]["ordinary_history"]["config"]["initial_model_sha256"] = "q" * 64
    with pytest.raises(ValueError, match="unequal starting checkpoint"):
        analyze(mismatch)
    mismatch = copy.deepcopy(items)
    mismatch[1]["holdout"]["experiment"]["train_sha256"] = "x" * 64
    reseal_experiment(mismatch[1])
    with pytest.raises(ValueError, match="disagrees with training"):
        analyze(mismatch)
    mismatch = copy.deepcopy(items)
    mismatch[1]["holdout"]["experiment"]["holdout_sha256"] = "x" * 64
    reseal_experiment(mismatch[1])
    with pytest.raises(ValueError, match="Held-out source groups"):
        analyze(mismatch)


def test_reported_score_or_fingerprint_forgery_rejected():
    altered = replica(2)
    altered["holdout"]["ordinary_score_delta"] += 0.1
    with pytest.raises(ValueError, match="Recomputed holdout score"):
        analyze([altered])
    altered = replica(2)
    altered["holdout"]["experiment"]["fingerprint"] = "forged"
    with pytest.raises(ValueError, match="fingerprint"):
        analyze([altered])


def test_single_training_replica_explicitly_has_no_seed_uncertainty():
    result = analyze([replica(2)])
    assert result["training_seed_uncertainty"] == "NOT_ESTIMATED"
    assert result["between_training_seed_sd"] is None
    assert result["decision_grade"] is False
