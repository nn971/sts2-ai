"""Formal P2 closure gate accepts full-Act evidence and rejects short smokes."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tools.check_p2_h22_closure import assess


def accepted_report() -> dict:
    seeds = [f"independent-{i}" for i in range(16)]
    walls = (120.0, 90.0, 89.0, 121.0)
    return {
        "schema": "sts2-p2-actor-abba-acceptance-v1",
        "experiment": "h21-collator",
        "parity": {
            "passed": True, "within_run_replay": True,
            "cross_run_public_teacher": True,
            "exact_search_and_episode_work": True,
            "seed_count": 16,
        },
        "full_act_gate": {
            "passed": True, "require_complete": True,
            "min_seed_count": 16, "require_cuda": True,
            "require_search_roots": True,
        },
        "workload": {
            "seeds": seeds, "worker_count": 15,
            "max_decisions": 4096, "search_simulations": 16,
            "search_depth": 8, "max_searches": 2,
            "search_transitions_per_run": 3200,
            "real_transitions_per_run": 2000,
            "ascension": 0, "map_profile": "generated",
            "censored_episode_executions_per_run": 0,
            "completed_episode_executions_per_run": 32,
        },
        "source": {
            "ai_sha": "ai-revision", "emulator_sha": "emulator-revision",
            "checkpoint_sha256": "checkpoint-hash",
            "emulator_cli": {"path": "/emulator/cli.dll", "sha256": "cli-hash"},
        },
        "public_schema_id": "prototype-ai-v1",
        "actor_deterministic_cuda": True,
        "actor_h21_collators": [
            "reference", "packed", "packed", "reference",
        ],
        "configurations": {
            "A_H21_reference": {
                "h21_collator": "reference", "actor_max_batch": 4,
                "fixed_shape_batch_size": 4, "rollout_transport": "fused",
                "p2_prefix_cache_handles": 0, "actor_profile": False,
            },
            "B_H21_packed": {
                "h21_collator": "packed", "actor_max_batch": 4,
                "fixed_shape_batch_size": 4, "rollout_transport": "fused",
                "p2_prefix_cache_handles": 0, "actor_profile": False,
            },
        },
        "cohorts": [
            {
                "arm": name, "wall_seconds": wall, "completed_episodes": 32,
                "censored_episodes": 0, "search_transitions": 3200,
            }
            for name, wall in zip(("A1", "B1", "B2", "A2"), walls, strict=True)
        ],
        "ratio_of_total_baseline_to_candidate_wall": 241 / 179,
        "order_pair_speedup_A1_over_B1": 120 / 90,
        "order_pair_speedup_A2_over_B2": 121 / 89,
        "candidate_faster_in_both_pairs": True,
    }


def test_p2_acceptance_requires_full_act_exact_work_and_measured_gain():
    original = accepted_report()
    frozen = deepcopy(original)
    result = assess(original, checkpoint_sha256="checkpoint-hash")
    assert original == frozen
    assert result["p2_5_gate_passed"] is True
    assert result["seed_count"] == 16
    assert result["full_act_episodes_per_arm"] == 32
    assert result["packed_collator_decision"] == "eligible_for_default_promotion"
    assert result["wall_speedup"] == pytest.approx(241 / 179)
    assert result["search_depth"] == 8


@pytest.mark.parametrize(("modify", "match"), [
    (lambda r: r["full_act_gate"].update({"require_complete": False}),
     "strict CUDA/full-Act"),
    (lambda r: r["workload"].update({"seeds": ["short"]}),
     "workload"),
    (lambda r: r["workload"].update({"worker_count": 4}),
     "workload"),
    (lambda r: r["workload"].update({"censored_episode_executions_per_run": 2}),
     "two complete"),
    (lambda r: r["parity"].update({"cross_run_public_teacher": False}),
     "exact parity"),
    (lambda r: r["parity"].update({"exact_search_and_episode_work": False}),
     "exact parity"),
    (lambda r: r.update({"actor_deterministic_cuda": False}),
     "Wrong H22"),
    (lambda r: r["configurations"]["B_H21_packed"].update({
        "actor_max_batch": 8,
    }), "Unmatched"),
    (lambda r: r["cohorts"][1].update({"censored_episodes": 1}),
     "incomplete"),
    (lambda r: r.update({
        "ratio_of_total_baseline_to_candidate_wall": 999,
    }), "speedup disagrees"),
])
def test_p2_closure_rejects_insufficient_or_mismatched_evidence(modify, match):
    raw = accepted_report()
    modify(raw)
    with pytest.raises(ValueError, match=match):
        assess(raw)


def test_p2_closure_rejects_checkpoint_identity_drift():
    with pytest.raises(ValueError, match="different H22 checkpoint"):
        assess(accepted_report(), checkpoint_sha256="other")


def test_p2_closure_allows_measured_negative_result_without_promoting():
    report = accepted_report()
    walls = [120.0, 119.0, 118.0, 121.0]
    for row, wall in zip(report["cohorts"], walls, strict=True):
        row["wall_seconds"] = wall
    report["ratio_of_total_baseline_to_candidate_wall"] = 241 / 237
    report["order_pair_speedup_A1_over_B1"] = 120 / 119
    report["order_pair_speedup_A2_over_B2"] = 121 / 118
    report["candidate_faster_in_both_pairs"] = True
    assert assess(report)["packed_collator_decision"] == (
        "keep_opt_in_measured_benefit_insufficient"
    )
    report["cohorts"][2]["wall_seconds"] = 130.0
    report["ratio_of_total_baseline_to_candidate_wall"] = 241 / 249
    report["order_pair_speedup_A2_over_B2"] = 121 / 130
    report["candidate_faster_in_both_pairs"] = False
    assert assess(report)["packed_collator_decision"] == (
        "keep_opt_in_measured_benefit_insufficient"
    )
