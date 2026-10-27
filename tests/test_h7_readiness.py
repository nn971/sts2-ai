"""H7 throughput reports must preserve complete source and context provenance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_ai.evaluation.h7_readiness import REPORT_SCHEMA, make_readiness_report
from sts2_ai.nature.h5_curriculum import source_bank


def _cohort(seeds: tuple[str, ...]):
    from sts2_ai.nature.h5_curriculum import cell_for_source

    return {
        "episodes": tuple(
            SimpleNamespace(
                source_seed=seed,
                scenario_id=cell_for_source(seed).scenario,
                deck_preset_id=cell_for_source(seed).deck,
                draw_mode="reference",
                outcome="victory" if i % 2 else "defeat",
                terminal_replay_verified=True,
                decisions=(
                    SimpleNamespace(phase="player"),
                    SimpleNamespace(phase="draw_card"),
                    SimpleNamespace(phase="player_choice"),
                ),
            )
            for i, seed in enumerate(seeds)
        ),
        "elapsed_seconds": 8.0,
        "process_count": 8,
    }


def test_report_has_full_curriculum_and_measured_decision_throughput():
    seeds = source_bank("train", 16, study_seed=49)
    report = make_readiness_report(
        _cohort(seeds), seeds, workers=15,
        model_sha256="a" * 64, parameter_count=14_659_972,
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["train_cells"] == 16
    assert all(count == 1 for count in report["by_cell"].values())
    assert report["combats"] == 16
    assert report["player_decisions"] == 32
    assert report["player_decisions_per_second"] == 4
    assert report["actor_inference_device"] == "cpu"
    assert report["worker_processes_observed"] == 8


def test_report_rejects_source_loss_tampering_or_incomplete_curriculum():
    seeds = source_bank("train", 16, study_seed=49)
    cohort = _cohort(seeds)
    with pytest.raises(ValueError, match="unverified"):
        make_readiness_report(
            {**cohort, "episodes": cohort["episodes"][::-1]}, seeds,
            workers=15, model_sha256="a" * 64, parameter_count=14_659_972,
        )
    with pytest.raises(ValueError, match="source-complete"):
        make_readiness_report(
            cohort, seeds[:-1], workers=15,
            model_sha256="a" * 64, parameter_count=14_659_972,
        )
    with pytest.raises(ValueError, match="source-complete"):
        make_readiness_report(
            cohort, seeds, workers=4,
            model_sha256="a" * 64, parameter_count=14_659_972,
        )
    damaged = list(cohort["episodes"])
    damaged[0] = SimpleNamespace(**{
        **vars(damaged[0]), "terminal_replay_verified": False,
    })
    with pytest.raises(ValueError, match="unverified"):
        make_readiness_report(
            {**cohort, "episodes": tuple(damaged)}, seeds,
            workers=15, model_sha256="a" * 64, parameter_count=14_659_972,
        )
