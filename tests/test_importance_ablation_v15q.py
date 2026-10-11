"""Strict v15q same-target ablation analysis and preflight tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_combat_diagnostics_v15p import fake_report

from sts2_ai.evaluation.ablation_analysis import compare_ablation, render_markdown


def _training(mix: float):
    return {
        "architecture": "relational",
        "base_model": "base.json",
        "vocabulary": "vocabulary.json",
        "emulator_revision": "pinned",
        "rounds_completed": 2,
        "training_settings": {
            "snapshot_sampling": "importance",
            "snapshot_priority_mix": mix,
            "snapshot_target_balance": 0.5,
            "snapshot_corpus": "train.jsonl",
            "diagnose_ppo": True,
            "seed": 19,
            "workers": 2,
            "episodes_per_round": 2,
        },
        "rows": [
            {
                "round_index": i, "rollout_source": "combat_snapshots",
                "played": 2, "wins": 1, "censored": 0,
                "resolved_combats": 2, "optimizer_steps": 3,
                "combat_decisions": 6, "ppo_samples_used": 6,
                "rollout_seconds": 1.2, "optimizer_seconds": 0.8,
                "mean_preclip_grad_norm": 0.6,
                "gradient_clip_fraction": 0.2,
                "raw_advantage_std": 1.5,
                "decision_weight_ess_fraction": 0.8,
                "importance_ess": 1.9,
                "importance_weight_max": 1.2,
                "mean_combat_target": 0.5,
            }
            for i in range(2)
        ],
    }


def _reports():
    return {
        "target_train": _training(0.0),
        "priority_train": _training(0.7),
        "target_eval": fake_report(0, (7, 8, 6, 7)),
        "priority_eval": fake_report(1, (9, 9, 8, 8)),
        "manifest": {
            "arms": {"target": 0.0, "priority": 0.7},
            "common_settings": {
                "rounds": 2, "episodes": 2, "target_balance": 0.5
            },
        },
    }


def test_paired_ablation_comparison_uses_same_source_clusters():
    result = compare_ablation(**_reports(), samples=100)
    assert result["scenarios"] == 4
    assert result["source_runs"] == 2
    pair = result["evaluation"]["priority_vs_target"]
    assert pair["delta_exit_hp"] == pytest.approx(1.5)
    assert pair["delta_win_rate"] == 0
    assert pair["cluster_95pct_delta_hp"] is not None
    assert result["evaluation"]["by_tier"]["elite"]["entries"] == 2
    assert result["training"]["priority"]["played"] == 4
    assert "Optimizer steps" in render_markdown(result)


@pytest.mark.parametrize("alter, message", [
    ("other_target", "requires proposal mixing"),
    ("different_settings", "mismatched frozen training settings"),
    ("different_rounds", "Unequal training rounds"),
    ("different_budget", "Unequal training combat budget"),
    ("censored", "censored"),
    ("no_ppo_diagnostics", "missing --diagnose-ppo"),
    ("not_paired", "Baseline results differ"),
    ("different_corpus", "different corpus"),
    ("different_manifest", "target distribution mismatch"),
])
def test_ablation_fails_closed_on_mismatch(alter, message):
    reports = _reports()
    if alter == "other_target":
        reports["priority_train"]["training_settings"]["snapshot_priority_mix"] = 0.3
    elif alter == "different_settings":
        reports["priority_train"]["training_settings"]["seed"] = 20
    elif alter == "different_rounds":
        reports["priority_train"]["rows"].pop()
        reports["priority_train"]["rounds_completed"] -= 1
    elif alter == "different_budget":
        reports["priority_train"]["rows"][0]["played"] = 3
    elif alter == "censored":
        reports["target_train"]["rows"][0]["censored"] = 1
    elif alter == "no_ppo_diagnostics":
        reports["priority_train"]["training_settings"].pop("diagnose_ppo")
    elif alter == "not_paired":
        reports["priority_eval"]["rows"][0]["baseline"]["hp_remaining"] = 9
    elif alter == "different_corpus":
        reports["priority_eval"]["evaluation"]["identity"]["corpus_sha256"] = "mismatch"
    elif alter == "different_manifest":
        reports["manifest"]["common_settings"]["target_balance"] = 0.2
    with pytest.raises(ValueError, match=message):
        compare_ablation(**reports, samples=10)


def test_manifest_requires_equal_episodes_per_round():
    reports = _reports()
    reports["manifest"]["common_settings"]["episodes"] = 3
    with pytest.raises(ValueError, match="episode budget"):
        compare_ablation(**reports, samples=10)


def test_ablation_plan_only_validates_holdout_and_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from tools.run_importance_ablation_v15p import main

    base = tmp_path / "base.json"
    vocab = tmp_path / "vocab.json"
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    for path in (base, vocab):
        path.write_text("{}")
    train.write_text(json.dumps({
        "seed": "train-1", "ascension": 0, "source_policy": "base",
    }) + "\n")
    dev.write_text(json.dumps({
        "seed": "dev-1", "ascension": 0, "source_policy": "base",
    }) + "\n")
    root = tmp_path / "ablation"
    args = [
        "ablation", "--base-model", str(base), "--vocabulary", str(vocab),
        "--snapshot-corpus", str(train), "--eval-corpus", str(dev),
        "--output-root", str(root), "--rounds", "2",
        "--episodes", "2", "--plan-only",
    ]
    monkeypatch.setattr(sys, "argv", args)
    main()
    main()  # plan-only may reuse an identical provenance manifest
    manifest = json.loads((root / "ablation-manifest.json").read_text())
    assert manifest["arms"] == {"target": 0.0, "priority": 0.7}
    dev.write_text(train.read_text())
    with pytest.raises(ValueError, match="source-run leakage"):
        main()


def test_existing_eval_verification_rejects_stale_artifacts(tmp_path: Path):
    from tools.run_importance_ablation_v15p import (
        _verify_evaluation,
        file_sha256,
    )

    base, model, corpus, report = (
        tmp_path / filename for filename in (
            "base.json", "model.json", "dev.jsonl", "evaluation.json"
        )
    )
    base.write_text("base")
    model.write_text("model")
    corpus.write_text("corpus")
    identities = {
        "baseline_sha256": file_sha256(base),
        "candidate_sha256": file_sha256(model),
        "corpus_sha256": file_sha256(corpus),
        "max_combat_decisions": 256,
    }
    report.write_text(json.dumps({"evaluation": {"identity": identities}}))
    _verify_evaluation(
        report, base=base, model=model, corpus=corpus, max_decisions=256
    )
    model.write_text("new model")
    with pytest.raises(ValueError, match="Stale evaluation fingerprint"):
        _verify_evaluation(
            report, base=base, model=model, corpus=corpus, max_decisions=256
        )
