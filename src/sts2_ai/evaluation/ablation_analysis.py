"""Compare two *completed* v15q importance-sampling experiments.

Experimental treatments must differ only in the proposal mixing parameter.
Both held-out evaluations must use the identical baseline, corpus and replay
parameters. Outcomes are compared at the paired combat level; uncertainty is
clustered on the original source run, never on individual fights.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sts2_ai.evaluation.combat_diagnostics import diagnose_reports

SCHEMA = "sts2-v15q-importance-ablation-comparison-v1"
ARMS = {"target": 0.0, "priority": 0.7}


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _training_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows", [])
    completed = report.get("rounds_completed")
    if (not isinstance(rows, list) or not rows
            or completed != len(rows)
            or [r.get("round_index") for r in rows] != list(range(len(rows)))):
        raise ValueError("Training report has missing or discontinuous rounds")
    if any(r.get("rollout_source") != "combat_snapshots" for r in rows):
        raise ValueError("Ablation must train only from combat snapshots")
    if any(r.get("censored") for r in rows):
        raise ValueError("Ablation produced censored combats; cannot compare cleanly")
    if any(r.get("mean_preclip_grad_norm") is None for r in rows):
        raise ValueError("Missing --diagnose-ppo round metrics")
    played = sum(r["played"] for r in rows)
    if played <= 0:
        raise ValueError("No training combats played")
    return {
        "rounds": len(rows),
        "played": played,
        "training_win_rate": sum(r["wins"] for r in rows) / played,
        "resolved_combats": sum(r["resolved_combats"] for r in rows),
        "optimizer_steps": sum(r["optimizer_steps"] for r in rows),
        "combat_decisions": sum(r["combat_decisions"] for r in rows),
        "ppo_samples_used": sum(r["ppo_samples_used"] for r in rows),
        "rollout_seconds": math.fsum(r["rollout_seconds"] for r in rows),
        "optimizer_seconds": math.fsum(r["optimizer_seconds"] for r in rows),
        "mean_grad_norm": _mean([r["mean_preclip_grad_norm"] for r in rows]),
        "mean_grad_clip_fraction": _mean([
            r["gradient_clip_fraction"] for r in rows
        ]),
        "mean_raw_advantage_std": _mean([
            r["raw_advantage_std"] for r in rows
        ]),
        "mean_decision_weight_ess_fraction": _mean([
            r["decision_weight_ess_fraction"] for r in rows
        ]),
        "mean_draw_ess_fraction": _mean([
            r["importance_ess"] / r["played"] for r in rows
        ]),
        "max_importance_weight": max(r["importance_weight_max"] for r in rows),
        "round_metrics": [
            {
                "round": r["round_index"] + 1,
                "wins": r["wins"], "played": r["played"],
                "mean_combat_target": r["mean_combat_target"],
                "mean_grad_norm": r["mean_preclip_grad_norm"],
                "grad_clip_fraction": r["gradient_clip_fraction"],
                "advantage_std": r["raw_advantage_std"],
                "draw_ess_fraction": r["importance_ess"] / r["played"],
                "decision_ess_fraction": r["decision_weight_ess_fraction"],
            }
            for r in rows
        ],
    }


def _immutable_settings(train: dict[str, Any]) -> dict[str, Any]:
    settings = dict(train.get("training_settings") or {})
    if settings.get("snapshot_sampling") != "importance":
        raise ValueError("Expected importance sampling in both training reports")
    settings.pop("snapshot_priority_mix", None)
    return settings


def _assert_target(
    train: dict[str, Any], expected: float, *, name: str,
) -> None:
    settings = train["training_settings"]
    if settings.get("snapshot_priority_mix") != expected:
        raise ValueError(
            f"{name} arm requires proposal mixing lambda={expected}"
        )
    if settings.get("diagnose_ppo") is not True:
        raise ValueError(f"{name} arm is missing --diagnose-ppo")


def _delta_row(
    target: dict[str, Any], priority: dict[str, Any],
) -> dict[str, Any]:
    return {
        **{
            k: priority[k]
            for k in ("seed", "floor", "tier", "progress", "enemy_ids")
        },
        "baseline": target["candidate"],
        "candidate": priority["candidate"],
    }


def _pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [
        int(row["candidate"]["outcome"] == "victory")
        - int(row["baseline"]["outcome"] == "victory")
        for row in rows
    ]
    hp = [
        row["candidate"]["hp_remaining"] - row["baseline"]["hp_remaining"]
        for row in rows
    ]
    return {
        "entries": len(rows),
        "delta_survival": _mean(wins),
        "delta_exit_hp": _mean(hp),
        "priority_only_wins": sum(x == 1 for x in wins),
        "target_only_wins": sum(x == -1 for x in wins),
    }


def compare_ablation(
    *, target_train: dict[str, Any], priority_train: dict[str, Any],
    target_eval: dict[str, Any], priority_eval: dict[str, Any],
    manifest: Mapping[str, Any] | None = None, samples: int = 2000,
    seed: int = 29,
) -> dict[str, Any]:
    """Fail closed on mismatches rather than comparing unrelated runs."""
    _assert_target(target_train, 0.0, name="target")
    _assert_target(priority_train, 0.7, name="priority")
    if _immutable_settings(target_train) != _immutable_settings(priority_train):
        raise ValueError("Ablation arms have mismatched frozen training settings")
    for key in ("architecture", "base_model", "vocabulary", "emulator_revision"):
        if target_train.get(key) != priority_train.get(key):
            raise ValueError(f"Ablation training {key} mismatch")
    target_training = _training_summary(target_train)
    priority_training = _training_summary(priority_train)
    if target_training["rounds"] != priority_training["rounds"]:
        raise ValueError("Unequal training rounds")
    if target_training["played"] != priority_training["played"]:
        raise ValueError("Unequal training combat budget")
    if target_training["optimizer_steps"] != priority_training["optimizer_steps"]:
        # We report the gap, not fail: variable-length combat affects steps.
        pass
    if target_eval.get("evaluation", {}).get("mode", "greedy") != "greedy":
        raise ValueError("Target ablation evaluation must be greedy")
    if priority_eval.get("evaluation", {}).get("mode", "greedy") != "greedy":
        raise ValueError("Priority ablation evaluation must be greedy")
    for arm, training, evaluation in (
        ("target", target_train, target_eval),
        ("priority", priority_train, priority_eval),
    ):
        expected_candidate = training.get("candidate_model")
        if expected_candidate is None and training.get("checkpoint"):
            expected_candidate = str(
                Path(training["checkpoint"]).with_name("model.json")
            )
        if expected_candidate is not None and (
            expected_candidate != evaluation["candidate_model"]
        ):
            raise ValueError(f"{arm} evaluation candidate path mismatch")
    comparison = diagnose_reports(
        {0: target_eval, 1: priority_eval},
        bootstrap_samples=samples, seed=seed, min_enemy_count=10,
    )
    if manifest is not None:
        if manifest.get("arms") != ARMS:
            raise ValueError("Ablation manifest has unexpected arms")
        files = manifest.get("inputs", {})
        ids = target_eval.get("evaluation", {}).get("identity", {})
        for field, source in (
            ("baseline_sha256", "base_model"),
            ("corpus_sha256", "eval_corpus"),
        ):
            if source in files and ids.get(field) != files[source]["sha256"]:
                raise ValueError(f"Ablation manifest/evaluation mismatch: {source}")
        settings = manifest.get("common_settings", {})
        if settings.get("rounds") != target_training["rounds"]:
            raise ValueError("Training rounds differ from ablation manifest")
        if settings.get("episodes") * settings.get("rounds") != target_training["played"]:
            raise ValueError("Training episode budget differs from manifest")
        if (
            settings.get("target_balance")
            != target_train["training_settings"]["snapshot_target_balance"]
        ):
            raise ValueError("Ablation manifest target distribution mismatch")
    first_rows = target_eval["rows"]
    second_rows = priority_eval["rows"]
    paired = [
        _delta_row(target, priority)
        for target, priority in zip(first_rows, second_rows, strict=True)
    ]
    by_tier = {
        tier: _pair_stats([r for r in paired if r["tier"] == tier])
        for tier in ("weak", "normal", "elite", "boss")
    }
    by_progress = {
        band: _pair_stats([r for r in paired if r["progress"] == band])
        for band in ("early", "middle", "late")
    }
    return {
        "schema": SCHEMA,
        "interpretation": "lambda=0.7 (priority) minus lambda=0 (target)",
        "source_runs": comparison["source_runs"],
        "scenarios": comparison["scenarios"],
        "evaluation": {
            "target_vs_v8": comparison["checkpoints"][0],
            "priority_vs_v8": comparison["checkpoints"][1],
            "priority_vs_target": comparison["first_to_last"],
            "by_tier": by_tier,
            "by_progress": by_progress,
        },
        "training": {"target": target_training, "priority": priority_training},
        "notes": [
            "One seed per arm does not establish causal robustness across training randomness.",
            "Do not count repeated snapshots from the same original source run as independent.",
            "PPO self-normalizes minibatch importance weights; its gradient is a surrogate.",
            "Source-run bootstrap intervals are descriptive, not multiple-testing corrected.",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    comparison = result["evaluation"]
    paired = comparison["priority_vs_target"]
    win = paired["cluster_95pct_delta_win_rate"]
    hp = paired["cluster_95pct_delta_hp"]

    def ci(value: list[float] | None, *, pct: bool = False) -> str:
        if value is None:
            return "n/a"
        return (
            f"[{value[0] * 100:+.2f}, {value[1] * 100:+.2f}] pp"
            if pct else f"[{value[0]:+.3f}, {value[1]:+.3f}] HP"
        )

    lines = [
        "# v15q — same-target importance sampling ablation",
        "",
        f"**{result['scenarios']} matched combat entries** across "
        f"{result['source_runs']} originating runs. "
        "All differences are prioritized proposal (lambda=0.7) "
        "minus direct target sampling (lambda=0).",
        "",
        f"Survival difference: **{paired['delta_win_rate'] * 100:+.2f} pp** "
        f"(cluster CI {ci(win, pct=True)}).",
        f"Exit HP difference: **{paired['delta_exit_hp']:+.3f} HP** "
        f"(cluster CI {ci(hp)}).",
        "",
        "## Held-out results versus v8",
        "",
        "| Arm | Wins | Δ survival vs v8 | Δ HP vs v8 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("target", "priority"):
        row = comparison[f"{arm}_vs_v8"]
        lines.append(
            f"| {arm} | {row['candidate_wins']}/{row['count']} "
            f"| {row['delta_win_rate'] * 100:+.2f} pp "
            f"| {row['delta_exit_hp']:+.3f} |"
        )
    lines.extend([
        "", "## Direct paired contrast by encounter tier", "",
        "| Tier | Scenarios | Δ wins | Δ HP |",
        "|---|---:|---:|---:|",
    ])
    for tier, row in comparison["by_tier"].items():
        if row["entries"]:
            lines.append(
                f"| {tier} | {row['entries']} | "
                f"{row['delta_survival'] * 100:+.2f} pp | "
                f"{row['delta_exit_hp']:+.3f} |"
            )
    lines.extend([
        "", "## PPO training telemetry", "",
        "| Metric | Target λ=0 | Priority λ=0.7 |",
        "|---|---:|---:|",
    ])
    a, b = (result["training"][arm] for arm in ("target", "priority"))
    for label, key, fmt in (
        ("Rounds", "rounds", "d"),
        ("Total combats", "played", "d"),
        ("Optimizer steps", "optimizer_steps", "d"),
        ("Training win rate", "training_win_rate", ".3%"),
        ("Mean gradient norm", "mean_grad_norm", ".3f"),
        ("Gradient clipping fraction", "mean_grad_clip_fraction", ".3%"),
        ("Mean advantage std", "mean_raw_advantage_std", ".3f"),
        ("Draw ESS fraction", "mean_draw_ess_fraction", ".3%"),
        ("Decision-weight ESS", "mean_decision_weight_ess_fraction", ".3%"),
    ):
        lines.append(
            f"| {label} | {format(a[key], fmt)} | {format(b[key], fmt)} |"
        )
    lines.extend([
        "", "The same target objective does not imply identical finite-sample "
        "optimizer gradients. These are single-seed explorations; use multiple "
        "training seeds and a new final holdout before claiming improvement.", "",
    ])
    return "\n".join(lines)
