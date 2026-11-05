#!/usr/bin/env python3
"""Audit H22 multi-round pilot on identical search-free heldout seeds.

Compares a fixed-horizon pre-update baseline with every post-update
evaluation. This validates the experiment and reports paired outcomes;
it does not assert that policy improvement is statistically established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

H22_SCHEMA = "sts2-h22-h21-grounded-joint-act-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _episodes(stage: dict[str, Any], count: int, label: str) -> dict[str, dict]:
    _require(
        stage.get("played") == count
        and stage.get("completed") == count
        and stage.get("censored") == 0,
        f"{label}: expected {count} completed, uncensored episodes",
    )
    records = stage["floor_progress"]["runs"]
    _require(len(records) == count, f"{label}: missing per-seed floor records")
    by_seed: dict[str, dict] = {}
    for record in records:
        seed = record["seed"]
        floor = record["max_floor_reached"]
        _require(
            isinstance(seed, str) and bool(seed) and seed not in by_seed
            and type(floor) is int and 1 <= floor <= 16
            and record["outcome"] in ("defeat", "act_clear"),
            f"{label}: invalid or duplicate per-seed floor record",
        )
        by_seed[seed] = record
    _require(
        stage.get("searches", 0) == 0 and stage.get("search_transitions", 0) == 0,
        f"{label}: heldout evaluation must have search disabled",
    )
    return by_seed


def _actor(stage: dict[str, Any], label: str, fixed_rows: int) -> None:
    actor = stage["rollout_profile"]["actor_service"]
    _require(
        isinstance(actor, dict)
        and actor.get("device") == "cuda"
        and actor.get("p2_deterministic_cuda") is True
        and actor.get("p2_cuda_cublas_workspace_config") == ":4096:8"
        and actor.get("p2_h21_collator") == "packed"
        and actor.get("p2_fixed_shape_batch_size") == fixed_rows
        and actor.get("p2_fixed_shape_bucket_forwards", 0) >= 1
        and actor.get("p2_fixed_shape_physical_rows", 0) >= actor.get("requests", 0)
        and actor.get("requests", 0) > 0,
        f"{label}: missing packed fixed-shape deterministic CUDA receipts",
    )


def _stage_summary(
    records: dict[str, dict], stage: dict[str, Any],
) -> dict[str, float | int]:
    values = [r["max_floor_reached"] for r in records.values()]
    return {
        "mean_max_floor": mean(values),
        "median_max_floor": median(values),
        "floor_one_deaths": sum(
            r["outcome"] == "defeat" and r["death_floor"] == 1
            for r in records.values()
        ),
        "act_clears": sum(r["outcome"] == "act_clear" for r in records.values()),
        "combat_wins": stage["combat_progress"]["won_fights"],
        "decisions": stage["decisions"],
    }


def verify_pilot(
    root: Path, *, expected_rounds: int = 3,
    expected_train: int = 15, expected_heldout: int = 16,
    expected_workers: int = 15, fixed_rows: int = 4,
    evaluation_horizon: float = 0.4113950303216448,
) -> dict[str, Any]:
    """Require matched evaluation conditions and summarize paired floors."""
    for name, value in (
        ("rounds", expected_rounds), ("train", expected_train),
        ("heldout", expected_heldout), ("workers", expected_workers),
        ("fixed_rows", fixed_rows),
    ):
        _require(type(value) is int and value > 0, f"Invalid expected {name}")
    _require(
        math.isfinite(evaluation_horizon) and 0 <= evaluation_horizon < 1,
        "Invalid expected fixed evaluation horizon",
    )
    progress = json.loads((root / "progress.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    _require(
        progress["experiment"]["schema"] == H22_SCHEMA,
        "Expected H22 public-v1/H21 checkpoint family",
    )
    opts = progress["experiment"]["options"]
    expected = {
        "representation": "h21", "actor_inference": "shared",
        "actor_device": "cuda", "actor_deterministic_cuda": True,
        "actor_h21_collator": "packed", "actor_fixed_shape_batch_size": fixed_rows,
        "workers": expected_workers, "runs_per_round": expected_train,
        "heldout_runs": expected_heldout, "rounds": expected_rounds,
        "map_profile": "generated", "ascension": 0,
        "rollout_transport": "fused",
    }
    for key, value in expected.items():
        _require(opts.get(key) == value, f"Experiment option drift: {key}")
    _require(
        opts.get("actor_max_batch", 0) >= fixed_rows
        and opts.get("evaluation_horizon_lambda") == evaluation_horizon,
        "Fixed evaluation horizon/batch configuration differs from pilot",
    )
    rounds = progress["rounds"]
    _require(len(rounds) == expected_rounds, "Pilot has incomplete training rounds")
    _require((root / "optimizer-state.pt").is_file(), "Missing optimizer state")
    _actor(baseline, "baseline", fixed_rows)
    baseline_records = _episodes(baseline, expected_heldout, "baseline")
    seeds = tuple(sorted(baseline_records))
    stages = [{"label": "baseline", **_stage_summary(baseline_records, baseline)}]
    per_seed: dict[str, dict[str, int | str]] = {
        seed: {"seed": seed, "baseline": baseline_records[seed]["max_floor_reached"]}
        for seed in seeds
    }
    train_stages: list[dict[str, float | int]] = []
    paired: list[dict[str, float | int]] = []
    previous = baseline_records

    for index, item in enumerate(rounds, 1):
        label = f"round_{index}"
        _require(item.get("round") == index, f"{label}: nonsequential training round")
        _require(
            isinstance(item.get("horizon_lambda"), (int, float))
            and 0 <= item["horizon_lambda"] < 1
            and item["update"]["optimizer_steps"] > 0,
            f"{label}: missing actual training update",
        )
        _require(
            item["train"]["played"] == expected_train
            and item["train"]["completed"] == expected_train
            and item["train"]["censored"] == 0
            and item["train"]["searches"] >= 1
            and item["train"]["search_transitions"] >= 1,
            f"{label}: incomplete search-assisted training cohort",
        )
        _actor(item["train"], f"{label} training", fixed_rows)
        _actor(item["heldout"], f"{label} heldout", fixed_rows)
        evaluated = _episodes(item["heldout"], expected_heldout, label)
        _require(tuple(sorted(evaluated)) == seeds, f"{label}: heldout seed drift")
        checkpoint = root / "models" / f"joint-r{index:04d}.pt"
        _require(checkpoint.is_file(), f"{label}: checkpoint missing")
        _require(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            == item["checkpoint_sha256"],
            f"{label}: checkpoint SHA-256 mismatch",
        )
        current = _stage_summary(evaluated, item["heldout"])
        stages.append({"label": label, **current})
        train_stages.append({
            "round": index, "train_mean_max_floor":
                item["train"]["floor_progress"]["mean_max_floor_reached"],
            "train_floor_one_deaths": sum(
                x["death_floor"] == 1
                for x in item["train"]["floor_progress"]["runs"]
            ),
            "search_roots": item["train"]["searches"],
            "optimizer_steps": item["update"]["optimizer_steps"],
            "collection_tps":
                item["timing"]["approx_emulator_transitions_per_second"],
        })
        baseline_deltas = [
            evaluated[s]["max_floor_reached"] - baseline_records[s]["max_floor_reached"]
            for s in seeds
        ]
        previous_deltas = [
            evaluated[s]["max_floor_reached"] - previous[s]["max_floor_reached"]
            for s in seeds
        ]
        paired.append({
            "round": index,
            "mean_delta_from_baseline": mean(baseline_deltas),
            "mean_delta_from_previous": mean(previous_deltas),
            "improved_seeds": sum(d > 0 for d in baseline_deltas),
            "unchanged_seeds": sum(d == 0 for d in baseline_deltas),
            "regressed_seeds": sum(d < 0 for d in baseline_deltas),
        })
        for seed in seeds:
            per_seed[seed][label] = evaluated[seed]["max_floor_reached"]
        previous = evaluated

    report: dict[str, Any] = {
        "schema": "sts2-h22-matched-heldout-learning-pilot-v1",
        "status": "validated matched evaluation; learning progress is descriptive",
        "fingerprint": progress["fingerprint"],
        "emulator_revision": progress["experiment"]["emulator_revision"],
        "evaluation_horizon_lambda": evaluation_horizon,
        "fixed_heldout_seed_count": expected_heldout,
        "stages": stages,
        "paired_rounds": paired,
        "train_stages": train_stages,
        "per_seed": list(per_seed.values()),
        "note": (
            "Identical heldout seeds, temperature, horizon and search-free evaluation; "
            "sample size and policy-induced stochastic paths still limit inference."
        ),
    }
    (root / "learning-pilot-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--expected-rounds", type=int, default=3)
    p.add_argument("--expected-train", type=int, default=15)
    p.add_argument("--expected-heldout", type=int, default=16)
    p.add_argument("--expected-workers", type=int, default=15)
    p.add_argument("--fixed-rows", type=int, default=4)
    p.add_argument(
        "--evaluation-horizon-lambda", type=float, default=0.4113950303216448,
    )
    args = p.parse_args()
    result = verify_pilot(
        args.output_root, expected_rounds=args.expected_rounds,
        expected_train=args.expected_train,
        expected_heldout=args.expected_heldout,
        expected_workers=args.expected_workers, fixed_rows=args.fixed_rows,
        evaluation_horizon=args.evaluation_horizon_lambda,
    )
    print(json.dumps({
        "status": result["status"],
        "stages": result["stages"],
        "paired_rounds": result["paired_rounds"],
        "train_stages": result["train_stages"],
        "report": str(args.output_root / "learning-pilot-report.json"),
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
