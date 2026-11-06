#!/usr/bin/env python3
"""Paired H22 checkpoint selection on fresh seeds; preserve final-audit independence.

The same *selection* seeds are used for every checkpoint. This is distinct from
the one-time independent final audit: do not reuse selection seeds as final evidence.
All model inference uses the original frozen H22 packed CUDA configuration.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from tools.audit_h22_independent import audit_checkpoint
    from tools.train_h16_joint_act_pilot import write_json
except ModuleNotFoundError:
    from audit_h22_independent import audit_checkpoint
    from train_h16_joint_act_pilot import write_json


SCHEMA = "sts2-h23-paired-h22-checkpoint-selection-v1"


def boss_stats(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["floor_progress"]["runs"]
    reached = [r for r in records if (r.get("max_floor_reached") or 0) >= 16]
    won = {r["seed"] for r in records if r["outcome"] == "act_clear"}
    if len(won) != summary["cleared"] or not won.issubset(
        {r["seed"] for r in reached}
    ):
        raise ValueError("Act clears and floor-16 reached records disagree")
    entry = [
        r["post_room_hp"]["15"]
        for r in reached if "15" in r["post_room_hp"]
    ]
    entry_fractions = [
        r["hp"] / max(1, r["max_hp"]) for r in entry
    ]
    lost_boss = [
        r["enemy_hp_removed_fraction"]
        for r in summary["combat_progress"]["lost_fights"]
        if r["floor"] == 16
    ]
    return {
        "episodes": len(records),
        "completed": summary["completed"],
        "censored": summary["censored"],
        "cleared": len(won),
        "act_clear_rate": len(won) / len(records) if records else None,
        "mean_floor": summary["floor_progress"]["mean_max_floor_reached"],
        "boss_reached": len(reached),
        "boss_clear_rate_given_reached": (
            len(won) / len(reached) if reached else None
        ),
        "boss_entry_hp_observations": len(entry),
        "mean_boss_entry_hp": (
            statistics.mean(r["hp"] for r in entry) if entry else None
        ),
        "mean_boss_entry_hp_fraction": (
            statistics.mean(entry_fractions) if entry_fractions else None
        ),
        "boss_defeats_recorded": len(lost_boss),
        "mean_enemy_hp_removed_on_boss_defeat": (
            statistics.mean(lost_boss) if lost_boss else None
        ),
        "by_seed": {
            r["seed"]: {
                "floor": r["max_floor_reached"],
                "boss_reached": (r.get("max_floor_reached") or 0) >= 16,
                "cleared": r["outcome"] == "act_clear",
            }
            for r in records
        },
    }


def paired_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = left["by_seed"], right["by_seed"]
    if set(a) != set(b):
        raise ValueError("Compared checkpoints have nonmatching seeds")
    keys = sorted(a)
    return {
        "seeds": len(keys),
        "mean_floor_right_minus_left": statistics.mean(
            b[k]["floor"] - a[k]["floor"] for k in keys
        ),
        "floor_better_right": sum(b[k]["floor"] > a[k]["floor"] for k in keys),
        "floor_better_left": sum(b[k]["floor"] < a[k]["floor"] for k in keys),
        "floor_equal": sum(b[k]["floor"] == a[k]["floor"] for k in keys),
        "boss_reached_only_right": sum(
            b[k]["boss_reached"] and not a[k]["boss_reached"] for k in keys
        ),
        "boss_reached_only_left": sum(
            a[k]["boss_reached"] and not b[k]["boss_reached"] for k in keys
        ),
        "act_clear_only_right": sum(
            b[k]["cleared"] and not a[k]["cleared"] for k in keys
        ),
        "act_clear_only_left": sum(
            a[k]["cleared"] and not b[k]["cleared"] for k in keys
        ),
    }


def select_for_diagnosis(metrics: dict[str, dict[str, Any]]) -> str:
    """Exploratory choice only; do not report it as independent confirmation."""
    return max(
        sorted(metrics),
        key=lambda label: (
            metrics[label]["cleared"],
            metrics[label]["boss_reached"],
            metrics[label]["mean_floor"],
        ),
    )


def selection(
    *, progress: Path, emulator_root: Path, output_root: Path,
    rounds: list[int], seeds: int, workers: int, tag: str, execute: bool,
) -> dict[str, Any]:
    if len(rounds) < 2 or len(set(rounds)) != len(rounds):
        raise ValueError("At least two unique checkpoint rounds required")
    if seeds <= 0 or workers <= 0:
        raise ValueError("Seed and worker counts must be positive")
    if "final" in tag:
        raise ValueError("This cohort is for checkpoint selection, not a final audit")
    planned: dict[str, dict[str, Any]] = {}
    for number in rounds:
        label = f"r{number:04d}"
        path = output_root / f"{label}-selection.json"
        args = {
            "progress_path": progress,
            "emulator_root": emulator_root,
            "audit_count": seeds,
            "audit_tag": tag,
            "round_number": number,
            "workers": workers,
            "output": path,
        }
        identity = audit_checkpoint(**args, execute=False)
        planned[label] = {"path": str(path), "identity": identity}
    cohorts = [tuple(item["identity"]["seeds"]) for item in planned.values()]
    if any(cohort != cohorts[0] for cohort in cohorts):
        raise ValueError("Candidate checkpoints were assigned different seeds")

    if not execute:
        return {
            "execute": False, "schema": SCHEMA,
            "rounds": rounds, "seeds": seeds, "tag": tag,
            "checkpoint_shas": {
                label: entry["identity"]["checkpoint_sha256"]
                for label, entry in planned.items()
            },
            "cohort_first_seed": cohorts[0][0],
            "cohort_last_seed": cohorts[0][-1],
            "output_root": str(output_root),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    for label, entry in planned.items():
        path = Path(entry["path"])
        identity = entry["identity"]
        if path.is_file():
            document = json.loads(path.read_text())
            for key in ("training_fingerprint", "checkpoint_sha256", "round",
                        "audit_count", "audit_tag", "seeds", "search"):
                if document.get(key) != identity.get(key):
                    raise ValueError(f"Existing {label} audit identity mismatch: {key}")
            if document.get("schema") != "sts2-h22-independent-checkpoint-audit-v1":
                raise ValueError(f"Unexpected existing {label} audit schema")
        else:
            print(f"[h23-selection] evaluating {label} on {seeds} seeds", flush=True)
            audit_checkpoint(
                progress, emulator_root=emulator_root, audit_count=seeds,
                audit_tag=tag, round_number=int(label[1:]),
                workers=workers, output=path, execute=True,
            )
            document = json.loads(path.read_text())
        summary = document["summary"]
        if (summary["played"] != seeds or summary["completed"] != seeds
                or summary["censored"] != 0 or summary["searches"] != 0):
            raise ValueError(f"Invalid search-free selection results for {label}")
        if {r["seed"] for r in summary["floor_progress"]["runs"]} != set(cohorts[0]):
            raise ValueError(f"Seed mismatch in observed results for {label}")
        metrics[label] = boss_stats(summary)
        print(f"[h23-selection] {label}: floors={metrics[label]['mean_floor']:.3f} "
              f"reaches={metrics[label]['boss_reached']}/{seeds} "
              f"clears={metrics[label]['cleared']}/{seeds}", flush=True)

    labels = list(metrics)
    pairs = {
        f"{left}_vs_{right}": paired_difference(metrics[left], metrics[right])
        for i, left in enumerate(labels) for right in labels[i + 1:]
    }
    report = {
        "schema": SCHEMA,
        "training_fingerprint": next(iter(planned.values()))["identity"]["training_fingerprint"],
        "purpose": "checkpoint-selection, not independent final validation",
        "tag": tag, "seed_count": seeds,
        "model_shas": {
            label: entry["identity"]["checkpoint_sha256"]
            for label, entry in planned.items()
        },
        "metrics": metrics, "paired": pairs,
        "candidate_for_boss_diagnosis": select_for_diagnosis(metrics),
        "limitations": [
            "This cohort selects a model; final generalization needs different seeds.",
            "The same seed is a paired comparison but model actions can diverge.",
            "Boss-entry HP is the public completed floor-15 HP, when recorded.",
            "Rare Act clears have wide sampling uncertainty at this cohort size.",
        ],
    }
    path = output_root / "selection-summary.json"
    if path.exists():
        raise ValueError("Selection summary already exists; use a new output root")
    write_json(path, report)
    print(f"[h23-selection] report={path}", flush=True)
    return {
        "execute": True, "report": str(path),
        "candidate_for_boss_diagnosis": report["candidate_for_boss_diagnosis"],
        "metrics": {key: {k: v for k, v in d.items() if k != "by_seed"}
                    for key, d in metrics.items()},
        "paired": pairs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--round", type=int, action="append", dest="rounds")
    p.add_argument("--seeds", type=int, default=48)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--tag", default="h23-select-v1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    try:
        result = selection(
            progress=args.progress, emulator_root=args.emulator_root,
            output_root=args.output_root,
            rounds=args.rounds or [95, 120, 150],
            seeds=args.seeds, workers=args.workers, tag=args.tag,
            execute=args.execute,
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        p.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
