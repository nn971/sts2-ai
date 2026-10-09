"""Compare the two boss-shaping trials on identical, completed held-out runs.

Reads only recorded fixed-seed monitor JSONL; never influences training.
Example:
  python tools/compare_boss_ablation.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_CONTROL = Path("results/native-boss-control-50-report.monitor.jsonl")
DEFAULT_SHAPED = Path("results/native-boss-shaped-50-report.monitor.jsonl")


def load(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("schema") != "sts2-fixed-seed-selfplay-monitor-v1":
            raise ValueError(f"Unrecognized monitor format: {path}")
        index = obj["training_round"]
        if index in rows:
            raise ValueError(f"Duplicate round {index} in {path}")
        rows[index] = obj
    if not rows:
        raise ValueError(f"Empty monitoring history: {path}")
    return rows


def paired_frontier(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, float | None]:
    a = {item["seed"]: item for item in left["per_seed"]}
    b = {item["seed"]: item for item in right["per_seed"]}
    if a.keys() != b.keys():
        raise ValueError("Monitor records do not have identical held-out seeds")
    differences = [
        b[seed]["frontier_progress"] - a[seed]["frontier_progress"]
        for seed in sorted(a)
        if not a[seed]["censored"] and not b[seed]["censored"]
    ]
    return len(differences), (statistics.fmean(differences) if differences else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--shaped", type=Path, default=DEFAULT_SHAPED)
    args = parser.parse_args()
    control, shaped = load(args.control), load(args.shaped)
    common = sorted(control.keys() & shaped.keys())
    if not common:
        raise ValueError("No matching evaluation rounds; use the same monitor cadence")
    print("round | completed pairs | shaped − control frontier | "
          "control boss(entries/losses/clears, avg damage) | "
          "shaped boss(entries/losses/clears, avg damage)", flush=True)
    for index in common:
        a, b = control[index], shaped[index]
        for field in ("seed_count", "seed_prefix", "environment"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Paired monitoring mismatch: {field}")
        paired, delta = paired_frontier(a, b)
        fields = []
        for row in (a, b):
            boss = row["boss_health_completed_only"]
            fields.append(
                f"{boss['boss_entries']}/{boss['boss_defeats']}/"
                f"{boss['boss_clears']}, "
                f"{boss['mean_boss_damage_fraction_on_defeat']}"
            )
        print(f"{index:>5} | {paired:>15} | {delta!s:>25} | "
              f"{fields[0]} | {fields[1]}", flush=True)
    print(
        "\nCaution: the boss-damage average is conditional on reaching and "
        "dying at the boss, so two differently selected groups are NOT "
        "a paired causal comparison. Compare boss entry and clear rates "
        "along with the paired frontier change.", flush=True
    )


if __name__ == "__main__":
    main()
