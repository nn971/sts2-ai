#!/usr/bin/env python3
"""Frozen two-stage H9 experiment: confirmation selects, zero-shot measures."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_combat_fixtures import (
    fixture_for_index,
    source_bank,
    structural_key,
)

CONFIRMATION_RUNS = 512  # 2 stochastic worlds x 256 trained configurations
ZERO_SHOT_RUNS = 256  # 4 stochastic worlds x 64 disjoint configurations
ROUNDS = (70, 75, 80)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_index(key: str) -> tuple[int, str]:
    parts = key.split(":", 2)
    if len(parts) != 3 or parts[0] != "structured":
        raise ValueError(f"Invalid structured cell: {key}")
    return int(parts[1]), parts[2]


def validate(
    report: dict[str, Any], *, split: str, count: int,
    models: dict[str, Path], seed_digest: str, args: argparse.Namespace,
) -> None:
    from sts2_ai.models.h7_public_policy import H7PlayerSpec

    cfg = report["config"]
    expected_models = {
        label: H7PlayerSpec.from_file(path, seed=31337, greedy=True).model_sha256
        for label, path in models.items()
    }
    bindings = args.emulator_bindings / "sts2_emulator" / "__init__.py"
    expected_cfg = {
        "split": split,
        "runs": count,
        "curriculum": "h8-structured",
        "structured_curriculum_seed": 20261027,
        "source_digest": seed_digest,
        "models": expected_models,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": digest(args.emulator_dll),
        "emulator_bindings_sha256": digest(bindings),
        "actor_device": args.actor_device,
        "actor_collation": "packed",
        "local_reference_nature": True,
    }
    for key, value in expected_cfg.items():
        if cfg.get(key) != value:
            raise ValueError(f"{split} evaluation metadata mismatch: {key}")
    lower, upper = (0, 256) if split == "confirmation" else (256, 320)
    replica_count = count // (upper - lower)
    for label, arm in report["arms"].items():
        if (arm["combats"] != count or arm["wins"] + arm["defeats"] != count or
                len(arm["by_cell"]) != upper - lower):
            raise ValueError(f"{label} has missing/invalid completed fights")
        indices = set()
        for name, cell in arm["by_cell"].items():
            index, _ = config_index(name)
            if (not lower <= index < upper or index in indices or
                    cell["combats"] != replica_count or
                    cell["wins"] + cell["defeats"] != replica_count or
                    cell["terminal_replay_verified"] != replica_count):
                raise ValueError(f"{label} invalid source or journal: {name}")
            indices.add(index)


def encounter_summary(arm: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, cell in arm["by_cell"].items():
        groups[config_index(key)[1]].append(cell)
    output = {}
    for enemy, cells in sorted(groups.items()):
        n = sum(x["combats"] for x in cells)
        wins = sum(x["wins"] for x in cells)
        output[enemy] = {
            "combats": n,
            "wins": wins,
            "win_rate": wins / n,
            "mean_score": sum(x["mean_score"] * x["combats"] for x in cells) / n,
            "victory_mean_exit_hp": (
                sum(x["victory_mean_exit_hp"] * x["wins"] for x in cells
                    if x["wins"]) / wins if wins else None
            ),
        }
    return output


def cluster_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Paired estimates; resample configurations rather than correlated episodes."""
    if a["by_cell"].keys() != b["by_cell"].keys():
        raise ValueError("Comparison arms do not cover identical configurations")
    deltas = [
        (b["by_cell"][k]["mean_score"] - a["by_cell"][k]["mean_score"],
         b["by_cell"][k]["wins"] / b["by_cell"][k]["combats"] -
         a["by_cell"][k]["wins"] / a["by_cell"][k]["combats"])
        for k in sorted(a["by_cell"])
    ]
    rng = random.Random(73127)
    means = [[], []]
    for _ in range(2000):
        selected = rng.choices(deltas, k=len(deltas))
        for dimension in (0, 1):
            means[dimension].append(
                sum(x[dimension] for x in selected) / len(selected))
    for values in means:
        values.sort()
    return {
        "configuration_clusters": len(deltas),
        "mean_score_delta": sum(x[0] for x in deltas) / len(deltas),
        "win_rate_delta": sum(x[1] for x in deltas) / len(deltas),
        "score_delta_bootstrap_95": [means[0][50], means[0][1950]],
        "win_rate_delta_bootstrap_95": [means[1][50], means[1][1950]],
        "improved_configurations": sum(x[0] > 0 for x in deltas),
        "worsened_configurations": sum(x[0] < 0 for x in deltas),
    }


def evaluate(
    args: argparse.Namespace, *, split: str, count: int,
    models: dict[str, Path], output: Path, seed_digest: str,
) -> dict[str, Any]:
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        validate(result, split=split, count=count, models=models,
                 seed_digest=seed_digest, args=args)
        print(f"[h9-generalization] Reusing checked {output}", flush=True)
        return result
    evaluator = Path(__file__).with_name("evaluate_h7_reference_combat.py")
    cmd = [
        sys.executable, "-u", str(evaluator),
        "--curriculum", "h8-structured",
        "--structured-curriculum-seed", "20261027",
        "--split", split, "--runs", str(count), "--study-seed", "73127",
        "--actor-seed", "31337", "--reference-seed", "23387",
        "--emulator-dll", str(args.emulator_dll),
        "--emulator-bindings", str(args.emulator_bindings),
        "--emulator-revision", args.emulator_revision,
        "--workers", str(args.workers),
        "--actor-device", args.actor_device,
        "--actor-collation", "packed", "--local-reference-nature",
        "--max-boundaries", "1200", "--output", str(output), "--execute",
    ]
    for label, path in models.items():
        cmd.extend(("--player", f"{label}={path}"))
    print(f"[h9-generalization] {split}: {count} fights per policy", flush=True)
    subprocess.run(cmd, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    validate(result, split=split, count=count, models=models,
             seed_digest=seed_digest, args=args)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path,
                   default=Path("results/h9-structured-a10-r80"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/h9-generalization-a10"))
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--actor-device", default="cuda")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.workers < 1:
        p.error("Worker count must be positive")

    progress_path = args.run_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    cfg = progress["config"]
    if (cfg["curriculum"] != "h8-structured-256+64-v1" or
            cfg["structured_curriculum_seed"] != 20261027 or
            len(progress["history"]) != 80 or
            progress["history"][-1]["round"] != 80):
        p.error("Expected completed H9 80-round structured run")

    models = args.run_root / "models"
    candidates = {
        f"r{n}": models / f"player-r{n:04d}.pt" for n in ROUNDS
    }
    initial = models / "player-r0000.pt"
    for path in (*candidates.values(), initial):
        if not path.is_file():
            p.error(f"Missing frozen checkpoint: {path}")

    train = {structural_key(fixture_for_index(i)) for i in range(256)}
    unseen = {structural_key(fixture_for_index(i)) for i in range(256, 320)}
    if len(train) != 256 or len(unseen) != 64 or train & unseen:
        p.error("Train and zero-shot structures overlap")
    if cfg["emulator_revision"] != args.emulator_revision:
        print("[h9-generalization] NOTE: evaluation emulator revision differs "
              "from training; both evaluated policies use one DLL.", flush=True)

    confirm_digest = canonical_sha256(
        source_bank("confirmation", CONFIRMATION_RUNS))
    unseen_digest = canonical_sha256(source_bank("zero-shot", ZERO_SHOT_RUNS))
    print("[h9-generalization] 512 confirmation fights per R70/R75/R80; "
          "then 256 zero-shot fights per selected/initial model.", flush=True)
    if not args.execute:
        print("[h9-generalization] Dry run; append --execute.", flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirmation = evaluate(
        args, split="confirmation", count=CONFIRMATION_RUNS,
        models=candidates, output=args.output_dir / "confirmation.json",
        seed_digest=confirm_digest,
    )
    for label, arm in confirmation["arms"].items():
        print(f"[h9-generalization] confirmation {label}: "
              f"{arm['wins']}/{arm['combats']}, score {arm['mean_score']:.4f}",
              flush=True)
    # Predeclared rule. Never inspect structural zero-shot before selection.
    best = max(ROUNDS, key=lambda n: (
        confirmation["arms"][f"r{n}"]["mean_score"],
        confirmation["arms"][f"r{n}"]["wins"], -n))
    chosen = f"r{best}"
    print(f"[h9-generalization] Selected {chosen} on confirmation alone",
          flush=True)
    zero_shot = evaluate(
        args, split="zero-shot", count=ZERO_SHOT_RUNS,
        models={"initial": initial, chosen: candidates[chosen]},
        output=args.output_dir / "zero-shot.json",
        seed_digest=unseen_digest,
    )
    for label, arm in zero_shot["arms"].items():
        print(f"[h9-generalization] zero-shot {label}: "
              f"{arm['wins']}/{arm['combats']}, score {arm['mean_score']:.4f}",
              flush=True)
    comparisons = {
        f"r{n}-minus-r70": cluster_comparison(
            confirmation["arms"]["r70"], confirmation["arms"][f"r{n}"])
        for n in (75, 80)
    }
    report = {
        "schema": "sts2-h9-two-stage-generalization-v1",
        "input_progress_sha256": digest(progress_path),
        "training_emulator_revision": cfg["emulator_revision"],
        "evaluation_emulator_revision": args.emulator_revision,
        "evaluation_emulator_dll_sha256": digest(args.emulator_dll),
        "confirmation_source_digest": confirm_digest,
        "zero_shot_source_digest": unseen_digest,
        "selection_rule": "max confirmation mean_score, then wins, then earlier round",
        "selected": chosen,
        "confirmation": {
            "models": {
                label: {"wins": arm["wins"], "combats": arm["combats"],
                        "mean_score": arm["mean_score"],
                        "by_encounter": encounter_summary(arm)}
                for label, arm in confirmation["arms"].items()
            },
            "paired_cluster_comparisons": comparisons,
        },
        "zero_shot": {
            "models": {
                label: {"wins": arm["wins"], "combats": arm["combats"],
                        "mean_score": arm["mean_score"],
                        "by_encounter": encounter_summary(arm)}
                for label, arm in zero_shot["arms"].items()
            },
            "selected_minus_initial": cluster_comparison(
                zero_shot["arms"]["initial"], zero_shot["arms"][chosen]),
        },
    }
    destination = args.output_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    comparison = report["zero_shot"]["selected_minus_initial"]
    print(f"[h9-generalization] zero-shot {chosen} - initial: "
          f"score {comparison['mean_score_delta']:+.4f}, "
          f"win rate {comparison['win_rate_delta']:+.1%}", flush=True)
    print(f"[h9-generalization] Report: {destination}", flush=True)


if __name__ == "__main__":
    main()
