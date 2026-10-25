#!/usr/bin/env python3
"""H5 reference-law confirmation and held-out-cell evaluation.

Evaluates exported frozen greedy policies on matched source-world IDs.
Different policies may consume different subsequent Chance-Law draws; this is
source pairing, NOT identical counterfactual draw paths or native RNG parity.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.h4_reward import H4_REWARD
from sts2_ai.nature.h5_curriculum import (
    DECK_POOL,
    SCENARIO_POOL,
    cell_for_source,
    source_bank,
)
from sts2_ai.nature.h5_curriculum import (
    SCHEMA as CURRICULUM_SCHEMA,
)
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256

SCHEMA = "sts2-ai-h5-reference-confirmation-v1"


def _score(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Missing H5 evaluation worlds")
    cells: dict[str, list[dict]] = {}
    tiers: dict[str, list[dict]] = {}
    for row in rows:
        cell = cell_for_source(row["seed"])
        if cell is None:
            raise ValueError("Unsupported H5 evaluation source")
        cells.setdefault(cell.key, []).append(row)
        tiers.setdefault(cell.tier, []).append(row)

    def aggregate(items: list[dict]) -> dict:
        return {
            "n": len(items),
            "score": statistics.mean(r["score"] for r in items),
            "win_rate": statistics.mean(r["victory"] for r in items),
            "exit_hp": statistics.mean(r["exit_hp"] for r in items),
        }

    return {
        **aggregate(rows),
        "by_tier": {key: aggregate(items)
                    for key, items in sorted(tiers.items())},
        "by_cell": {key: aggregate(items)
                    for key, items in sorted(cells.items())},
    }


def _paired(a: list[dict], b: list[dict], *, bootstrap_seed: int) -> dict:
    if [r["seed"] for r in a] != [r["seed"] for r in b]:
        raise ValueError("H5 paired score rows have different source IDs")
    delta_score = [right["score"] - left["score"]
                   for left, right in zip(a, b, strict=True)]
    delta_win = [int(right["victory"]) - int(left["victory"])
                 for left, right in zip(a, b, strict=True)]
    rng = random.Random(bootstrap_seed)
    n = len(a)
    replicates = sorted(
        statistics.mean(delta_score[rng.randrange(n)] for _ in range(n))
        for _ in range(2000)
    )
    return {
        "n_paired_source_worlds": n,
        "score_delta": statistics.mean(delta_score),
        "victory_delta": statistics.mean(delta_win),
        "score_delta_source_bootstrap_95pct": [
            replicates[49], replicates[1949],
        ],
        "interpretation": (
            "Descriptive paired-by-source estimate. Paths may diverge after "
            "policy actions; this is not a pathwise-coupled causal estimate."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append", required=True,
                   help="Unique label=path for exported frozen player JSON")
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("confirmation", "zero-shot"),
                   default="confirmation")
    p.add_argument("--eval-episodes", type=int, default=160)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--study-seed", type=int, default=29)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    models: dict[str, Path] = {}
    for entry in args.player:
        label, sep, path = entry.partition("=")
        if not sep or not label or label in models:
            p.error("Each --player must be a unique label=path")
        models[label] = Path(path)
    if (args.eval_episodes < 1 or args.workers < 1
            or args.max_boundaries < 1 or args.study_seed < 0):
        p.error("Invalid H5 evaluation settings")
    if not (args.emulator_dll.is_file()
            and (args.emulator_bindings/"sts2_emulator"/"__init__.py").is_file()
            and all(path.is_file() for path in models.values())):
        p.error("Missing player export, semantic emulator, or bindings")
    try:
        seeds = source_bank(
            args.split, args.eval_episodes, study_seed=args.study_seed,
        )
    except ValueError as exc:
        p.error(str(exc))

    config = {
        "schema": SCHEMA,
        "curriculum_schema": CURRICULUM_SCHEMA,
        "split": args.split,
        "seed_digest": canonical_sha256(seeds),
        "player_models": {key: file_sha256(path)
                          for key, path in sorted(models.items())},
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(
            args.emulator_bindings/"sts2_emulator"/"__init__.py"
        ),
        "eval_episodes": len(seeds),
        "workers": args.workers,
        "max_boundaries": args.max_boundaries,
        "study_seed": args.study_seed,
        "actor_seed": args.actor_seed,
        "reference_seed": args.reference_seed,
        "draw_mode": "reference",
        "reward_version": H4_REWARD,
    }
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return

    if args.resume:
        if not args.output.is_file():
            p.error("Cannot resume without an existing H5 report")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved.get("config") != config:
            p.error("H5 resume configuration or pinned artifact mismatch")
        worlds = saved["worlds"]
    else:
        if args.output.exists():
            p.error("H5 output exists; use --resume or a new --output")
        worlds = {}

    nature = PpoDrawNature(seed=109, mix=0.75)  # Inert reference-law adapter.
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")

    def save(complete: bool = False) -> None:
        summaries = {
            label: _score(rows) for label, rows in sorted(worlds.items())
        }
        comparisons = {}
        baseline = sorted(worlds)[0] if worlds else None
        if baseline is not None:
            for label in sorted(worlds):
                if label != baseline:
                    comparisons[f"{label}-minus-{baseline}"] = _paired(
                        worlds[baseline], worlds[label],
                        bootstrap_seed=90117,
                    )
        atomic_write(args.output, json.dumps({
            "config": config, "worlds": worlds,
            "summaries": summaries, "comparisons": comparisons,
            "complete": complete,
        }, indent=2, sort_keys=True)+"\n")

    for label, model_path in sorted(models.items()):
        if label in worlds:
            if tuple(row["seed"] for row in worlds[label]) != seeds:
                p.error("Resumed H5 source worlds have changed")
            print(f"[h5-eval] reuse {label}", flush=True)
            continue
        spec = FrozenPlayerSpec.from_file(
            model_path, seed=args.actor_seed, greedy=True,
            temperature=1.0, action_scope="all-legal-v1",
        )
        result = draw_only_batch(
            seeds, nature=nature, spec=spec,
            emulator_command=command,
            emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision,
            reference_seed=args.reference_seed+700000009,
            workers=args.workers, max_boundaries=args.max_boundaries,
            scenario_pool=SCENARIO_POOL, deck_pool=DECK_POOL,
            draw_mode="reference", reward_version=H4_REWARD,
            progress_every=max(1, len(seeds)//8),
        )
        rows = []
        for seed, ep in zip(seeds, result["episodes"], strict=True):
            cell = cell_for_source(seed)
            if (cell is None or ep.source_seed != seed
                    or ep.scenario_id != cell.scenario
                    or ep.deck_preset_id != cell.deck
                    or ep.draw_mode != "reference"
                    or not ep.terminal_replay_verified):
                raise RuntimeError("H5 evaluation world provenance drift")
            rows.append({
                "seed": seed, "scenario": cell.scenario,
                "deck": cell.deck, "tier": cell.tier,
                "score": ep.score, "victory": ep.outcome == "victory",
                "exit_hp": ep.exit_hp,
                "terminal_fingerprint": ep.terminal_fingerprint,
            })
        worlds[label] = rows
        save()
        metrics = _score(rows)
        print(f"[h5-eval] {label}/{args.split} score={metrics['score']:.4f} "
              f"win={metrics['win_rate']:.3f} "
              f"seconds={result['elapsed_seconds']:.1f}", flush=True)
    save(complete=True)


if __name__ == "__main__":
    main()
