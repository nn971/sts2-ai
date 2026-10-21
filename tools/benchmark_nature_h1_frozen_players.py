#!/usr/bin/env python3
"""Compare historical frozen combat models on the same actual draw-only fixture.

This is an *evaluation* of candidate player policies, NOT Nature learning.
Every rollout uses emulator-authored Chance Law for enemy intents and draws;
the emulator alone decides legal actions, real mechanics and hidden deck state.

Supply actual portable phase-split checkpoint JSON files. The default reference
source corpus is deliberately disjoint from H1 training and final evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import TwoSiteNature

SCHEMA = "sts2-ai-nature-h1-frozen-model-compat-benchmark-v1"
ALLOWED_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")


def candidate_argument(value: str) -> tuple[str, Path]:
    label, divider, location = value.partition("=")
    if not divider or not ALLOWED_LABEL.fullmatch(label) or not location:
        raise argparse.ArgumentTypeError(
            "Use --candidate NAME=/path/to/portable-phase-split.json"
        )
    return label, Path(location)


def source_seeds(split: str, count: int) -> tuple[str, ...]:
    if split not in ("selection", "confirmation") or count < 1:
        raise ValueError("Invalid checkpoint-comparison source partition")
    # Neither prefix appears in the H1 train/evaluation namespaces.
    return tuple(f"nature-h1-frozen-{split}-{i:08d}" for i in range(count))


def bootstrap_ci(values: tuple[float, ...], seed: int, repeats: int = 1500) -> list[float]:
    if not values or repeats < 100 or not all(math.isfinite(v) for v in values):
        raise ValueError("Cannot bootstrap incomplete model results")
    r = random.Random(seed)
    n = len(values)
    samples = sorted(
        math.fsum(values[r.randrange(n)] for _ in range(n)) / n
        for _ in range(repeats)
    )
    return [samples[int(repeats * 0.025)],
            samples[min(repeats - 1, int(repeats * 0.975))]]


def summarize_reference_cohorts(cohorts: list[tuple]) -> dict:
    """Cluster over source seed, retaining correlated reference RNG replicates."""
    if not cohorts or not cohorts[0]:
        raise ValueError("No completed combat cohorts")
    n = len(cohorts[0])
    seeds = tuple(e.source_seed for e in cohorts[0])
    if len(set(seeds)) != n:
        raise ValueError("Source seeds are not unique")
    openings = []
    for cohort in cohorts:
        if len(cohort) != n or tuple(e.source_seed for e in cohort) != seeds:
            raise ValueError("Reference cohorts differ in source order")
        if any(e.draw_mode != "reference" or not e.terminal_replay_verified
               for e in cohort):
            raise ValueError("Reference cohort contains unverified combat")
        openings.append(tuple(e.source_opening_fingerprint for e in cohort))
    if any(opening != openings[0] for opening in openings[1:]):
        raise ValueError("Reference RNG changed a source opening across replicates")

    by_source = tuple(
        statistics.mean(cohort[i].score for cohort in cohorts)
        for i in range(n)
    )
    scores = tuple(e.score for cohort in cohorts for e in cohort)
    victories = tuple(e.outcome == "victory" for cohort in cohorts for e in cohort)
    hps = tuple(e.exit_hp for cohort in cohorts for e in cohort)
    decision_count = sum(e.nontrivial_draw_count for cohort in cohorts for e in cohort)
    draw_count = sum(e.controlled_draw_count for cohort in cohorts for e in cohort)
    intent_count = sum(e.reference_intent_count for cohort in cohorts for e in cohort)
    across = [
        abs(cohorts[0][i].score - cohorts[1][i].score) for i in range(n)
    ] if len(cohorts) >= 2 else []
    # Comparing source-matched reference runs reveals whether changing physical
    # draw outcomes can have a meaningful effect for this particular player.
    draw_changed = 0
    for i in range(n):
        if len(cohorts) < 2:
            break
        first = tuple(x.selected_id for x in cohorts[0][i].decisions
                      if x.phase == "nature:draw_card")
        second = tuple(x.selected_id for x in cohorts[1][i].decisions
                       if x.phase == "nature:draw_card")
        draw_changed += first != second

    return {
        "source_count": n,
        "reference_replicates": len(cohorts),
        "combat_count": len(scores),
        "source_seeds_sha256": canonical_sha256(seeds),
        "mean_score": statistics.mean(scores),
        "mean_score_source_bootstrap_95pct":
            bootstrap_ci(by_source, 48621),
        "win_rate": statistics.mean(victories),
        "mean_exit_hp": statistics.mean(hps),
        "score_std": statistics.pstdev(scores),
        "nontrivial_draw_count": decision_count,
        "nontrivial_draws_per_combat": decision_count / len(scores),
        "physical_draws_per_combat": draw_count / len(scores),
        "stochastic_intents_per_combat": intent_count / len(scores),
        "reference_draw_changed_pair_fraction": (
            draw_changed / n if len(cohorts) >= 2 else None
        ),
        "score_changed_pair_fraction": (
            statistics.mean(delta > 1e-12 for delta in across)
            if across else None
        ),
        "mean_abs_draw_replicate_score_delta": (
            statistics.mean(across) if across else None
        ),
        "chance_probability_status": "UNVERIFIED",
        "test_only": False,
        "decision_grade": False,
    }


def compare_source_matched_scores(
    first: list[tuple], second: list[tuple],
) -> dict:
    if not first or len(first) != len(second):
        raise ValueError("Models have mismatched replication counts")
    n = len(first[0])
    gap = []
    for i in range(n):
        a = []
        b = []
        for ca, cb in zip(first, second, strict=True):
            ea, eb = ca[i], cb[i]
            if (ea.source_seed != eb.source_seed
                    or ea.reference_seed != eb.reference_seed
                    or ea.source_opening_fingerprint !=
                    eb.source_opening_fingerprint
                    or ea.first_public_fingerprint != eb.first_public_fingerprint):
                raise ValueError("Candidate comparisons lack matching openings")
            a.append(ea.score)
            b.append(eb.score)
        gap.append(statistics.mean(a) - statistics.mean(b))
    if not gap:
        raise ValueError("No paired source scores")
    return {
        "source_count": len(gap),
        "mean_score_delta_first_minus_second": statistics.mean(gap),
        "paired_source_bootstrap_95pct": bootstrap_ci(tuple(gap), 99235),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True,
                        type=candidate_argument, metavar="NAME=FILE")
    parser.add_argument("--emulator-dll", required=True, type=Path)
    parser.add_argument("--emulator-bindings", required=True, type=Path)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("selection", "confirmation"),
                        default="selection")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--reference-replicates", type=int, default=2)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--reference-seed", type=int, default=23387)
    parser.add_argument("--actor-seed", type=int, default=31337)
    parser.add_argument("--action-scope",
                        choices=("silent-basic-v1", "all-legal-v1"),
                        default="silent-basic-v1")
    parser.add_argument("--sample-player", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--allow-test-models", action="store_true",
                        help="Engineering-only: admit explicitly untrained test JSONs")
    args = parser.parse_args()

    if (args.episodes < 2 or args.reference_replicates < 2
            or args.workers < 1 or args.temperature <= 0
            or not math.isfinite(args.temperature)
            or not args.emulator_dll.is_file()
            or not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file()
            or len({label for label, _ in args.candidate}) != len(args.candidate)):
        parser.error("Invalid comparison budget, duplicate label or missing emulator")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite an existing checkpoint evaluation")
    specs: dict[str, FrozenPlayerSpec] = {}
    model_data: dict[str, dict] = {}
    for label, path in args.candidate:
        try:
            spec = FrozenPlayerSpec.from_file(
                path, seed=args.actor_seed,
                greedy=not args.sample_player,
                temperature=args.temperature, action_scope=args.action_scope,
            )
            model = spec.load_pinned()
        except (ValueError, OSError) as exc:
            parser.error(f"{label}: checkpoint import failed: {exc}")
        if (not args.allow_test_models
                and any(t in model.model_id.lower()
                        for t in ("test-only", "untrained", "synthetic"))):
            parser.error(f"{label}: only test-only weights found; pass a real checkpoint")
        specs[label] = spec
        model_data[label] = {
            "model_id": model.model_id,
            "model_sha256": spec.model_sha256,
            "combat_head_format": model.combat.format_id,
            "combat_value_objective": model.combat_value_objective,
            "action_scope": args.action_scope,
            "original_checkpoint_path": str(path),
            "test_only": args.allow_test_models,
            "frozen": True,
        }
        print(
            f"[h1-player] loaded {label} format={model.combat.format_id} "
            f"objective={model.combat_value_objective} sha={spec.model_sha256[:16]}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = source_seeds(args.split, args.episodes)
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")
    baseline = TwoSiteNature(seed=1)
    all_results: dict[str, list[tuple]] = {}
    for label, _ in args.candidate:
        cohorts = []
        for rep in range(args.reference_replicates):
            reference_seed = args.reference_seed + rep * 1000000007
            print(
                f"[h1-player] running={label} split={args.split} "
                f"reference_replicate={rep+1}/{args.reference_replicates} "
                f"source_count={len(seeds)} workers={args.workers}",
                flush=True,
            )
            batch = draw_only_batch(
                seeds, nature=baseline, spec=specs[label],
                emulator_command=command,
                emulator_bindings=args.emulator_bindings,
                emulator_revision=args.emulator_revision,
                reference_seed=reference_seed,
                draw_mode="reference",
                workers=args.workers, progress_every=max(1, len(seeds) // 10),
            )
            if batch["source_count"] != len(seeds):
                raise RuntimeError("Model evaluation produced censored combats")
            cohorts.append(batch["episodes"])
            output = args.output_dir / f"{label}-reference-{rep:02d}.jsonl"
            atomic_write(
                output,
                "".join(json.dumps(asdict(row), sort_keys=True) + "\n"
                        for row in batch["episodes"]),
            )
            print(f"[h1-player] wrote {output}", flush=True)
        all_results[label] = cohorts
        model_data[label]["summary"] = summarize_reference_cohorts(cohorts)

    labels = [label for label, _ in args.candidate]
    report = {
        "schema": SCHEMA,
        "split": args.split,
        "source_seed_hash": canonical_sha256(seeds),
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(
            args.emulator_bindings / "sts2_emulator" / "__init__.py"
        ),
        "sampling": {
            "reference_replicates": args.reference_replicates,
            "reference_seed": args.reference_seed,
            "action_scope": args.action_scope,
            "greedy_player": not args.sample_player,
            "actor_seed": args.actor_seed,
            "player_temperature": args.temperature,
            "source_count": args.episodes,
            "workers": args.workers,
        },
        "candidates": model_data,
        "paired_differences": {
            f"{labels[i]}_minus_{labels[j]}": compare_source_matched_scores(
                all_results[labels[i]], all_results[labels[j]]
            )
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        },
        "player_selection_status": "EVALUATED_NOT_CERTIFIED",
        "nature_training_status": "NOT_STARTED",
        "native_probability_calibration": "UNVERIFIED_NOT_REQUIRED",
    }
    output = args.output_dir / "report.json"
    atomic_write(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    for label in labels:
        summary = model_data[label]["summary"]
        print(
            f"[h1-player] {label} score={summary['mean_score']:.5f} "
            f"win={summary['win_rate']:.3f} hp={summary['mean_exit_hp']:.2f} "
            f"nontrivial_draws_per_combat="
            f"{summary['nontrivial_draws_per_combat']:.2f} "
            f"score_draw_variation={summary['score_changed_pair_fraction']:.3f}",
            flush=True,
        )
    print(f"[h1-player] COMPLETE report={output}", flush=True)


if __name__ == "__main__":
    main()
