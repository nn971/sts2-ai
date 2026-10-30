#!/usr/bin/env python3
"""Repeated A10 H15 ReferenceNature PUCT / policy-value training.

A declared-emulator-law research experiment, not a native-game posterior.
CPU workers own independent H8 sessions and ReferenceNature cursors; the
main process performs CUDA optimizer updates. Completed rounds resume exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sts2_ai.emulator.h15_h8_reference import H8_DECLARED_LAW_MODEL
from sts2_ai.nature.combat_scenarios import (
    DECK_PRESETS,
    SCENARIOS,
    deck_for_seed,
    scenario_for_seed,
)
from sts2_ai.training.h15_search_supervision import decode_target

REGIME = "fixed-h8-fixture-declared-chance-law"
SCHEMA = "sts2-h15-reference-round-training-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed63(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 1


def atomic_json(path: Path, record: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def searches_for_mode(mode: str, max_searches: int) -> int:
    if mode not in ("train", "search", "greedy") or max_searches < 0:
        raise ValueError("Invalid H15 collection mode or search budget")
    return max_searches if mode in ("train", "search") else 0


def _collect_worker(
    cfg: dict[str, Any], checkpoint: str, case_ids: list[str],
    mode: str,
) -> dict[str, Any]:
    """Reuse one emulator JSONL process and one CPU evaluator for a case chunk."""
    import torch

    torch.set_num_threads(1)
    from sts2_ai.emulator.chance import PublicHistoryStep
    from sts2_ai.emulator.h15_h8_reference import (
        H8FixtureEntry,
        H8ReferenceCombatBackend,
        H8ReferenceNaturePosterior,
    )
    from sts2_ai.evaluation.combat_snapshots import POLICY
    from sts2_ai.models.h15_policy_value import H15CombatEvaluator
    from sts2_ai.training.h15_native_search_collector import collect_live_combat_search

    sys.path.insert(0, str(Path(cfg["emulator_root"]) / "bindings/python"))
    from sts2_emulator import Session  # type: ignore[import-not-found]

    teacher = H15CombatEvaluator.load(Path(checkpoint), device="cpu")
    records = []
    started = time.monotonic()
    with Session(["dotnet", cfg["emulator_dll"], "prototype-ai-jsonl"]) as session:
        for case_id in case_ids:
            scenario = scenario_for_seed(case_id, tuple(cfg["scenarios"]))
            deck = deck_for_seed(case_id, tuple(cfg["decks"]))
            entry = H8FixtureEntry(
                seed=cfg["fixture_seed"], encounter_id=scenario.encounter_id,
                native_entry=scenario.native_entry, act=scenario.act,
                total_floor=scenario.total_floor, ascension=scenario.ascension,
                first_monster_combat_id=scenario.first_monster_combat_id,
                map_column=scenario.map_column, map_row=scenario.map_row,
                fixture_deck_preset_id=deck,
            )
            backend = H8ReferenceCombatBackend(
                session, entry, emulator_revision=cfg["emulator_revision"],
                use_declared_chance_laws=True,
            )
            sampler = H8ReferenceNaturePosterior(
                backend, max_candidates=cfg["max_candidates"],
            )
            source_seed = seed63(case_id + ":nature")
            source = backend.begin_reference(source_seed)
            try:
                obs = backend.observe(source, POLICY)
                history = (PublicHistoryStep(
                    obs, None, backend.legal_actions(source),
                ),)
                report = collect_live_combat_search(
                    backend, source, history, fair_sampler=sampler,
                    teacher=teacher, source_seed=case_id,
                    entry_hash=obs.observation_hash, encounter_tier=(
                        "boss" if "boss" in scenario.scenario_id
                        else "weak" if scenario.scenario_id.startswith(
                            ("shrinker-", "nibbits-")) else "normal"
                    ),
                    simulations=cfg["simulations"],
                    max_decisions=cfg["max_decisions"],
                    max_searches=searches_for_mode(mode, cfg["max_searches"]),
                    seed=seed63(case_id + ":search"),
                    tree_depth=cfg["tree_depth"],
                    rollout_depth=cfg["rollout_depth"], leaf_mode="value",
                    search_schedule=cfg["search_schedule"],
                )
                if backend.live_handles:
                    raise RuntimeError("H15 worker leaked H8 reference handle(s)")
                targets = []
                for row in report.targets:
                    row = dict(row)
                    row["chance_prior_id"] = H8_DECLARED_LAW_MODEL
                    row["experiment_regime"] = REGIME
                    row["source_scenario_id"] = scenario.scenario_id
                    row["source_deck_preset_id"] = deck
                    row["source_nature_seed"] = source_seed
                    row["source_fixture_seed"] = cfg["fixture_seed"]
                    if (decode_target(row).search_model_sha256 !=
                            teacher.checkpoint_sha256):
                        raise ValueError("H15 teacher target checkpoint drift")
                    targets.append(row)
                if mode == "train" and not targets:
                    raise RuntimeError("Training combat completed without searched roots")
                records.append({
                    "case_id": case_id, "encounter": scenario.scenario_id,
                    "deck": deck, "source_nature_seed": source_seed,
                    "outcome": report.outcome, "reward": report.reward,
                    "exit_hp": report.exit_hp, "decisions": report.combat_decisions,
                    "searched_roots": report.searched_decisions,
                    "searched_root_diagnostics": list(report.searched_root_diagnostics),
                    "search_overrides": sum(
                        bool(d["search_overrode_greedy"])
                        for d in report.searched_root_diagnostics
                    ),
                    "targets": targets,
                })
                print(f"[h15-reference-{mode}] case={case_id} "
                      f"encounter={scenario.scenario_id} deck={deck} "
                      f"outcome={report.outcome} hp={report.exit_hp} "
                      f"roots={len(targets)}", flush=True)
            except BaseException as exc:
                backend.close()
                raise RuntimeError(
                    f"H15 ReferenceNature case={case_id} "
                    f"scenario={scenario.scenario_id} deck={deck} "
                    f"mode={mode} failed: {type(exc).__name__}: {exc}"
                ) from exc
    return {
        "teacher_sha256": teacher.checkpoint_sha256, "mode": mode,
        "cases": case_ids, "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }


def collect_cases(
    cfg: dict[str, Any], teacher: Path, case_ids: list[str],
    mode: str, destination: Path,
) -> list[dict[str, Any]]:
    """Persist complete chunks so an interrupted round can resume collection."""
    if not case_ids:
        raise ValueError("Collection requires explicit source cases")
    destination.mkdir(parents=True, exist_ok=True)
    groups = [case_ids[i::cfg["workers"]]
              for i in range(min(cfg["workers"], len(case_ids)))]
    model_hash = sha256(teacher)
    by_case: dict[str, dict[str, Any]] = {}
    pending = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for index, group in enumerate(groups):
            output = destination / f"worker-{index:02d}.json"
            if output.exists():
                saved = json.loads(output.read_text(encoding="utf-8"))
                if (saved.get("teacher_sha256") != model_hash
                        or saved.get("mode") != mode
                        or saved.get("cases") != group):
                    raise RuntimeError(f"Incompatible saved H15 worker chunk: {output}")
                for record in saved["records"]:
                    by_case[record["case_id"]] = record
                print(f"[h15-reference-resume] {mode} {output.name} "
                      f"cases={len(group)}", flush=True)
            else:
                pending.append((
                    pool.submit(_collect_worker, cfg, str(teacher), group, mode),
                    output,
                ))
        for future in as_completed([f for f, _ in pending]):
            output = next(path for f, path in pending if f is future)
            result = future.result()
            if result["teacher_sha256"] != model_hash:
                raise ValueError("Worker returned a different teacher identity")
            atomic_json(output, result)
            for record in result["records"]:
                by_case[record["case_id"]] = record
            print(f"[h15-reference-chunk] {mode} {output.name} "
                  f"cases={len(result['records'])} "
                  f"elapsed_s={result['elapsed_seconds']:.1f}", flush=True)
    if set(by_case) != set(case_ids):
        raise ValueError("Reference cohort is missing or duplicating source cases")
    return [by_case[key] for key in case_ids]


def write_training_targets(records: list[dict[str, Any]], path: Path) -> int:
    rows = [row for r in records for row in r["targets"]]
    if not rows:
        raise ValueError("No H15 PUCT visits in successful combat cohort")
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        raise ValueError("Empty reference evaluation")
    return {
        "combats": n,
        "victories": sum(r["outcome"] == "victory" for r in records),
        "win_rate": sum(r["outcome"] == "victory" for r in records) / n,
        "mean_exit_hp": sum(r["exit_hp"] for r in records) / n,
        "mean_reward": sum(r["reward"] for r in records) / n,
        "mean_decisions": sum(r["decisions"] for r in records) / n,
        "searched_roots": sum(r["searched_roots"] for r in records),
        "search_overrides": sum(r["search_overrides"] for r in records),
        "search_override_rate": (
            sum(r["search_overrides"] for r in records) /
            sum(r["searched_roots"] for r in records)
            if sum(r["searched_roots"] for r in records) else None
        ),
        "mean_search_visit_kl_to_prior": (
            sum(
                d["visit_policy_kl_from_prior"] for r in records
                for d in r["searched_root_diagnostics"]
            ) / sum(r["searched_roots"] for r in records)
            if sum(r["searched_roots"] for r in records) else None
        ),
        "encounter_counts": {
            name: sum(r["encounter"] == name for r in records)
            for name in sorted({r["encounter"] for r in records})
        },
    }


def run_optimizer(cfg: dict[str, Any], teacher: Path, data: Path,
                  output: Path) -> None:
    cmd = [
        sys.executable, str(Path(__file__).with_name("train_h15_search_distill.py")),
        "--initialize-from-model", str(teacher),
        "--search-data", str(data),
        "--accepted-chance-prior", H8_DECLARED_LAW_MODEL,
        "--output", str(output), "--epochs", str(cfg["epochs"]),
        "--batch-size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--value-weight", str(cfg["value_weight"]),
        "--device", cfg["learner_device"],
        "--seed", str(cfg["seed"]), "--execute",
    ]
    print("[h15-reference-optimizer] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--initialize-from-model", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--train-combats", type=int, default=128)
    p.add_argument("--eval-combats", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--scenarios", default="mawler-a10,cubex-a10,vine-shambler-a10")
    p.add_argument(
        "--decks",
        default="silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1",
    )
    p.add_argument("--fixture-seed", default="h15-reference-m15-fixture-v1")
    p.add_argument("--seed-prefix", default="h15-reference-m15-v1")
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--simulations", type=int, default=32)
    p.add_argument("--max-searches", type=int, default=1)
    p.add_argument(
        "--search-schedule", choices=("first", "turn_start"), default="first",
    )
    p.add_argument("--tree-depth", type=int, default=8)
    p.add_argument("--rollout-depth", type=int, default=128)
    p.add_argument("--max-candidates", type=int, default=4096)
    p.add_argument("--max-decisions", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.000003)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--learner-device", default="cuda")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if (min(args.rounds, args.train_combats, args.eval_combats,
            args.eval_every, args.workers, args.simulations,
            args.max_searches, args.tree_depth, args.rollout_depth,
            args.max_candidates, args.max_decisions, args.epochs,
            args.batch_size) <= 0 or args.seed < 0
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.value_weight) or args.value_weight < 0
            or not args.fixture_seed or not args.seed_prefix
            or len(args.emulator_revision) != 40):
        p.error("Invalid H15 reference training configuration")
    scenarios = tuple(x.strip() for x in args.scenarios.split(","))
    decks = tuple(x.strip() for x in args.decks.split(","))
    if (not scenarios or len(set(scenarios)) != len(scenarios)
            or any(x not in SCENARIOS or SCENARIOS[x].ascension != 10
                   for x in scenarios)
            or not decks or len(set(decks)) != len(decks)
            or any(x not in DECK_PRESETS for x in decks)):
        p.error("Only explicit, unique supported A10 scenarios/decks are valid")
    dll = args.emulator_dll.resolve()
    initial = args.initialize_from_model.resolve()
    emulator_root = args.emulator_root.resolve()
    if (not dll.is_file() or not initial.is_file()
            or not (emulator_root / "bindings/python").is_dir()):
        p.error("Missing pinned DLL, model, or emulator Python bindings")

    # Direct CLI execution places tools/ (not necessarily the repo root) on
    # sys.path. Import the existing feedback helpers from that same directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_h15_h8_feedback_cycle import (
        prepare_consistent_h15_teacher,
        verify_distillation_receipt,
        verify_parameter_update,
    )

    config = {
        "schema": SCHEMA, "emulator_root": str(emulator_root),
        "emulator_revision": args.emulator_revision,
        "emulator_dll": str(dll), "emulator_dll_sha256": sha256(dll),
        "source_model_sha256": sha256(initial),
        "scenarios": list(scenarios), "decks": list(decks),
        "fixture_seed": args.fixture_seed, "seed_prefix": args.seed_prefix,
        "workers": args.workers, "simulations": args.simulations,
        "max_searches": args.max_searches,
        "search_schedule": args.search_schedule, "tree_depth": args.tree_depth,
        "rollout_depth": args.rollout_depth, "max_candidates": args.max_candidates,
        "max_decisions": args.max_decisions, "train_combats": args.train_combats,
        "eval_combats": args.eval_combats, "eval_every": args.eval_every,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "value_weight": args.value_weight,
        "learner_device": args.learner_device, "seed": args.seed,
        "code_sha256": {
            name: sha256(Path(__file__).resolve().parents[1] / name)
            for name in (
                "tools/train_h15_reference_rounds.py",
                "tools/run_h15_h8_feedback_cycle.py",
                "tools/train_h15_search_distill.py",
                "src/sts2_ai/training/h15_native_search_collector.py",
                "src/sts2_ai/search/combat_puct.py",
                "src/sts2_ai/training/h15_search_supervision.py",
                "src/sts2_ai/emulator/h15_h8_reference.py",
                "src/sts2_ai/models/h15_combat_codec.py",
            )
        },
    }
    print("[h15-reference-preflight] " + json.dumps({
        **{k: config[k] for k in ("schema", "emulator_revision", "scenarios",
            "decks", "workers", "train_combats", "eval_combats", "simulations")},
        "rounds": args.rounds, "output_root": str(args.output_root),
        "execute": args.execute,
    }, sort_keys=True), flush=True)
    if not args.execute:
        print("[h15-reference-preflight] add --execute to train; no files changed")
        return

    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("H15 resume configuration/provenance changed")
    else:
        atomic_json(config_path, config)
    bootstrap = out / "bootstrap-h15.pt"
    if not bootstrap.exists():
        receipt = prepare_consistent_h15_teacher(initial, bootstrap)
        atomic_json(out / "bootstrap.json", receipt)
    boot_receipt = json.loads((out / "bootstrap.json").read_text())
    if (sha256(bootstrap) != boot_receipt["bootstrap_sha256"]
            or boot_receipt["source_sha256"] != config["source_model_sha256"]):
        raise RuntimeError("H15 bootstrap model provenance changed")

    completed: list[dict[str, Any]] = []
    teacher = bootstrap
    for round_number in range(1, args.rounds + 1):
        folder = out / f"round-{round_number:04d}"
        partial = out / f"round-{round_number:04d}.partial"
        if folder.exists():
            report = json.loads((folder / "round.json").read_text())
            checkpoint = folder / "student.pt"
            if (not checkpoint.is_file()
                    or report["updated_sha256"] != sha256(checkpoint)
                    or report["source_teacher_sha256"] != sha256(teacher)):
                raise RuntimeError(f"Inconsistent H15 completed round {round_number}")
            completed.append(report)
            teacher = checkpoint
            print(f"[h15-reference-resume] completed_round={round_number} "
                  f"teacher={sha256(teacher)}", flush=True)
            continue
        partial.mkdir(parents=True, exist_ok=True)
        train_ids = [
            f"{args.seed_prefix}:train:r{round_number:04d}:i{i:05d}"
            for i in range(args.train_combats)
        ]
        started = time.monotonic()
        train_records = collect_cases(
            config, teacher, train_ids, "train", partial / "train",
        )
        samples = write_training_targets(
            train_records, partial / "targets.jsonl",
        )
        atomic_json(partial / "collection.json", {
            "round": round_number, "model_sha256": sha256(teacher),
            "targets_sha256": sha256(partial / "targets.jsonl"),
            "target_rows": samples, "collection": aggregate(train_records),
        })
        updated = partial / "student.pt"
        if not updated.exists():
            run_optimizer(config, teacher, partial / "targets.jsonl", updated)
        optimizer = verify_distillation_receipt(
            updated, initial_sha=sha256(teacher),
            source_sha=sha256(partial / "targets.jsonl"),
            expected_steps=args.epochs * math.ceil(samples / args.batch_size),
        )
        delta = verify_parameter_update(teacher, updated)
        evaluation: dict[str, Any] = {}
        if round_number % args.eval_every == 0 or round_number == args.rounds:
            holdout_ids = [
                f"{args.seed_prefix}:heldout:i{i:05d}"
                for i in range(args.eval_combats)
            ]
            frozen_report = out / "frozen-h9-eval.json"
            if frozen_report.exists():
                evaluation["frozen_h9"] = json.loads(frozen_report.read_text())
            else:
                baseline_records = collect_cases(
                    config, initial, holdout_ids, "greedy",
                    out / "frozen-h9-eval-parts",
                )
                evaluation["frozen_h9"] = aggregate(baseline_records)
                atomic_json(frozen_report, evaluation["frozen_h9"])
            evaluation["h15_raw"] = aggregate(collect_cases(
                config, updated, holdout_ids, "greedy",
                partial / "eval-raw",
            ))
            evaluation["h15_puct"] = aggregate(collect_cases(
                config, updated, holdout_ids, "search",
                partial / "eval-search",
            ))
            print("[h15-reference-eval] " + json.dumps({
                "round": round_number, "evaluation": evaluation,
            }, sort_keys=True), flush=True)
        result = {
            "round": round_number, "status": "complete",
            "source_teacher_sha256": sha256(teacher),
            "updated_sha256": sha256(updated),
            "source_combat_count": len(train_records),
            "target_rows": samples, "collection": aggregate(train_records),
            "training_fit": optimizer,
            "parameter_update": delta, "evaluation": evaluation,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(partial / "round.json", result)
        os.replace(partial, folder)
        teacher = folder / "student.pt"
        completed.append(result)
        atomic_json(out / "progress.json", {
            "schema": SCHEMA, "completed_rounds": len(completed),
            "latest_model": str(teacher), "latest_model_sha256": sha256(teacher),
            "rounds": [{
                "round": item["round"], "target_rows": item["target_rows"],
                "in_sample_policy_kl": item["training_fit"][
                    "in_sample_fit_after"]["policy_kl_to_search"],
                "mean_reward": item["collection"]["mean_reward"],
                "evaluation": item["evaluation"],
            } for item in completed],
        })
        print("[h15-reference-round] " + json.dumps({
            "round": round_number, "teacher": sha256(teacher),
            "targets": samples, "train_mean_reward": result["collection"]["mean_reward"],
            "training_fit": optimizer, "elapsed_seconds": result["elapsed_seconds"],
        }, sort_keys=True), flush=True)
    print("[h15-reference-complete] " + json.dumps({
        "rounds": len(completed), "latest_checkpoint": str(teacher),
        "latest_sha256": sha256(teacher),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
