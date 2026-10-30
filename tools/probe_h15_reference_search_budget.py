#!/usr/bin/env python3
"""Matched H15 ReferenceNature PUCT budget probe on an existing M15 holdout.

Reuses the saved frozen updated H15 checkpoint, public experiment fixtures and
source Nature seeds. CPU workers run independent complete reference combats.
No optimizer step or original-game chance-law parity is involved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sts2_ai.training.h15_search_supervision import decode_target

SCHEMA = "sts2-h15-reference-puct-budget-probe-v1"


def summarize_search_q(records: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        diagnostic
        for record in records
        for diagnostic in record["searched_root_diagnostics"]
    ]
    if len(diagnostics) != len(records):
        raise ValueError("Probe requires exactly one real searched root per combat")
    spreads = []
    for diagnostic in diagnostics:
        visits = diagnostic["root_visits"]
        values = diagnostic["root_mean_action_values"]
        if (len(visits) != len(values) or
                sum(visits) != diagnostic["search_simulations"]):
            raise ValueError("Inconsistent H15 root value/visit telemetry")
        selected_q = [q for n, q in zip(visits, values, strict=True) if n > 0]
        if (not selected_q or any(
            not isinstance(q, (int, float)) or not math.isfinite(q)
            for q in selected_q
        )):
            raise ValueError("Invalid selected H15 action value estimate")
        spreads.append(max(selected_q) - min(selected_q))
    return {
        "mean_visited_action_q_spread": sum(spreads) / len(spreads),
        "mean_expanded_public_histories": sum(
            d["search_expanded_information_sets"] for d in diagnostics
        ) / len(diagnostics),
        "mean_simulated_transitions": sum(
            d["search_transitions"] for d in diagnostics
        ) / len(diagnostics),
        "total_neural_cutoffs": sum(
            d["search_neural_cutoffs"] for d in diagnostics
        ),
        "total_terminal_evaluations": sum(
            d["search_terminal_evaluations"] for d in diagnostics
        ),
    }


def validate_budgets(raw: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(v.strip()) for v in raw.split(","))
    except ValueError as exc:
        raise ValueError("Budgets must be comma-separated positive integers") from exc
    if (not budgets or any(b < 1 for b in budgets)
            or len(budgets) != len(set(budgets))
            or list(budgets) != sorted(budgets)):
        raise ValueError("Budgets must be unique positive integers in ascending order")
    return budgets


def main() -> None:
    # The executable lives alongside the existing M15 helpers under tools/.
    # Import at CLI entry so static telemetry tests remain importable as
    # tools.probe_h15_reference_search_budget from the project root.
    from audit_h15_reference_search import compare_saved_evaluations, read_cases
    from train_h15_reference_rounds import (
        aggregate,
        atomic_json,
        collect_cases,
        sha256,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True,
                   help="Existing completed M15 training output (immutable)")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--output-root", type=Path, required=True,
                   help="New separate experiment directory")
    p.add_argument("--budgets", default="16,64,128")
    p.add_argument("--cases", type=int, default=15)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if min(args.round, args.cases, args.workers) < 1:
        p.error("Positive round/cases/workers required")
    try:
        budgets = validate_budgets(args.budgets)
    except ValueError as exc:
        p.error(str(exc))
    original = args.source_run.resolve()
    folder = original / f"round-{args.round:04d}"
    checkpoint = folder / "student.pt"
    source_config_file = original / "config.json"
    source_config = json.loads(source_config_file.read_text(encoding="utf-8"))
    manifest = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    model_hash = sha256(checkpoint)
    if manifest["updated_sha256"] != model_hash:
        raise ValueError("Source M15 checkpoint hash disagrees with completed round")
    if args.cases > source_config["eval_combats"]:
        p.error("Requested more cases than original source holdout cohort")
    ids = [
        f"{source_config['seed_prefix']}:heldout:i{i:05d}"
        for i in range(args.cases)
    ]
    frozen_raw = read_cases(
        folder / "eval-raw", mode="greedy", model_sha256=model_hash,
    )
    if not set(ids).issubset(frozen_raw):
        raise ValueError("Requested heldout cases missing from original raw evaluation")
    paired_raw = {k: frozen_raw[k] for k in ids}
    config = dict(source_config)
    config["workers"] = args.workers
    config["max_searches"] = 1
    config["max_decisions"] = source_config["max_decisions"]
    config["code_sha256"] = {
        name: sha256(Path(__file__).resolve().parents[1] / name)
        for name in (
            "tools/probe_h15_reference_search_budget.py",
            "tools/train_h15_reference_rounds.py",
            "src/sts2_ai/training/h15_native_search_collector.py",
            "src/sts2_ai/search/combat_puct.py",
            "src/sts2_ai/search/h15_public_combat_adapter.py",
            "src/sts2_ai/emulator/h15_h8_reference.py",
            "src/sts2_ai/models/h15_policy_value.py",
        )
    }
    experiment = {
        "schema": SCHEMA, "reference_prior": "fixed-h8-fixture-declared-chance-law",
        "original_config_sha256": sha256(source_config_file),
        "original_round": args.round, "model_sha256": model_hash,
        "cases": ids, "budgets": list(budgets),
        "code_sha256": config["code_sha256"], "emulator_revision": config["emulator_revision"],
        "emulator_dll_sha256": config["emulator_dll_sha256"],
        "workers": args.workers,
    }
    print("[h15-budget-preflight] " + json.dumps(experiment, sort_keys=True), flush=True)
    if not args.execute:
        print("[h15-budget-preflight] pass --execute to collect real reference combats")
        return
    out = args.output_root.resolve()
    if out == original or original in out.parents or out in original.parents:
        raise ValueError("Budget probe output must be separate from completed source run")
    out.mkdir(parents=True, exist_ok=True)
    record_file = out / "experiment.json"
    if record_file.exists():
        if json.loads(record_file.read_text(encoding="utf-8")) != experiment:
            raise ValueError("Cannot resume a differently configured H15 budget probe")
    else:
        atomic_json(record_file, experiment)
    results: list[dict[str, Any]] = []
    for budget in budgets:
        case_config = dict(config)
        case_config["simulations"] = budget
        directory = out / f"sim-{budget:04d}"
        records = collect_cases(
            case_config, checkpoint, ids, "search", directory / "eval-search",
        )
        # Each target must remain grounded in an independently played combat
        # with exactly this search budget and a pinned model identity.
        for record in records:
            if len(record["targets"]) != 1:
                raise ValueError("Expected exactly one searched root per reference fight")
            target = record["targets"][0]
            sample = decode_target(target)
            if (target["search_simulations"] != budget
                    or sample.search_model_sha256 != model_hash
                    or sample.source_seed != record["case_id"]):
                raise ValueError("Stale or incorrectly budgeted H15 search target")
        matches = compare_saved_evaluations(
            paired_raw, {r["case_id"]: r for r in records},
        )
        data = {
            "simulations": budget, "case_ids": ids,
            "search": aggregate(records),
            "root_q": summarize_search_q(records),
            "paired_vs_raw": {
                key: value for key, value in matches.items() if key != "cases"
            },
        }
        atomic_json(directory / "budget.json", data)
        results.append(data)
        atomic_json(out / "progress.json", {
            "schema": SCHEMA, "completed_budgets": len(results),
            "results": results,
        })
        print("[h15-budget-result] " + json.dumps(data, sort_keys=True), flush=True)
    print("[h15-budget-complete] " + json.dumps({
        "budgets": list(budgets), "source_sha256": model_hash,
        "output_root": str(out),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
