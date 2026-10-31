#!/usr/bin/env python3
"""Summarize completed R10/R30 paired public traces without launching PUCT.

Runs directly on the worker-*.json trace chunks saved before the expensive
search phase. Safe after interrupting the original diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_h15_m17_r10_r30 import make_report  # noqa: E402


def summarize(cfg: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    all_cases: dict[str, dict[str, Any]] = {}
    hashes = cfg["checkpoint_hashes"]
    for chunk in chunks:
        if (chunk.get("stage") != "traces"
                or chunk.get("checkpoint_hashes") != hashes
                or len(chunk.get("cases", [])) != len(chunk.get("case_ids", []))):
            raise ValueError("Mixed or incomplete paired trace worker chunk")
        for case in chunk["cases"]:
            key = case["case_id"]
            if key in all_cases or key not in chunk["case_ids"]:
                raise ValueError("Duplicate or unexpected source case ID")
            all_cases[key] = case
    expected = [
        f"{cfg['seed_prefix']}:heldout:i{i:05d}"
        for i in range(cfg["cases"])
    ]
    if set(all_cases) != set(expected):
        missing = set(expected) - set(all_cases)
        raise ValueError(f"Incomplete paired traces: {len(missing)} missing cases")
    source = {"fixture_seed": cfg["fixture_seed"],
              "seed_prefix": cfg["seed_prefix"]}
    report = make_report([all_cases[x] for x in expected], [], source, hashes)
    report["search_stage"] = "not_run"
    report["interpretation_note"] = (
        "Policy/value disagreement on identical public states and action "
        "menus, without a search-quality comparison. The saved traces do "
        "not determine whether either policy is optimal."
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--paired-root", type=Path, required=True)
    args = p.parse_args()
    run = args.run_root.resolve()
    paired = args.paired_root.resolve()
    cfg = json.loads((paired / "config.json").read_text(encoding="utf-8"))
    if cfg["run_root"] != str(run):
        p.error("Paired configuration points to a different source run")
    source_cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if cfg["seed_prefix"] != source_cfg["seed_prefix"]:
        p.error("Source run and paired trace seed prefixes disagree")
    cfg["fixture_seed"] = source_cfg["fixture_seed"]
    files = sorted((paired / "traces").glob("worker-*.json"))
    if not files:
        p.error("Missing saved paired trace workers")
    chunks = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    report = summarize(cfg, chunks)
    output = paired / "trace-only-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("[h15-paired-traces-summary] " + json.dumps({
        "replayed": report["holdout_cases_replayed"],
        "r10_wins": report["r10_wins"],
        "r30_wins": report["r30_wins"],
        "regressed_cases": report["regressed_cases_count"],
        "recovered_cases": report["improved_cases_count"],
        "public_decisions_compared": report["public_decisions_paired"],
        "raw_disagreement_fraction": report["raw_action_disagreement_fraction"],
        "mean_policy_js_nats": report["mean_policy_js_nats"],
        "path": str(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
