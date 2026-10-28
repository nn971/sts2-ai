#!/usr/bin/env python3
"""Paired native H10 combat benchmark: frozen legacy H6 combat vs H9 Torch.

Replays the *exact* public source-run action prefix and verifies both public
and exact-state hashes before forking each authentic combat entry. The agents
see only fair observations and legal actions. This is a tactical comparison,
not a full-run Act-clear estimate or proof that either policy is optimal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from snapshot_eval_runtime import evaluate_paired_snapshots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--macro-model", type=Path, required=True,
                   help="H6-C JSON: H6 tactical baseline and H10 frozen macro")
    p.add_argument("--combat-checkpoint", type=Path, required=True,
                   help="Frozen H9 Torch checkpoint (.pt)")
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1,
                   help="Native emulator processes, one H9 model per worker")
    p.add_argument("--combat-device", choices=("cpu", "cuda"), default="cpu",
                   help="Use cpu for multiworker runs to avoid GPU duplication")
    p.add_argument("--max-combat-decisions", type=int, default=512)
    p.add_argument("--progress-every", type=int, default=64)
    p.add_argument("--limit-source-runs", type=int)
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    if args.workers > 1 and args.combat_device == "cuda":
        p.error("Multiple independent CUDA model copies risk exhausting GPU memory; use --combat-device cpu or --workers 1")
    report = evaluate_paired_snapshots(
        corpus=args.corpus,
        baseline_model=args.macro_model,
        candidate_model=args.combat_checkpoint,
        report_path=args.report,
        workers=args.workers,
        max_combat_decisions=args.max_combat_decisions,
        progress_every=args.progress_every,
        limit_source_runs=args.limit_source_runs,
        build=args.build,
        candidate_kind="h10_hybrid",
        candidate_macro_model=args.macro_model,
        combat_device=args.combat_device,
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
    )
    print(json.dumps({
        "report": str(args.report),
        "emulator_revision": report["emulator_revision"],
        "evaluation": report["evaluation"],
        "summary": report["summary"],
        "by_tier": report["by_tier"],
    }, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
