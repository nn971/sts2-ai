#!/usr/bin/env python3
"""Run pinned P2 actor A1/B1/B2/A2 cohorts sequentially, then audit parity.

Uses the existing H16-compatible emulator and frozen model. Refuses to reuse
any output directory containing a completed cohort. No training, no model
weights or PUCT configuration are changed. Each arm uses 15 emulator workers
by default and two uninstrumented within-seed replay executions.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = (("A1", 1, 0), ("B1", 16, 4), ("B2", 16, 4), ("A2", 1, 0))


def make_commands(args: argparse.Namespace) -> tuple[list[list[str]], list[Path]]:
    seeds = (
        [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
        if args.seeds else [
            f"{args.seed_prefix}-{i}" for i in range(args.seed_count)
        ]
    )
    if (
        not seeds or len(set(seeds)) != len(seeds)
        or args.workers < 1 or args.seed_count < 1
        or args.max_decisions < 1 or args.max_searches < 0
        or args.simulations not in (8, 16, 64)
        or args.depth < 0 or args.actor_max_delay_ms < 0
        or not 0 <= args.horizon_lambda <= 1
        or args.macro_prior_weight < 0
        or (args.macro_prior_weight and args.macro_model is None)
        or not 0 <= args.ascension <= 10
        or not args.seed_prefix.strip()
    ):
        raise ValueError("Invalid P2 A/B/B/A benchmark configuration")
    all_seeds = ",".join(seeds)
    common = [
        "--ai-root", str(args.ai_root),
        "--emulator-root", str(args.emulator_root),
        "--emulator-revision", args.emulator_revision,
        "--checkpoint", str(args.checkpoint),
        "--macro-prior-weight", str(args.macro_prior_weight),
        "--horizon-lambda", str(args.horizon_lambda),
        "--workers", str(args.workers),
        "--actor-inference", "shared",
        "--actor-device", args.actor_device,
        "--actor-max-delay-ms", str(args.actor_max_delay_ms),
        "--seeds", all_seeds,
        "--ascension", str(args.ascension),
        "--map-profile", args.map_profile,
        "--simulations", str(args.simulations),
        "--depth", str(args.depth),
        "--max-searches", str(args.max_searches),
        "--max-decisions", str(args.max_decisions),
        "--paired",
        "--no-episode-profile",
        "--execute",
    ]
    if args.macro_model is not None:
        common.extend(("--macro-model", str(args.macro_model)))
    experiment = getattr(args, "experiment", "batching")
    if experiment not in ("batching", "actor-profile"):
        raise ValueError("Invalid actor A/B/B/A experiment")
    modes = (
        ORDER if experiment == "batching" else
        (("A1", 16, 4), ("B1", 16, 4),
         ("B2", 16, 4), ("A2", 16, 4))
    )
    files = [
        args.output_root / name / "profile.json" for name, _, _ in modes
    ]
    calls = [
        [sys.executable, str(ROOT / "tools/profile_p2_parallel_cohort.py"),
         *common,
         "--actor-max-batch", str(max_batch),
         "--actor-fixed-shape-batch-size", str(fixed_size),
         *(
             ["--actor-profile"]
             if experiment == "actor-profile" and name in ("B1", "B2")
             else []
         ),
         "--output", str(output)]
        for (name, max_batch, fixed_size), output in zip(
            modes, files, strict=True
        )
    ]
    report = args.output_root / (
        "actor-profile-overhead-abba-report.json"
        if experiment == "actor-profile" else "actor-abba-report.json"
    )
    calls.append([
        sys.executable, str(ROOT / "tools/report_p2_actor_abba.py"),
        *(
            element
            for flag, path in zip(("--a1", "--b1", "--b2", "--a2"), files,
                                  strict=True)
            for element in (flag, str(path))
        ),
        "--experiment", experiment,
        "--output", str(report),
    ])
    return calls, [*files, report]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="Perform all four runs; otherwise preview commands")
    p.add_argument("--experiment", choices=("batching", "actor-profile"),
                   default="batching",
                   help="Batch-1 versus fixed4, or fixed4 actor profiling on/off")
    p.add_argument("--ai-root", type=Path, default=ROOT)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--macro-prior-weight", type=float, default=0.5)
    p.add_argument("--horizon-lambda", type=float, default=0.411395)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--actor-device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--actor-max-delay-ms", type=float, default=2.0)
    p.add_argument("--seeds",
                   help="Optional comma-separated independent seed cohort")
    p.add_argument("--seed-prefix", default="p2-a0-abba-independent-20261104")
    p.add_argument("--seed-count", type=int, default=32)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--map-profile", choices=("generated", "fixed-route"),
                   default="generated")
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--max-searches", type=int, default=2)
    p.add_argument("--max-decisions", type=int, default=80,
                   help="80 is capped throughput; set higher for full episodes")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    try:
        calls, outputs = make_commands(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.execute:
        existing = [str(output) for output in outputs if output.exists()]
        if existing:
            raise SystemExit(
                "P2 A/B/B/A refuses to overwrite existing evidence: "
                + ", ".join(existing)
            )
    print(json.dumps({
        "order": [a for a, _, _ in ORDER],
        "experiment": args.experiment,
        "instrumentation": (
            "A1/A2 actor profile off; B1/B2 actor profile on; "
            "search/wire profile off; strict B/B replay"
            if args.experiment == "actor-profile" else
            "actor/search/wire profiling disabled; strict B/B replay"
        ),
        "seed_count": len(args.seeds.split(",")) if args.seeds else args.seed_count,
        "commands": [shlex.join(call) for call in calls],
        "execute": args.execute,
    }, indent=2), flush=True)
    if args.execute:
        for index, call in enumerate(calls, start=1):
            print(f"[p2-abba] step {index}/{len(calls)}", flush=True)
            subprocess.run(call, cwd=args.ai_root, check=True)


if __name__ == "__main__":
    main()
