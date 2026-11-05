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
        or getattr(args, "max_simulation_steps", 512) < 1
        or args.simulations not in (8, 16, 64)
        or args.depth < 0 or args.actor_max_delay_ms < 0
        or not 0 <= args.horizon_lambda <= 1
        or args.macro_prior_weight < 0
        or (args.macro_prior_weight and args.macro_model is None)
        or not 0 <= args.ascension <= 10
        or not args.seed_prefix.strip()
        or getattr(args, "h21_max_batch", 4) not in (1, 4, 8, 16)
        or (getattr(args, "actor_deterministic_cuda", False)
            and (getattr(args, "experiment", "batching") != "h21-collator"
                 or not args.actor_device.startswith("cuda")))
    ):
        raise ValueError("Invalid P2 A/B/B/A benchmark configuration")
    if getattr(args, "require_full_act", False) and (
        getattr(args, "experiment", "batching") != "h21-collator"
        or args.actor_device != "cuda" or args.workers < 15
        or len(seeds) < 16 or args.max_decisions < 1024
        or args.max_searches < 1
    ):
        raise ValueError(
            "Full-Act H21 acceptance requires CUDA, >=15 workers, >=16 "
            "independent seeds, >=1024 decision limit and real PUCT searches"
        )
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
        "--max-simulation-steps", str(getattr(args, "max_simulation_steps", 512)),
        "--max-searches", str(args.max_searches),
        "--max-decisions", str(args.max_decisions),
        "--paired",
        "--no-episode-profile",
        *(
            ["--actor-deterministic-cuda"]
            if getattr(args, "actor_deterministic_cuda", False) else []
        ),
        "--execute",
    ]
    if args.macro_model is not None:
        common.extend(("--macro-model", str(args.macro_model)))
    experiment = getattr(args, "experiment", "batching")
    if experiment not in ("batching", "actor-profile", "prefix-cache",
                        "transport", "wire-profile", "native-fused-profile",
                        "h21-collator"):
        raise ValueError("Invalid actor A/B/B/A experiment")
    modes = (
        ORDER if experiment == "batching" else
        ((("A1", args.h21_max_batch, 0), ("B1", args.h21_max_batch, 0),
          ("B2", args.h21_max_batch, 0), ("A2", args.h21_max_batch, 0))
         if experiment == "h21-collator" else
         (("A1", 16, 4), ("B1", 16, 4),
          ("B2", 16, 4), ("A2", 16, 4)))
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
             ["--representation", "h21", "--actor-h21-collator",
              "packed" if name in ("B1", "B2") else "reference"]
             if experiment == "h21-collator" else []
         ),
         *(
             ["--actor-profile"]
             if experiment == "actor-profile" and name in ("B1", "B2")
             else []
         ),
         *(
             ["--wire-profile-only"]
             if experiment == "wire-profile" and name in ("B1", "B2")
             else []
         ),
         *(
             ["--native-fused-profile-only"]
             if experiment == "native-fused-profile"
             and name in ("B1", "B2") else []
         ),
         "--p2-prefix-cache-handles",
         str(64 if experiment == "prefix-cache"
             and name in ("B1", "B2") else 0),
         "--rollout-transport", (
             "legacy" if experiment == "transport" and name in ("A1", "A2")
             else "fused"
         ),
         "--output", str(output)]
        for (name, max_batch, fixed_size), output in zip(
            modes, files, strict=True
        )
    ]
    report = args.output_root / (
        "actor-profile-overhead-abba-report.json"
        if experiment == "actor-profile" else
        "p2-prefix-cache-abba-report.json"
        if experiment == "prefix-cache" else
        "p2-transport-abba-report.json"
        if experiment == "transport" else
        "p2-wire-profile-abba-report.json"
        if experiment == "wire-profile" else
        "p2-native-fused-profile-abba-report.json"
        if experiment == "native-fused-profile"
        else "p2-h22-collator-abba-report.json"
        if experiment == "h21-collator"
        else "actor-abba-report.json"
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
        *(
            ["--require-complete", "--min-seed-count", "16",
             "--require-cuda", "--require-search-roots"]
            if getattr(args, "require_full_act", False) else []
        ),
        "--output", str(report),
    ])
    return calls, [*files, report]


def preflight_checkpoint(args: argparse.Namespace) -> None:
    """Fail before A1 if the frozen checkpoint is missing or wrong-schema."""
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_file():
        raise ValueError(
            f"Missing frozen checkpoint: {checkpoint}. "
            "Run tools/create_p2_h22_frozen_checkpoint.py --initial-h16 "
            "<existing-H16-joint.pt> --output <H22-output.pt> first, or "
            "pass --checkpoint pointing to an existing H22 checkpoint."
        )
    if getattr(args, "experiment", "batching") == "h21-collator":
        from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet

        try:
            model = JointActPilotNet.load(checkpoint, device="cpu")
        except (RuntimeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Cannot load H22 benchmark checkpoint: {checkpoint}: {exc}"
            ) from exc
        if model.representation != "h21":
            raise ValueError(
                f"Checkpoint uses {model.representation}, expected H22 H21 v4: "
                f"{checkpoint}. Create a frozen H22 warm-start first."
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="Perform all four runs; otherwise preview commands")
    p.add_argument("--actor-deterministic-cuda", action="store_true",
                   help="Opt-in deterministic PyTorch CUDA graph aggregation in all "
                        "four H22 arms; use for reproducibility diagnosis")
    p.add_argument("--require-full-act", action="store_true",
                   help="H21 CUDA/15-worker/16-seed full-Act strict gate; "
                        "reject censored episodes and missing real PUCT")
    p.add_argument("--experiment", choices=("batching", "actor-profile", "prefix-cache",
                            "transport", "wire-profile",
                            "native-fused-profile", "h21-collator"),
                   default="batching",
                   help="Batching, actor profile, exact prefix, or JSONL fused/legacy transport")
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
    p.add_argument("--h21-max-batch", type=int, choices=(1, 4, 8, 16),
                   default=4, help="Matched H21 dynamic actor batch cap")
    p.add_argument("--seeds",
                   help="Optional comma-separated independent seed cohort")
    p.add_argument("--seed-prefix", default="p2-a0-abba-independent-20261104")
    p.add_argument("--seed-count", type=int, default=32)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--map-profile", choices=("generated", "fixed-route"),
                   default="generated")
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--max-simulation-steps", type=int, default=512,
                   help="Maximum actions in one full-depth combat simulation; "
                        "larger values permit long fights but never substitute "
                        "a value or a defeat at the limit")
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
        try:
            preflight_checkpoint(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
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
            "P2.1 A1/A2 prefix cache off; B1/B2 bounded cache on; "
            "actor/search/wire profiling off; strict B/B replay"
            if args.experiment == "prefix-cache" else
            "P2.2 A1/A2 legacy split JSONL; B1/B2 fused JSONL; "
            "cache/actor/search/wire profiling off; exact teacher replay"
            if args.experiment == "transport" else
            "P2.2 fused in all arms: A1/A2 wire profiling off; B1/B2 "
            "wire-only breakdown on; actor/PUCT profiling off"
            if args.experiment == "wire-profile" else
            "P2.2 fused in all arms: A1/A2 native profiler off; "
            "B1/B2 C# fused-only timers on; wire/actor/PUCT profiling off"
            if args.experiment == "native-fused-profile" else
            "H22 public-v1 reference/packed/packed/reference H21 collation; "
            "dynamic shape, same actor batch cap and checkpoint; "
            "all profiling disabled; strict public/teacher replay"
            if args.experiment == "h21-collator" else
            "actor/search/wire profiling disabled; strict B/B replay"
        ),
        "seed_count": len(args.seeds.split(",")) if args.seeds else args.seed_count,
        "commands": [shlex.join(call) for call in calls],
        "execute": args.execute,
        "require_full_act": args.require_full_act,
        "actor_deterministic_cuda": args.actor_deterministic_cuda,
        "max_simulation_steps": args.max_simulation_steps,
    }, indent=2), flush=True)
    if args.execute:
        for index, call in enumerate(calls, start=1):
            print(f"[p2-abba] step {index}/{len(calls)}", flush=True)
            subprocess.run(call, cwd=args.ai_root, check=True)


if __name__ == "__main__":
    main()
