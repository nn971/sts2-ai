"""Run source-identical H8 actor batching-window sweeps and validate parity.

Run against the same pinned M3 emulator and frozen H7 policy. The default
six-leg design repeats the baseline at both ends to expose GPU/host drift.
The long plan adds additional max-batch-limited configurations. Both plans
compare exact source, Nature, and replay journals before ranking throughput.

This is an explicit foreground command, never a background workload.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compare_h8_actor_batch_windows import compare

SHORT = ((16, 3.0), (16, 0.0), (16, 1.0),
         (16, 6.0), (8, 3.0), (16, 3.0))
FULL = ((16, 3.0), (16, 0.0), (16, 1.0), (16, 6.0),
        (8, 3.0), (8, 1.0), (4, 1.0), (16, 3.0))
SMOKE = ((2, 3.0), (1, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-dll", type=Path, required=True)
    ap.add_argument("--emulator-bindings", type=Path, required=True)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--player-checkpoint", type=Path)
    ap.add_argument("--actor-device", default="cuda")
    ap.add_argument("--actor-collation", default="packed",
                    choices=("legacy", "packed", "packed-contiguous"))
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--max-boundaries", type=int, default=1200)
    ap.add_argument("--warmup-per-worker", type=int, default=2)
    ap.add_argument("--replay-every", type=int, default=16)
    ap.add_argument("--plan", default="short", choices=("short", "full", "smoke"))
    ap.add_argument("--resume", action="store_true",
                    help="Reuse previous reports only after source parity "
                         "and frozen config validation by the comparator")
    ap.add_argument("--allow-censored", action="store_true",
                    help="CI source parity only; no completed-combat claim")
    args = ap.parse_args()
    if args.workers < 1 or args.episodes < 16:
        ap.error("Require positive worker count and at least 16 source episodes")
    if args.warmup_per_worker < 1:
        ap.error("Steady-state batch trace needs positive per-worker warmup")

    plan = {"short": SHORT, "full": FULL, "smoke": SMOKE}[args.plan]
    pilot = Path(__file__).with_name("benchmark_h8_h7_policy_pilot.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labeled = []
    for i, (max_batch, delay_ms) in enumerate(plan, 1):
        label = f"{i:02d}-batch{max_batch}-delay{delay_ms:g}ms"
        target = args.output_dir / (label + ".json")
        log = args.output_dir / (label + ".log")
        command = [
            sys.executable, "-u", str(pilot),
            "--emulator-dll", str(args.emulator_dll),
            "--emulator-bindings", str(args.emulator_bindings),
            "--emulator-revision", args.emulator_revision,
            "--output", str(target),
            "--order", "native", "--episodes", str(args.episodes),
            "--workers", str(args.workers),
            "--warmup-per-worker", str(args.warmup_per_worker),
            "--max-boundaries", str(args.max_boundaries),
            "--replay-every", str(args.replay_every),
            "--actor-device", args.actor_device,
            "--actor-collation", args.actor_collation,
            "--actor-max-batch", str(max_batch),
            "--actor-delay-ms", str(delay_ms),
            "--actor-batch-trace", "--sampled-mechanics-audit",
        ]
        if args.player_checkpoint:
            command += ["--player-checkpoint", str(args.player_checkpoint)]
        if target.exists():
            if not args.resume:
                raise FileExistsError(
                    f"{target} already exists; use a new output directory "
                    "or explicit --resume")
            print(f"[h8-batch-window] reuse {label}", flush=True)
        else:
            print(
                f"[h8-batch-window] {i}/{len(plan)} "
                f"{label} effective_cap={min(max_batch, args.workers)}",
                flush=True,
            )
            with log.open("w", encoding="utf-8") as sink:
                with subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                ) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        sink.write(line)
                    if process.wait():
                        raise RuntimeError(
                            f"{label} failed with exit code {process.returncode}; "
                            f"see {log}")
            if not target.is_file():
                raise RuntimeError("H8 benchmark did not write " + str(target))
        report = json.loads(target.read_text())
        if (int(report.get("actor_max_batch", -1)) != max_batch or
                float(report.get("actor_delay_ms", -1.0)) != delay_ms or
                report.get("actor_device") != args.actor_device or
                report.get("actor_collation") != args.actor_collation or
                report.get("emulator_revision") != args.emulator_revision or
                len(report.get("cohorts", [])) != 1 or
                report["cohorts"][0].get("episodes") != args.episodes or
                report["cohorts"][0].get("workers") !=
                min(args.workers, args.episodes)):
            raise ValueError(
                "Saved H8 batch-window run has different configuration: "
                + str(target))
        labeled.append((label, report))
    summary = compare(labeled, allow_censored=args.allow_censored)
    summary["plan"] = args.plan
    summary["report_paths"] = {
        name: str(args.output_dir / (name + ".json"))
        for name, _ in labeled
    }
    destination = args.output_dir / "sweep-summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print("[h8-batch-window] source/Nature/replay parity passed", flush=True)
    for row in summary["runs"]:
        print(
            f"  {row['label']:27s} "
            f"{row['completed_combats_per_second']:7.3f} combats/s  "
            f"mean_batch={row['mean_batch_size']:4.2f}  "
            f"queue_to_batch={row['queue_to_batch_mean_ms']:5.2f}ms  "
            f"fill={row['batch_fill_ms_per_batch']:5.2f}ms  "
            f"forward={row['forward_ms_per_batch']:5.2f}ms",
            flush=True,
        )
    print("[h8-batch-window] report:", destination, flush=True)


if __name__ == "__main__":
    main()
