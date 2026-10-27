#!/usr/bin/env python3
"""Verify identical draw-only transcripts in 1 vs 15 independent C# processes.

This is an integration/replay gate, not an H1 learning-effect experiment.
Each independent Python process owns one separate emulator JSONL process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", required=True, type=Path)
    parser.add_argument("--emulator-bindings", required=True, type=Path)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--baseline-learned", required=True, type=Path)
    parser.add_argument("--baseline-reference", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/nature-h1-worker-gate"))
    args = parser.parse_args()
    if args.workers != 15:
        parser.error("H1 process-isolation acceptance uses exactly 15 workers")
    baselines = {
        mode: json.loads(path.read_text(encoding="utf-8"))
        for mode, path in (
            ("learned", args.baseline_learned),
            ("reference", args.baseline_reference),
        )
    }
    for mode, result in baselines.items():
        if (result.get("draw_mode") != mode
                or result.get("emulator_revision") != args.emulator_revision
                or result.get("nontrivial_draw_count", 0) < 1
                or result.get("trace_digest") != result.get("replay_digest")):
            raise RuntimeError(f"Invalid single-process {mode} reference probe")
    if (baselines["learned"]["source_opening_fingerprint"] !=
            baselines["reference"]["source_opening_fingerprint"]):
        raise RuntimeError("Draw arms started from different source openings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, Path, subprocess.Popen[str]]] = []
    for idx in range(args.workers):
        mode = "learned" if idx % 2 == 0 else "reference"
        output = args.output_dir / f"{idx:02d}-{mode}.json"
        if output.exists():
            raise RuntimeError(f"Refusing to reuse worker output: {output}")
        command = [
            sys.executable, "-u", "tools/probe_nature_draw_only.py",
            "--emulator-dll", str(args.emulator_dll),
            "--emulator-bindings", str(args.emulator_bindings),
            "--emulator-revision", args.emulator_revision,
            "--source-seed", "nature-p0-weak-real-victory",
            "--entry-mode", "seeded-fixed-shrinker",
            "--draw-mode", mode,
            "--region", "Overgrowth",
            "--min-draw-choices", "1",
            "--output", str(output),
        ]
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((mode, output, proc))
    if len({process.pid for _, _, process in procs}) != args.workers:
        raise RuntimeError("Not all worker processes were independent")
    failures = []
    for mode, output, process in procs:
        try:
            log, _ = process.communicate(timeout=240)
        except subprocess.TimeoutExpired:
            process.kill()
            log, _ = process.communicate()
            failures.append(f"{mode} process {process.pid} timed out: {log[-5000:]}")
            continue
        if process.returncode != 0:
            failures.append(f"{mode} process {process.pid} failed: {log[-5000:]}")
            continue
        result = json.loads(output.read_text(encoding="utf-8"))
        baseline = baselines[mode]
        fields = (
            "trace_digest", "replay_digest", "source_opening_fingerprint",
            "first_public_fingerprint", "outcome", "exit_hp",
            "controlled_draw_count", "nontrivial_draw_count",
            "decisions",
        )
        different = [key for key in fields if result.get(key) != baseline.get(key)]
        if different:
            failures.append(
                f"{mode} process {process.pid}: mismatched fields {different}"
            )
        print(
            f"[nature-h1 workers] process={process.pid} mode={mode} "
            f"terminal={result['outcome']} draws={result['nontrivial_draw_count']} "
            f"exact={not different}",
            flush=True,
        )
    if failures:
        raise RuntimeError("1-vs-15 worker determinism gate failed:\n"
                           + "\n".join(failures))
    print(
        f"[nature-h1 workers] PASS 1-versus-{args.workers} "
        "isolated independent C# processes; both draw policies and "
        "complete transcripts exactly repeat",
        flush=True,
    )


if __name__ == "__main__":
    main()
