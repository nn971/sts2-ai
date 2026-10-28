#!/usr/bin/env python3
"""Interleaved A-B-B-A H8 emulator comparison with one frozen real H7 actor.

Build two pinned emulator revisions from temporary git archives (no git
checkout, worktree, or edits to the user's live repositories), then run four
128-source, 15-worker, same-seed CUDA H8+legacy-parity+PPO benchmarks.

Neither the sequential actor microbenchmark nor a single before/after pair
can isolate source mechanics from machine-load or CUDA clock drift. This
runner records both controls, timings, source parity and optional GPU
telemetry. All four run reports remain in --output-root. No gain is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BEFORE_REV = "36b150002162696e7f47c9491fefe71b609390ad"
AFTER_REV = "2e28f983d01a5a1b69bd8ff7808f78af22d4d810"
ORDER = (("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A"))
REPORT_SCHEMA = "sts2-h8-h7-reference-ppo-benchmark-v1"
SUMMARY_SCHEMA = "sts2-h8-interleaved-abba-v1"
AI_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing numeric {label}") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid nonpositive {label}: {value!r}")
    return result


def _nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing report field {'.'.join(keys)}")
        value = value[key]
    return value


def validate_and_compare(
    reports: dict[str, dict[str, Any]],
    before_sha: str, after_sha: str,
    expected_count: int, replay_every: int,
) -> dict[str, Any]:
    """Fail closed on model/source drift, censored fights, and mismatched replay."""
    if set(reports) != {name for name, _ in ORDER}:
        raise ValueError("Required exactly A1, B1, B2, A2 reports")
    first = reports["A1"]
    equal_fields = ("schema", "model_sha256", "model_parameters",
                    "source_digest", "reference_law", "source_count",
                    "workers", "max_boundaries", "replay_every")
    if first.get("schema") != REPORT_SCHEMA:
        raise ValueError("Unknown H8 PPO report schema")
    if first.get("source_count") != expected_count or (
            first.get("replay_every") != replay_every):
        raise ValueError("Expected source count or audit schedule differs")
    fingerprints: dict[str, str] = {}
    checks: dict[str, Any] = {}
    expected_replays = len(range(0, expected_count, replay_every))
    decisions = first.get("native_h8_actor_metrics", {}).get("requests")
    if not isinstance(decisions, int) or decisions <= 0:
        raise ValueError("Missing frozen Transformer inference requests")

    for name, arm in ORDER:
        report = reports[name]
        for field in equal_fields:
            if report.get(field) != first.get(field):
                raise ValueError(f"{name}: changed {field}")
        revision = before_sha if arm == "A" else after_sha
        if report.get("emulator_revision") != revision:
            raise ValueError(f"{name}: emulator revision mismatch")
        dll_sha = report.get("emulator_dll_sha256")
        if not isinstance(dll_sha, str) or len(dll_sha) != 64:
            raise ValueError(f"{name}: missing emulator binary fingerprint")
        if arm in fingerprints and fingerprints[arm] != dll_sha:
            raise ValueError(f"{name}: rebuilt/changed {arm} DLL within A-B-B-A")
        fingerprints[arm] = dll_sha
        if report.get("terminal_counts") != {"terminal": expected_count}:
            raise ValueError(f"{name}: censored/incomplete combats")
        if report.get("verified_terminal_replays") != expected_replays:
            raise ValueError(f"{name}: sampled replay mismatch")
        if report.get("victories", -1) + report.get("defeats", -1) != expected_count:
            raise ValueError(f"{name}: missing victory or defeat outcomes")
        if report.get("victories") != first.get("victories") or (
                report.get("defeats") != first.get("defeats")):
            raise ValueError(f"{name}: different terminal outcomes")
        if abs(report.get("score_mean", float("nan")) -
               first.get("score_mean", float("nan"))) > 1e-10 or (
                not math.isfinite(report.get("score_mean", float("nan")))):
            raise ValueError(f"{name}: terminal reward drift")
        if _nested(report, "native_h8_actor_metrics", "requests") != decisions:
            raise ValueError(f"{name}: player decision count differs")
        if _nested(report, "native_h8_player_decisions_per_second") <= 0:
            raise ValueError(f"{name}: missing player throughput")
        if _nested(report, "ppo", "source_decisions") != decisions:
            raise ValueError(f"{name}: PPO source decision count differs")
        if report["ppo"]["updates"] != first["ppo"]["updates"]:
            raise ValueError(f"{name}: PPO optimizer update count differs")
        legacy = report.get("legacy_h7")
        if not isinstance(legacy, dict) or (
            legacy.get("matching_source_trajectories") != expected_count or
            legacy.get("player_decisions") != decisions):
            raise ValueError(f"{name}: missing exact legacy H7/H8 source parity")
        _positive(legacy.get("wall_seconds"), f"{name}.legacy_wall")
        _positive(report.get("native_h8_rollout_seconds"), f"{name}.rollout")
        _positive(report.get("ppo_seconds"), f"{name}.ppo")
        _positive(report.get("total_h8_iteration_seconds"), f"{name}.iteration")
        _positive(_nested(report, "native_h8_actor_metrics", "forward_seconds"),
                  f"{name}.actor_forward")
        _positive(_nested(report, "native_h8_actor_metrics", "collate_seconds"),
                  f"{name}.actor_collate")
        checks[name] = {
            "revision": revision, "dll_sha256": dll_sha,
            "rollout_seconds": report["native_h8_rollout_seconds"],
            "complete_combats_per_second": (
                expected_count / report["native_h8_rollout_seconds"]),
            "legacy_h7_seconds": legacy["wall_seconds"],
            "ppo_seconds": report["ppo_seconds"],
            "total_iteration_seconds": report["total_h8_iteration_seconds"],
            "actor_forward_seconds": report["native_h8_actor_metrics"]["forward_seconds"],
            "actor_collate_seconds": report["native_h8_actor_metrics"]["collate_seconds"],
            "actor_batches": report["native_h8_actor_metrics"]["batches"],
            "actor_requests": decisions,
        }

    if fingerprints["A"] == fingerprints["B"]:
        raise ValueError("Same emulator DLL passed as both baseline and candidate")
    # Mirrored adjacent comparisons lessen gradual CPU/GPU temperature drift.
    # Speedup >1 means B is faster. Two pairs are NOT a confidence interval.
    def paired_speedup(field: str) -> dict[str, Any]:
        values = {name: checks[name][field] for name, _ in ORDER}
        near_first = values["A1"] / values["B1"]
        near_second = values["A2"] / values["B2"]
        return {
            "adjacent_pair_speedups": [near_first, near_second],
            "geometric_mean_speedup": math.sqrt(near_first * near_second),
            "baseline_drift_A2_over_A1": values["A2"] / values["A1"],
            "candidate_drift_B2_over_B1": values["B2"] / values["B1"],
            "baseline_mean_seconds": (values["A1"] + values["A2"]) / 2,
            "candidate_mean_seconds": (values["B1"] + values["B2"]) / 2,
        }
    rollout = paired_speedup("rollout_seconds")
    return {
        "schema": SUMMARY_SCHEMA,
        "order": [name for name, _ in ORDER],
        "baseline_revision": before_sha,
        "candidate_revision": after_sha,
        "source_digest": first["source_digest"],
        "model_sha256": first["model_sha256"],
        "source_count": expected_count,
        "workers": first["workers"],
        "terminal_outcomes": {
            "victories": first["victories"], "defeats": first["defeats"],
            "score_mean": first["score_mean"]},
        "verified_terminal_replays_per_run": expected_replays,
        "legacy_h7_trajectory_matches_per_run": expected_count,
        "player_decisions_per_run": decisions,
        "runs": checks,
        "rollout_comparison": rollout,
        "legacy_h7_comparison": paired_speedup("legacy_h7_seconds"),
        "ppo_comparison": paired_speedup("ppo_seconds"),
        "total_iteration_comparison": paired_speedup("total_iteration_seconds"),
        "forward_comparison": paired_speedup("actor_forward_seconds"),
        "collation_comparison": paired_speedup("actor_collate_seconds"),
        "target_rollout_combats_per_second": 50,
        "target_reached_in_both_candidate_runs": all(
            checks[name]["complete_combats_per_second"] >= 50
            for name in ("B1", "B2")),
        "interpretation": (
            "Identical actor, source bank, full terminal outcomes, sampled replays "
            "and H7/H8 choices enforced for every run. This is an A-B-B-A "
            "end-to-end throughput comparison, not a source-mechanics-only "
            "profiling result. Two paired speedups cannot establish statistical "
            "significance; machine-load and GPU-clock variation still matter."),
    }


def _command_with_log(command: list[str], cwd: Path, log: Path) -> None:
    print("[h8-abba] " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError(f"Command failed; see {log}")


def _pinned_revision(repo: Path, revision: str) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify",
         revision + "^{commit}"], text=True).strip()
    if len(commit) != 40:
        raise ValueError("Malformed source revision " + revision)
    return commit


def _build_archive(repo: Path, sha: str, destination: Path, log: Path) -> Path:
    destination.mkdir(parents=True)
    archive = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", sha],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert archive.stdout is not None
    untar = subprocess.run(["tar", "-xf", "-", "-C", str(destination)],
                           stdin=archive.stdout, capture_output=True)
    archive.stdout.close()
    archive_error = archive.stderr.read().decode(errors="replace")
    archive_code = archive.wait()
    if untar.returncode or archive_code:
        raise RuntimeError(
            f"Cannot archive {sha}: {archive_error} "
            f"{untar.stderr.decode(errors='replace')}")
    _command_with_log(
        ["dotnet", "build", "Sts2Emulator.sln", "-c", "Release", "-m:1"],
        destination, log)
    dll = destination / "src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll"
    if not dll.is_file():
        raise FileNotFoundError(f"Archive build missing emulator CLI: {dll}")
    return dll


def _start_gpu_probe(output: Path) -> tuple[subprocess.Popen[bytes] | None, Any]:
    if not shutil.which("nvidia-smi"):
        return None, None
    probe = output.open("wb")
    try:
        process = subprocess.Popen([
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
            "clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits", "--loop-ms=1000"],
            stdout=probe, stderr=subprocess.STDOUT)
    except OSError:
        probe.close()
        return None, None
    return process, probe


def _stop_gpu_probe(process: subprocess.Popen[bytes] | None, handle: Any) -> None:
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if handle is not None:
        handle.close()


def _run_benchmark(
    name: str, revision: str, dll: Path, args: argparse.Namespace,
    output_root: Path, bindings: Path,
) -> None:
    output = output_root / f"{name}.json"
    if output.exists():
        raise FileExistsError(
            f"Refusing to reuse {output}; choose another output directory")
    cmd = [
        sys.executable, "-u", str(AI_ROOT / "tools/benchmark_h8_training_iteration.py"),
        "--emulator-dll", str(dll),
        "--emulator-bindings", str(bindings),
        "--emulator-revision", revision,
        "--output", str(output),
        "--workers", str(args.workers), "--episodes", str(args.episodes),
        "--max-boundaries", str(args.max_boundaries),
        "--replay-every", str(args.replay_every),
        "--study-seed", str(args.study_seed),
        "--actor-seed", str(args.actor_seed),
        "--reference-seed", str(args.reference_seed),
        "--actor-device", args.actor_device,
        "--learner-device", args.learner_device,
        "--actor-max-batch", str(args.actor_max_batch),
        "--actor-delay-ms", str(args.actor_delay_ms),
        "--compare-legacy",
    ]
    if args.player_checkpoint is not None:
        cmd += ["--player-checkpoint", str(args.player_checkpoint)]
    probe, handle = ((None, None) if args.no_gpu_telemetry or
                     not args.actor_device.startswith("cuda")
                     else _start_gpu_probe(output_root / f"{name}-gpu.csv"))
    start = time.perf_counter()
    try:
        _command_with_log(cmd, AI_ROOT, output_root / f"{name}.log")
    finally:
        _stop_gpu_probe(probe, handle)
    print(f"[h8-abba] completed {name} in {time.perf_counter()-start:.2f}s",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-repo", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--before-revision", default=BEFORE_REV)
    p.add_argument("--after-revision", default=AFTER_REV)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--replay-every", type=int, default=8)
    p.add_argument("--study-seed", type=int, default=73127)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--actor-device", default="cuda")
    p.add_argument("--learner-device", default="cuda")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--actor-delay-ms", type=float, default=3)
    p.add_argument("--player-checkpoint", type=Path)
    p.add_argument("--no-gpu-telemetry", action="store_true")
    p.add_argument("--report-only", action="store_true",
                   help="Verify/combine four existing A1,B1,B2,A2 JSON reports")
    args = p.parse_args()
    if (args.workers < 1 or args.episodes < 16 or args.episodes % 16 or
            args.max_boundaries < 1 or args.replay_every < 1 or
            args.actor_max_batch < 1 or args.actor_delay_ms < 0):
        p.error("Need positive worker/sample budgets and balanced 16-cell episodes")
    repo = args.emulator_repo.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not args.report_only and not (repo / ".git").exists():
        p.error("--emulator-repo must be the existing git checkout")
    before = (_pinned_revision(repo, args.before_revision)
              if not args.report_only else args.before_revision)
    after = (_pinned_revision(repo, args.after_revision)
             if not args.report_only else args.after_revision)
    if before == after:
        p.error("A and B must be distinct emulator revisions")
    bindings = (args.emulator_bindings or repo / "bindings/python").resolve()
    if not args.report_only and not (
            bindings / "sts2_emulator" / "__init__.py").is_file():
        p.error("Missing emulator Python bindings")
    if args.player_checkpoint:
        args.player_checkpoint = args.player_checkpoint.resolve()
        if not args.player_checkpoint.is_file():
            p.error("Missing player checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        if any((output / f"{name}.json").exists() for name, _ in ORDER):
            p.error("Output already contains an A-B-B-A run; select a fresh directory")
        print("[h8-abba] Extracting pinned revisions into temporary build "
              "directories; existing branches/worktrees remain untouched.",
              flush=True)
        with tempfile.TemporaryDirectory(prefix="h8-emulator-abba-") as td:
            root = Path(td)
            dlls = {
                "A": _build_archive(repo, before, root / "source-A",
                                    output / "build-A.log"),
                "B": _build_archive(repo, after, root / "source-B",
                                    output / "build-B.log"),
            }
            metadata = {
                "baseline_revision": before, "candidate_revision": after,
                "emulator_dll_sha256": {arm: sha256_file(dll) for arm, dll in dlls.items()},
                "platform": platform.platform(),
                "python_executable": sys.executable,
                "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (output / "environment.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            for name, arm in ORDER:
                print(f"[h8-abba] RUN {name} / {arm} ({before if arm=='A' else after})",
                      flush=True)
                _run_benchmark(name, before if arm == "A" else after,
                               dlls[arm], args, output, bindings)
    reports = {
        name: json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        for name, _ in ORDER
    }
    result = validate_and_compare(
        reports, before, after, args.episodes, args.replay_every)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rate = result["rollout_comparison"]
    print(
        "[h8-abba] all 4 complete and exact legacy parity; rollout speedup "
        f"{rate['geometric_mean_speedup']:.3f}x, adjacent pair speedups "
        f"{rate['adjacent_pair_speedups']}, "
        f"target=50/s reached={result['target_reached_in_both_candidate_runs']}; "
        f"summary={output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
