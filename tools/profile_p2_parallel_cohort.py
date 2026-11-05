#!/usr/bin/env python3
"""P2.0 pinned 1/N-worker cohort profiler; diagnostic only, no optimizer.

Each worker owns one persistent JSONL emulator and a public-only policy.
The optional shared actor is the production H16 service. Per-worker RPC
and inference times overlap across workers; cohort wall is the throughput
denominator. Exact replay comparison is always within the same worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from p2_replay_diagnostics import classify_trace_difference
from profile_p2_joint_throughput import (
    ROOT,
    critical_path_shares,
    dll_identity,
    git_head,
    run_episode,
    sha256,
)

from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.models.h16_batch_actor import RemoteH16Actor, SharedH16Service
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet, PilotPolicy
from sts2_ai.training.h16_public_codec import H16_CODEC
from sts2_ai.training.h21_macro_offer_codec import H21_CODEC

SCHEMA = "sts2-p2-parallel-cohort-profile-v1"
_WORKER: dict[str, Any] = {}


def _worker_init(config: dict[str, Any], transport: Any) -> None:
    torch.set_num_threads(1)
    remote = RemoteH16Actor(transport) if transport is not None else None
    model = (
        None if remote is not None
        else JointActPilotNet.load(Path(config["checkpoint"]), device="cpu")
    )
    macro = (
        PhaseSplitNeuralModel.load(Path(config["macro_model"]))
        if config["macro_model"] is not None and config["macro_prior_weight"] > 0
        else None
    )
    policy = PilotPolicy(
        model, temperature=0.85,
        horizon_lambda=config["horizon_lambda"],
        macro_teacher=macro,
        macro_prior_weight=config["macro_prior_weight"],
        remote_actor=remote,
        representation=config["representation"],
    )
    backend = JsonlEmulatorBackend(
        repo_root=Path(config["ai_root"]),
        emulator_root=Path(config["emulator_root"]),
        expected_emulator_revision=config["emulator_revision"],
        build=False, profile_wire=False,
        expected_public_schema_id=config["public_schema_id"],
    )
    global _WORKER
    _WORKER = {"config": config, "policy": policy, "backend": backend}


def _worker_seed(seed: str) -> dict[str, Any]:
    cfg = _WORKER["config"]
    backend = _WORKER["backend"]
    policy = _WORKER["policy"]
    opts = {
        "seed": seed,
        "ascension": cfg["ascension"],
        "fixed_route": cfg["map_profile"] == "fixed-route",
        "max_decisions": cfg["max_decisions"],
        "max_searches": cfg["max_searches"],
        "simulations": cfg["simulations"],
        "search_depth": None if cfg["depth"] == 0 else cfg["depth"],
        "max_simulation_steps": cfg["max_simulation_steps"],
        "rollout_transport": cfg["rollout_transport"],
        "compact_card_selection": cfg["representation"] == "h21",
        "search_enabled": not cfg["no_search"],
        "p2_prefix_cache_handles": cfg["p2_prefix_cache_handles"],
    }
    if cfg["profile_overhead"]:
        arms = ("control_a", "profile_b1", "profile_b2", "control_a2")
    elif cfg["episode_profile"]:
        arms = ("profile_b1", "profile_b2") if cfg["paired"] else ("profile_b1",)
    else:
        # Two *unprofiled* repetitions still certify exact action/teacher
        # replay; they must be counted in full cohort wall/throughput.
        arms = ("control_a", "control_a2") if cfg["paired"] else ("control_a",)
    signature0: dict[str, Any] | None = None
    runs = []
    for arm in arms:
        instrumented = arm.startswith("profile_")
        try:
            report, signature = run_episode(
                backend, policy, **opts,
                profile_search=instrumented,
                profile_wire=instrumented or cfg["wire_profile_only"],
                native_profile=cfg["native_profile"] and instrumented,
                native_fused_profile=cfg["native_fused_profile_only"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"P2 cohort episode failed: seed={seed}, arm={arm}, "
                f"simulations={cfg['simulations']}, depth={cfg['depth']}"
            ) from exc
        report["arm"] = arm
        if signature0 is None:
            signature0 = signature
        elif signature != signature0:
            # Fail closed, but preserve the location/type of the first drift.
            # Mixed GPU batch shapes may change numerical logits enough to
            # alter PUCT teachers or sampled actions. Do not silently round
            # teacher distributions or accept a changed action trajectory.
            classification = classify_trace_difference(signature0, signature)
            difference = classification["first_difference"]
            assert difference is not None
            diagnostic = {
                "schema": "sts2-p2-public-replay-mismatch-v1",
                "seed": seed, "arm": arm,
                "actor_max_batch": cfg["actor_max_batch"],
                "actor_device": cfg["actor_device"],
                "first_difference": difference,
                "classification": classification,
                "expected_trace_sha256": hashlib.sha256(
                    json.dumps(signature0, sort_keys=True).encode()
                ).hexdigest(),
                "actual_trace_sha256": hashlib.sha256(
                    json.dumps(signature, sort_keys=True).encode()
                ).hexdigest(),
            }
            path = Path(cfg["replay_diagnostics_dir"]) / (
                hashlib.sha256(seed.encode()).hexdigest()[:12]
                + "-" + arm + ".json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"P2 exact replay mismatch ({classification['kind']}): "
                f"seed={seed}, arm={arm}, first={difference}, "
                f"semantic_parity={classification['semantic_parity']}, "
                f"search_work_parity={classification['search_work_parity']}, "
                f"diagnostics={path}"
            )
        runs.append(report)
    assert signature0 is not None
    return {
        "seed": seed,
        "runs": runs,
        "public_teacher_parity": True if len(arms) > 1 else None,
        "trace_sha256": hashlib.sha256(json.dumps(
            signature0, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true",
                        help="Explicitly allow replacing a finished profile.json "
                             "(default: preserve previous evidence)")
    parser.add_argument("--ai-root", type=Path, default=ROOT)
    parser.add_argument("--emulator-root", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--macro-model", type=Path)
    parser.add_argument("--macro-prior-weight", type=float, default=0.0)
    parser.add_argument("--horizon-lambda", type=float, default=0.25)
    parser.add_argument("--seeds", default=",".join(
        f"p2-a0-diagnostic-{i}" for i in range(8)
    ))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--actor-inference", choices=["shared", "local"], default="shared")
    parser.add_argument("--actor-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--actor-max-batch", type=int, default=16)
    parser.add_argument("--actor-max-delay-ms", type=float, default=2.0)
    parser.add_argument("--actor-deterministic-cuda", action="store_true",
                        help="Opt-in deterministic CUDA scatter/BLAS kernels in H22 "
                             "shared actor; diagnostic until benchmarked")
    parser.add_argument("--actor-profile", action="store_true",
                        help="Collect bounded queue/batch/forward distributions")
    parser.add_argument("--actor-batch-parity-probe", action="store_true",
                        help="Diagnostic only: re-evaluate singleton inside mixed batch; "
                             "extra inference invalidates throughput timing")
    parser.add_argument("--actor-fixed-shape-batch-size", type=int, default=0,
                        help="Opt-in fixed-row, shape-bucketed actor microbatches "
                             "(0 = existing dynamic batching)")
    parser.add_argument("--representation", choices=("h16", "h21"), default="h16")
    parser.add_argument("--actor-h21-collator",
                        choices=("reference", "packed"), default="reference")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--map-profile", choices=["generated", "fixed-route"],
                        default="generated")
    parser.add_argument("--simulations", type=int, choices=[8, 16, 64], default=16)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--max-simulation-steps", type=int, default=512)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--rollout-transport", choices=["fused", "legacy"], default="fused")
    parser.add_argument("--native-profile", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--profile-overhead", action="store_true")
    parser.add_argument("--p2-prefix-cache-handles", type=int, default=0,
                        help="P2.1 opt-in Native root-conditioned prefix cache; "
                             "0 = no cache, positive = max retained handles/root")
    parser.add_argument("--native-fused-profile-only", action="store_true",
                        help="Opt-in H16/v0 C# fused mechanics/serialization "
                             "timings, without PUCT/actor/JSONL-wire profiling")
    parser.add_argument("--wire-profile-only", action="store_true",
                        help="Wire JSONL byte/encode/pipe/decode timings only; "
                             "requires --no-episode-profile, no PUCT profiler")
    parser.add_argument("--no-episode-profile", action="store_true",
                        help="Disable per-root PUCT and JSONL-wire profiling "
                             "while retaining exact B/B public+teacher replay")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    a = parse_args()
    if not a.execute:
        raise SystemExit("Pass --execute for actual emulator profiling")
    if a.output.exists() and not a.overwrite_output:
        raise SystemExit(
            f"Refusing to overwrite existing P2 profile: {a.output}. "
            "Choose a new --output or pass --overwrite-output explicitly."
        )
    seeds = [x.strip() for x in a.seeds.split(",") if x.strip()]
    if (
        not seeds or len(set(seeds)) != len(seeds)
        or a.workers < 1 or a.depth < 0
        or a.max_decisions < 1 or a.max_searches < 0
        or a.max_simulation_steps < 1
        or a.p2_prefix_cache_handles < 0
        or a.p2_prefix_cache_handles > 128
        or not 0 <= a.ascension <= 10
        or not 0 <= a.horizon_lambda <= 1
        or a.macro_prior_weight < 0
        or (a.macro_prior_weight > 0 and a.macro_model is None)
        or a.actor_max_batch < 1 or a.actor_max_delay_ms < 0
        or (a.representation == "h21" and a.actor_fixed_shape_batch_size != 0)
        or (a.actor_h21_collator == "packed" and
            (a.representation != "h21" or a.actor_inference != "shared"))
        or (a.actor_deterministic_cuda and (
            a.representation != "h21" or a.actor_inference != "shared"
            or not a.actor_device.startswith("cuda")))
        or (a.no_episode_profile and (a.profile_overhead or a.native_profile))
        or (a.wire_profile_only and (
            not a.no_episode_profile or a.profile_overhead
            or a.native_profile or a.actor_profile
            or a.native_fused_profile_only
        ))
        or (a.native_fused_profile_only and (
            not a.no_episode_profile or a.profile_overhead
            or a.native_profile or a.actor_profile
        ))
        or (a.actor_batch_parity_probe and (
            not a.actor_profile or a.actor_inference != "shared"
        ))
        or (a.actor_fixed_shape_batch_size < 0
            or a.actor_fixed_shape_batch_size > a.actor_max_batch
            or (a.actor_fixed_shape_batch_size > 0
                and a.actor_inference != "shared")
            or (a.actor_fixed_shape_batch_size > 0
                and a.actor_batch_parity_probe))
    ):
        raise SystemExit("Invalid P2 parallel profiler configuration")
    if a.actor_inference == "local" and a.actor_device != "cpu":
        raise SystemExit("Local-worker inference is CPU-only; use shared CUDA actor")
    if not a.checkpoint.is_file():
        raise SystemExit(
            f"Missing frozen actor checkpoint: {a.checkpoint}. "
            "Create or select an H22 checkpoint before running the cohort."
        )
    if a.representation == "h21":
        try:
            checkpoint_model = JointActPilotNet.load(a.checkpoint, device="cpu")
        except (RuntimeError, ValueError, KeyError) as exc:
            raise SystemExit(f"Invalid H22 checkpoint: {a.checkpoint}: {exc}") from exc
        if checkpoint_model.representation != "h21":
            raise SystemExit(
                f"Expected H22/H21 v4 checkpoint, got "
                f"{checkpoint_model.representation}: {a.checkpoint}"
            )
        del checkpoint_model
    if a.actor_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA actor unavailable")
    if git_head(a.emulator_root) != a.emulator_revision:
        raise SystemExit("Emulator SHA mismatch")
    dll = dll_identity(a.emulator_root)
    if dll is None:
        raise SystemExit("Exactly one pinned Release emulator CLI DLL required")
    config = {
        "ai_root": str(a.ai_root.resolve()),
        "emulator_root": str(a.emulator_root.resolve()),
        "emulator_revision": a.emulator_revision,
        "checkpoint": str(a.checkpoint.resolve()),
        "macro_model": str(a.macro_model.resolve()) if a.macro_model else None,
        "macro_prior_weight": a.macro_prior_weight,
        "horizon_lambda": a.horizon_lambda,
        "ascension": a.ascension,
        "map_profile": a.map_profile,
        "simulations": a.simulations,
        "no_search": a.no_search,
        "depth": a.depth,
        "max_simulation_steps": a.max_simulation_steps,
        "p2_prefix_cache_handles": a.p2_prefix_cache_handles,
        "max_searches": a.max_searches,
        "max_decisions": a.max_decisions,
        "rollout_transport": a.rollout_transport,
        "representation": a.representation,
        "public_schema_id": (
            "prototype-ai-v1" if a.representation == "h21" else "prototype-ai-v0"
        ),
        "native_profile": a.native_profile,
        "paired": a.paired,
        "profile_overhead": a.profile_overhead,
        "episode_profile": not a.no_episode_profile,
        "wire_profile_only": a.wire_profile_only,
        "native_fused_profile_only": a.native_fused_profile_only,
        "actor_max_batch": a.actor_max_batch,
        "actor_device": a.actor_device,
        "actor_batch_parity_probe": a.actor_batch_parity_probe,
        "actor_fixed_shape_batch_size": a.actor_fixed_shape_batch_size,
        "actor_h21_collator": a.actor_h21_collator,
        "actor_deterministic_cuda": a.actor_deterministic_cuda,
        "replay_diagnostics_dir": str(
            a.output.resolve().parent / "replay-mismatches"
        ),
    }
    # Measure process startup, actor checkpoint load, all episodes, and teardown
    # in the cohort denominator. Worker sums below are intentionally overlapping.
    service = (
        SharedH16Service(
            checkpoint=a.checkpoint.resolve(), digest=sha256(a.checkpoint),
            workers=a.workers, device=a.actor_device,
            horizon_lambda=a.horizon_lambda,
            max_batch=a.actor_max_batch, max_delay_ms=a.actor_max_delay_ms,
            profile_actor=a.actor_profile,
            verify_batch_parity=a.actor_batch_parity_probe,
            fixed_shape_batch_size=a.actor_fixed_shape_batch_size,
            h21_collator=a.actor_h21_collator,
            deterministic_cuda=a.actor_deterministic_cuda,
        )
        if a.actor_inference == "shared" else nullcontext(None)
    )
    started = time.perf_counter()
    actor = None
    try:
        with service as actor:
            transport = actor.transport if actor is not None else None
            with ProcessPoolExecutor(
                max_workers=a.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init, initargs=(config, transport),
            ) as pool:
                seed_reports = list(pool.map(_worker_seed, seeds, chunksize=1))
    except Exception:
        # An exact replay mismatch aborts before normal report generation.
        # Keep standalone-vs-batched model statistics to diagnose it.
        if (a.actor_batch_parity_probe and actor is not None
                and actor.metrics.get("p2_batch_numerical_parity") is not None):
            probe_path = (
                a.output.resolve().parent / "actor-batch-numerical-parity.json"
            )
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path.write_text(json.dumps(
                actor.metrics["p2_batch_numerical_parity"],
                indent=2, sort_keys=True,
            ) + "\n", encoding="utf-8")
        raise
    cohort_wall = time.perf_counter() - started
    runs = [row for seed in seed_reports for row in seed["runs"]]
    profile_runs = [r for r in runs if r["arm"].startswith("profile_")]
    # Report sampled Python worker sums as overlapping resource demand; only
    # the cohort elapsed time is valid for measured parallel throughput.
    actor_sum = sum(float(r["actor"]["inference_seconds"]) for r in profile_runs)
    bridge_sum = sum(float(op["total_seconds"]) for r in profile_runs
                     for op in r["bridge"].values())
    worker_elapsed_sum = sum(float(r["episode_wall_seconds"]) for r in profile_runs)
    profile_diagnostics = (
        critical_path_shares(worker_elapsed_sum, bridge_sum, actor_sum)
        if a.workers == 1 and worker_elapsed_sum > 0 else None
    )
    result = {
        "schema": SCHEMA,
        "evidence": "P2 diagnostic; no training, no throughput optimization",
        "identity": {
            "ai_sha": git_head(a.ai_root),
            "emulator_sha": a.emulator_revision,
            "emulator_cli": dll,
            "checkpoint_sha256": sha256(a.checkpoint),
            "macro_sha256": sha256(a.macro_model) if a.macro_model else None,
            "worker_count": a.workers,
            "actor_inference": a.actor_inference,
            "actor_device": a.actor_device,
            "actor_max_batch": a.actor_max_batch,
            "actor_max_delay_ms": a.actor_max_delay_ms,
            "actor_profile": a.actor_profile,
            "actor_batch_parity_probe": a.actor_batch_parity_probe,
            "actor_fixed_shape_batch_size": a.actor_fixed_shape_batch_size,
            "actor_h21_collator": a.actor_h21_collator,
            "actor_deterministic_cuda": a.actor_deterministic_cuda,
            "representation": a.representation,
            "public_schema_id": config["public_schema_id"],
            "codec": H21_CODEC if a.representation == "h21" else H16_CODEC,
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "seeds": seeds,
            **config,
        },
        "seed_reports": seed_reports,
        "cohort_wall_seconds": cohort_wall,
        # Cohort throughput includes every actually executed A/B/B/A arm.
        # Profile-only worker resource sums are reported separately below.
        "completed_episodes": sum(int(r["complete"]) for r in runs),
        "censored_episodes": sum(int(not r["complete"]) for r in runs),
        "profiled_completed_episodes": sum(int(r["complete"]) for r in profile_runs),
        "search_roots": sum(int(r["search_roots"]) for r in runs),
        "search_transitions": sum(int(r["search_transitions"]) for r in runs),
        "real_transitions": sum(int(r["real_transitions"]) for r in runs),
        "p2_prefix_cache_diagnostics": {
            metric: sum(
                (float if metric == "receipt_seconds" else int)(
                    run.get("p2_prefix_cache", {}).get(metric, 0)
                ) for run in runs
            )
            for metric in (
                "roots", "hits", "misses", "emulator_step_calls_avoided",
                "receipt_calls", "receipt_seconds",
                "skipped_capacity", "skipped_provider",
            )
        },
        "worker_elapsed_sum_seconds": worker_elapsed_sum,
        "worker_bridge_sum_seconds": bridge_sum,
        "worker_actor_inference_sum_seconds": actor_sum,
        "sequential_worker_wall_attribution": profile_diagnostics,
        "measured_full_cohort_completed_episodes_per_second": (
            sum(int(r["complete"]) for r in runs) / cohort_wall
        ),
        "measured_full_cohort_search_transitions_per_second": (
            sum(int(r["search_transitions"]) for r in runs) / cohort_wall
        ),
        "shared_actor_metrics": (actor.metrics if a.actor_inference == "shared" else None),
        "warnings": [
            "Same seed cohort must be run separately at 1 and 15 workers",
            "Worker-summed times overlap; never add them or use as cohort wall",
            "Generated-map native-structure RNG is not installed-game calibrated",
            ("Pinned H21 public-v1 grounded model and relation graph"
             if a.representation == "h21" else
             "Actor model forward/codec remain H16; H21 joint forward not measured"),
            "Opt-in actor queue samples are capped at the first 8192 values",
            "CUDA transfer synchronization is enabled only with --actor-profile",
            *(
                ["PUCT/JSONL instrumentation disabled; public/teacher replay retained"]
                if a.no_episode_profile else []
            ),
            "This diagnostic is not a 50+ completed-combats-per-second claim",
            *(
                ["Batch parity probe adds GPU forwards: timing is NOT a throughput benchmark"]
                if a.actor_batch_parity_probe else []
            ),
            *(
                ["P2.1 retained-state cache is opt-in, root-conditioned only; "
                 "verify exact teacher replay against zero-cache control"]
                if a.p2_prefix_cache_handles else []
            ),
            *(
                ["Experimental fixed-row batching pads each shape bucket; "
                 "compare exact replay and its physical-row overhead before acceptance"]
                if a.actor_fixed_shape_batch_size else []
            ),
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "profile": str(a.output),
        "workers": a.workers,
        "seed_count": len(seeds),
        "completed": result["completed_episodes"],
        "censored": result["censored_episodes"],
        "cohort_wall_seconds": cohort_wall,
    }, indent=2))


if __name__ == "__main__":
    main()
