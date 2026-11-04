#!/usr/bin/env python3
"""P2.0 single-worker H16/H18 measured search/bridge probe.

Runs ordinary frozen public-only H16 inference and exact-root-conditioned
Native RNG PUCT with the existing emulator API. Does not train or change
the PUCT budget. The emitted report is diagnostic, not an A/B acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotEpisode,
    PilotPolicy,
    collect_act_episode,
)
from sts2_ai.training.h16_public_codec import H16_CODEC, set_h16_rollout_seed

SCHEMA = "sts2-p2-joint-throughput-profile-v1"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()


def dll_identity(root: Path) -> dict[str, str] | None:
    binaries = sorted(
        (root / "src/Sts2Emulator.Cli/bin/Release").glob(
            "*/Sts2Emulator.Cli.dll"
        )
    )
    if len(binaries) != 1:
        return None
    return {"path": str(binaries[0]), "sha256": sha256(binaries[0])}


def extract_signature(episode: PilotEpisode) -> dict[str, Any]:
    """The public decision and teacher trace used for paired replay checks."""
    return {
        "outcome": episode.outcome,
        "searches": episode.searches,
        "transitions": episode.search_transitions,
        "actions": [
            {
                "observation_hash": d.observation.observation_hash,
                "action_id": d.legal_actions[d.chosen_index].action_id,
                "legal_action_ids": [a.action_id for a in d.legal_actions],
                "teacher_policy": list(d.search_policy) if d.search_policy is not None else None,
            }
            for d in episode.decisions
        ],
        "final_public_frame": episode.frames[-1],
    }


def critical_path_shares(
    elapsed: float, bridge: float, inference: float,
) -> list[dict[str, float | str]]:
    """Nonoverlapping single-worker attribution, *not* parallel wall sums.

    Sampler/PUCT timings are nested within these intervals and must not
    be added as an independent contribution to critical-path time.
    """
    if elapsed <= 0:
        raise ValueError("Positive episode wall time required")
    other = max(0.0, elapsed - bridge - inference)
    entries = [
        ("jsonl_rpc_wall", bridge),
        ("public_actor_inference", inference),
        ("python_orchestration_and_unattributed", other),
    ]
    return [
        {
            "name": name,
            "seconds": seconds,
            "fraction_of_episode_wall": seconds / elapsed,
            "ideal_remove_all_speedup_bound": (
                1 / (1 - min(0.999999, seconds / elapsed))
            ),
        }
        for name, seconds in sorted(entries, key=lambda entry: -entry[1])
    ]


def run_episode(
    backend: JsonlEmulatorBackend, policy: PilotPolicy, *,
    seed: str, ascension: int, fixed_route: bool,
    max_decisions: int, max_searches: int, simulations: int,
    search_depth: int | None, max_simulation_steps: int,
    rollout_transport: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The H16 policy and emulator streams have stable per-episode seed owners.
    set_h16_rollout_seed(seed)
    rng_seed = int(hashlib.sha256(
        (seed + "/p2-profile-player-v1").encode()
    ).hexdigest()[:16], 16)
    backend.reset_operation_profile()
    policy.inference_calls = 0
    policy.inference_seconds = 0.0
    roots: list[dict[str, object]] = []
    from sts2_ai.emulator.run_environment import EXPERIMENTAL_FIXED_ROUTE_PROFILE
    start = time.perf_counter()
    episode = collect_act_episode(
        backend, policy,
        seed=seed,
        ascension=ascension,
        map_profile_id=(EXPERIMENTAL_FIXED_ROUTE_PROFILE if fixed_route else None),
        compact_card_selection=not fixed_route,
        max_decisions=max_decisions,
        search_probability=1.0,
        max_searches=max_searches,
        simulations=simulations,
        search_depth=search_depth,
        max_simulation_steps=max_simulation_steps,
        rng=random.Random(rng_seed),
        rollout_transport=rollout_transport,
        search_profiles=roots,
    )
    elapsed = time.perf_counter() - start
    ops = backend.operation_profile()
    wire = backend.wire_operation_profile()
    bridge_wall = sum(op.total_seconds for op in ops.values())
    actor_wall = policy.inference_seconds
    report = {
        "seed": seed,
        "outcome": episode.outcome,
        "complete": episode.complete,
        "decisions": len(episode.decisions),
        "max_floor": max(
            (f.get("floor", 0) for f in episode.frames
             if type(f.get("floor")) is int),
            default=0,
        ),
        "search_roots": len(roots),
        "search_simulations": sum(int(r["simulations"]) for r in roots),
        "search_transitions": episode.search_transitions,
        "real_transitions": len(episode.decisions),
        "real_transitions_per_second": len(episode.decisions) / elapsed,
        "all_transitions_per_second": (
            len(episode.decisions) + episode.search_transitions
        ) / elapsed,
        "episode_wall_seconds": elapsed,
        "completed_runs_per_second": (1 / elapsed if episode.complete else 0.0),
        "search_transitions_per_second": episode.search_transitions / elapsed,
        "bridge": {
            op: asdict(stats) for op, stats in ops.items()
        },
        "wire": {
            op: asdict(stats) for op, stats in wire.items()
        },
        "actor": {
            "inference_calls": policy.inference_calls,
            "inference_seconds": actor_wall,
            "note": "CPU/public codec and forward are combined in PilotPolicy.infer",
        },
        "search": roots,
        "critical_path_shares": critical_path_shares(
            elapsed, bridge_wall, actor_wall,
        ),
        "nested_search_wall_seconds": sum(float(r["wall_seconds"]) for r in roots),
        "diagnostic_missing": [
            "C# self-time and allocations", "retained-handle bytes",
            "worker contention and queue histograms",
            "GPU synchronization-specific timings (when using CUDA)",
        ],
    }
    return report, extract_signature(episode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="Actually run the emulator probe; otherwise only print help")
    p.add_argument("--ai-root", type=Path, default=ROOT)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True,
                   help="Required exact HEAD of the pinned emulator checkout")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Frozen H16 JointActPilotNet checkpoint")
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--macro-prior-weight", type=float, default=0.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seeds", default="p2-diagnostic-0,p2-diagnostic-1")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--map-profile", choices=["generated", "fixed-route"],
                   default="generated")
    p.add_argument("--simulations", type=int, choices=[8, 16, 64], default=16)
    p.add_argument("--depth", type=int, default=8,
                   help="0 = full-combat PUCT; otherwise measured action depth")
    p.add_argument("--max-simulation-steps", type=int, default=512)
    p.add_argument("--max-searches", type=int, default=2)
    p.add_argument("--max-decisions", type=int, default=80)
    p.add_argument("--horizon-lambda", type=float, default=0.25)
    p.add_argument("--rollout-transport", choices=["fused", "legacy"], default="fused")
    p.add_argument("--paired", action="store_true",
                   help="Repeat each episode and demand identical public/teacher traces")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not a.execute:
        raise SystemExit("Pass --execute to run the diagnostic probe")
    if (
        not 0 <= a.ascension <= 10 or a.depth < 0
        or a.max_searches < 0 or a.max_decisions <= 0
        or a.max_simulation_steps <= 0
        or not 0 <= a.horizon_lambda <= 1
        or a.macro_prior_weight < 0
    ):
        raise SystemExit("Invalid P2 probe configuration")
    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("Expected nonempty unique diagnostic seed list")
    if a.macro_prior_weight and a.macro_model is None:
        raise SystemExit("Macro prior weight requires --macro-model")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    ai_revision = git_head(a.ai_root)
    actual_emulator_revision = git_head(a.emulator_root)
    if actual_emulator_revision != a.emulator_revision:
        raise SystemExit("Emulator revision mismatch; refuse unpinned profiling")
    dll = dll_identity(a.emulator_root)
    if dll is None:
        raise SystemExit("Expected exactly one Release CLI DLL: build pinned emulator first")

    model = JointActPilotNet.load(a.checkpoint, device=a.device)
    macro = (PhaseSplitNeuralModel.load(a.macro_model)
             if a.macro_model is not None and a.macro_prior_weight > 0 else None)
    policy = PilotPolicy(
        model, temperature=0.85,
        horizon_lambda=a.horizon_lambda,
        macro_teacher=macro,
        macro_prior_weight=a.macro_prior_weight,
    )
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    reports = []
    paired_checks = []
    with JsonlEmulatorBackend(
        repo_root=a.ai_root, emulator_root=a.emulator_root,
        expected_emulator_revision=a.emulator_revision,
        build=False, profile_wire=True,
    ) as backend:
        for seed in seeds:
            options = {
                "seed": seed,
                "ascension": a.ascension,
                "fixed_route": a.map_profile == "fixed-route",
                "max_decisions": a.max_decisions,
                "max_searches": a.max_searches,
                "simulations": a.simulations,
                "search_depth": None if a.depth == 0 else a.depth,
                "max_simulation_steps": a.max_simulation_steps,
                "rollout_transport": a.rollout_transport,
            }
            first, signature = run_episode(backend, policy, **options)
            reports.append(first)
            if a.paired:
                second, second_signature = run_episode(backend, policy, **options)
                if signature != second_signature:
                    raise RuntimeError(f"Public replay/teacher mismatch: seed={seed}")
                # Timing varies; compare deterministic root visits/teacher only.
                for left, right in zip(first["search"], second["search"], strict=True):
                    for field in ("root_visits", "teacher_policy", "transitions"):
                        if left[field] != right[field]:
                            raise RuntimeError(
                                f"PUCT mismatch: {field} seed={seed}"
                            )
                paired_checks.append({"seed": seed, "public_teacher_parity": True})

    result = {
        "schema": SCHEMA,
        "evidence": "diagnostic single-worker measured probe; no training/optimization",
        "identity": {
            "ai_sha": ai_revision,
            "emulator_sha": actual_emulator_revision,
            "emulator_cli": dll,
            "checkpoint_sha256": sha256(a.checkpoint),
            "macro_sha256": sha256(a.macro_model) if a.macro_model else None,
            "codec": H16_CODEC,
            "observation_policy": "prototype-fair-v0",
            "stochastic_continuation": "h16-native-act1-root-conditioned-privileged-v1",
            "map_profile": a.map_profile,
            "ascension": a.ascension,
            "seeds": seeds,
            "worker_count": 1,
            "search_simulations": a.simulations,
            "search_depth": None if a.depth == 0 else a.depth,
            "max_searches": a.max_searches,
            "rollout_transport": a.rollout_transport,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "device": a.device,
        },
        "runs": reports,
        "paired_checks": paired_checks,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if a.device == "cuda" else None
        ),
        "warnings": [
            "Source-generated map RNG is experimental and uncalibrated",
            "Profile-off paired A/B/A and 15-worker contention still pending",
            "Search timing is nested within bridge/actor workload; never sum overlapping times",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "profile": str(a.output), "seeds": seeds, "runs": len(reports),
        "search_roots": sum(int(x["search_roots"]) for x in reports),
        "completed_runs": sum(bool(x["complete"]) for x in reports),
    }, indent=2))


if __name__ == "__main__":
    main()
