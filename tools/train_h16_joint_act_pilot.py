#!/usr/bin/env python3
"""First H16 joint Act-1 self-play, delayed-HP and local-PUCT experiment.

Full Act trajectories supply *both* macro and combat updates. The optional
fixed H6 macro reference enters only as an action prior; its behavior is part
of the logged on-policy distribution. The H9/H7 shared Transformer is the
trainable actor backbone. The original HP combat objective is preserved as
a research baseline, never relabelled as binary combat victory.

PUCT runs only within combat, on short exact-state forks. These are
root-conditioned/possibly privileged approximation targets, and must not be
called certified public-history-fair search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import torch

from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import EXPERIMENTAL_FIXED_ROUTE_PROFILE
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.shared_transformer import SharedObjectTransformer
from sts2_ai.training.h16_joint_act_pilot import (
    SCHEMA,
    JointActPilotNet,
    PilotEpisode,
    PilotPolicy,
    collect_act_episode,
    train_joint_cohort,
)

_WORKER: dict[str, Any] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False,
        prefix="." + path.name + "-",
    ) as stream:
        name = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(name, path)


def _worker_init(config: dict[str, Any]) -> None:
    torch.set_num_threads(1)
    model = JointActPilotNet.load(Path(config["checkpoint"]), device="cpu")
    macro = (PhaseSplitNeuralModel.load(Path(config["macro_model"]))
             if config["macro_model"] is not None else None)
    policy = PilotPolicy(
        model, temperature=config["temperature"],
        horizon_lambda=config["horizon_lambda"], macro_teacher=macro,
        macro_prior_weight=config["macro_prior_weight"],
    )
    emulator = JsonlEmulatorBackend(
        repo_root=Path(config["ai_root"]),
        emulator_root=Path(config["emulator_root"]),
        expected_emulator_revision=config["emulator_revision"],
        build=False,
    )
    global _WORKER
    _WORKER = {"backend": emulator, "policy": policy, "config": config}


def _worker_run(seed: str) -> PilotEpisode:
    cfg = _WORKER["config"]
    rng = random.Random(int(digest(
        [seed, cfg["study_seed"], cfg["checkpoint_sha256"], "h16-player"],
    )[:16], 16))
    backend = _WORKER["backend"]
    policy = _WORKER["policy"]
    backend.reset_operation_profile()
    policy.inference_calls = 0
    policy.inference_seconds = 0.0
    episode = collect_act_episode(
        backend, policy,
        seed=seed, ascension=10,
        max_decisions=cfg["max_decisions"],
        search_probability=cfg["search_probability"],
        max_searches=cfg["max_searches"],
        simulations=cfg["simulations"],
        search_depth=cfg["search_depth"],
        rng=rng,
        rollout_transport=cfg["rollout_transport"],
    )
    return dataclass_replace(
        episode,
        bridge_profile={
            op: (v.calls, v.total_seconds)
            for op, v in backend.operation_profile().items()
        },
        actor_inference_calls=policy.inference_calls,
        actor_inference_seconds=policy.inference_seconds,
    )


def collect(
    args: argparse.Namespace, checkpoint: Path, *,
    seeds: list[str], horizon_lambda: float, search: bool,
) -> list[PilotEpisode]:
    cfg = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "macro_model": str(args.macro_model.resolve()) if args.macro_model else None,
        "ai_root": str(args.ai_root.resolve()),
        "emulator_root": str(args.emulator_root.resolve()),
        "emulator_revision": args.emulator_revision,
        "temperature": args.temperature,
        "macro_prior_weight": args.macro_prior_weight,
        "horizon_lambda": horizon_lambda,
        "max_decisions": args.max_decisions,
        "search_probability": args.search_probability if search else 0.0,
        "max_searches": args.max_searches if search else 0,
        "simulations": args.simulations,
        "search_depth": args.search_depth,
        "study_seed": args.study_seed,
        "rollout_transport": args.rollout_transport,
    }
    # Spawn avoids inheriting the learner's CUDA context into CPU workers.
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init, initargs=(cfg,),
    ) as pool:
        return list(pool.map(_worker_run, seeds, chunksize=1))


def summarize(episodes: list[PilotEpisode]) -> dict[str, Any]:
    completed = [e for e in episodes if e.complete]
    cleared = sum(e.outcome == "act_clear" for e in completed)
    resource_signatures = set()
    potion_counts: list[int] = []
    potion_actions = 0
    operations: dict[str, dict[str, float | int]] = {}
    for episode in episodes:
        for op, (calls, seconds) in (episode.bridge_profile or {}).items():
            record = operations.setdefault(op, {"calls": 0, "seconds": 0.0})
            record["calls"] += calls
            record["seconds"] += seconds
        potion_actions += sum(
            decision.legal_actions[decision.chosen_index].kind == "use_potion"
            for decision in episode.decisions
        )
        last_completed = 0
        for state in episode.frames:
            visits = state.get("completed_rooms")
            n = len(visits) if isinstance(visits, list) else 0
            if n <= last_completed:
                continue
            last_completed = n
            potions = state.get("potions")
            relics = state.get("relics")
            deck = state.get("deck")
            if not all(isinstance(x, list) for x in (potions, relics, deck)):
                continue
            present = [p for p in potions if p is not None]
            potion_counts.append(len(present))
            resource_signatures.add(digest([
                potions, relics, deck,
            ]))
    return {
        "played": len(episodes),
        "completed": len(completed),
        "censored": len(episodes) - len(completed),
        "cleared": cleared,
        "act_clear_rate_among_completed": (
            cleared / len(completed) if completed else None
        ),
        "mean_terminal_hp_fraction": (
            sum(max(0.0, min(1.0, e.frames[-1]["hp"] /
                max(1, e.frames[-1]["max_hp"]))) for e in completed) /
            len(completed) if completed else None
        ),
        "decisions": sum(len(e.decisions) for e in episodes),
        "searches": sum(e.searches for e in episodes),
        "search_transitions": sum(e.search_transitions for e in episodes),
        "search_overrides": sum(e.search_overrides for e in episodes),
        "search_override_rate": (
            sum(e.search_overrides for e in episodes) /
            sum(e.searches for e in episodes)
            if any(e.searches for e in episodes) else None
        ),
        "potion_actions": potion_actions,
        "rollout_profile": {
            "bridge_calls": sum(v["calls"] for v in operations.values()),
            "bridge_seconds": sum(v["seconds"] for v in operations.values()),
            "operations": dict(sorted(operations.items())),
            "actor_inference_calls": sum(e.actor_inference_calls for e in episodes),
            "actor_inference_seconds": sum(e.actor_inference_seconds for e in episodes),
        },
        "mean_potions_at_room_completion": (
            sum(potion_counts) / len(potion_counts) if potion_counts else None
        ),
        "distinct_public_resource_signatures": len(resource_signatures),
    }


def horizon_lambda_for_round(index: int, tau_rounds: float) -> float:
    """Slow, smooth increase; 1/(1-lambda) is effective geometric horizon."""
    if index < 0 or not math.isfinite(tau_rounds) or tau_rounds <= 0:
        raise ValueError("Invalid geometric-horizon schedule")
    return -math.expm1(-index / tau_rounds)


def _checkpoint(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name("." + path.name + ".tmp")
    torch.save(value, target)
    os.replace(target, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-h9", type=Path, required=True)
    p.add_argument("--macro-model", type=Path, default=None)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--ai-root", type=Path, default=Path.cwd())
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--runs-per-round", type=int, default=16)
    p.add_argument("--heldout-runs", type=int, default=8)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument(
        "--rollout-transport", choices=("legacy", "fused"), default="fused",
        help="Compare original multi-request frames to fused rollout frames.",
    )
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--hp-target", choices=("geometric", "post_room"),
                   default="geometric",
                   help="Slow geometric HP-to-Act curriculum, or fixed post-room control")
    p.add_argument("--horizon-tau-rounds", type=float, default=200.0,
                   help="Exponential lambda ramp time constant in training rounds")
    p.add_argument("--auxiliary-horizon-weight", type=float, default=0.5,
                   help="Weight of extra random-lambda critic regression")
    p.add_argument("--macro-prior-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--search-start-round", type=int, default=1)
    p.add_argument("--search-probability", type=float, default=1.0)
    p.add_argument("--max-searches", type=int, default=0,
                   help="Maximum combat PUCT roots/run; zero means every eligible decision")
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--search-depth", type=int, default=8)
    p.add_argument("--minibatch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.00001)
    p.add_argument("--study-seed", type=int, default=20261101)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.runs_per_round, args.heldout_runs, args.workers,
            args.max_decisions, args.search_start_round,
            args.simulations, args.search_depth, args.minibatch, args.epochs) <= 0
            or args.max_searches < 0 or args.study_seed < 0
            or not math.isfinite(args.horizon_tau_rounds)
            or args.horizon_tau_rounds <= 0
            or args.auxiliary_horizon_weight < 0
            or not 0 <= args.search_probability <= 1
            or not 0 < args.temperature <= 5
            or args.macro_prior_weight < 0 or args.lr <= 0):
        p.error("Invalid H16 pilot configuration")
    for path in (args.initial_h9, args.macro_model):
        if path is not None and not path.is_file():
            p.error(f"Missing checkpoint: {path}")
    source = {
        "schema": SCHEMA,
        "map_profile": EXPERIMENTAL_FIXED_ROUTE_PROFILE,
        "chance_provenance": "emulator-native-structure-ordinary-prototype-rng",
        "puct_provenance": "h16-native-act1-root-conditioned-privileged-v1",
        "initial_h9_sha256": sha256(args.initial_h9),
        "macro_sha256": sha256(args.macro_model) if args.macro_model else None,
        "emulator_revision": args.emulator_revision,
        "options": {
            key: value for key, value in vars(args).items()
            if key not in (
                "execute", "resume", "output_root", "initial_h9",
                "macro_model", "emulator_root", "ai_root", "device",
            )
        },
    }
    fingerprint = digest(source)
    print(json.dumps({
        "fingerprint": fingerprint, "experiment": source,
        "execute": args.execute, "device": args.device,
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu for a small smoke")
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    root = args.output_root
    progress_path = root / "progress.json"
    optimizer_path = root / "optimizer-state.pt"
    history: list[dict[str, Any]] = []
    if args.resume:
        if not progress_path.is_file() or not optimizer_path.is_file():
            p.error("--resume requires progress.json and optimizer-state.pt")
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != fingerprint:
            p.error("Cannot resume different H16 pilot configuration")
        history = progress["rounds"]
        if len(history) > args.rounds:
            p.error("Existing checkpoint exceeds configured round count")
        model = JointActPilotNet.load(
            root / "models" / f"joint-r{len(history):04d}.pt",
            device=args.device,
        )
    else:
        if progress_path.exists() or optimizer_path.exists():
            p.error("Output contains an existing run; use --resume or a new directory")
        model = JointActPilotNet(
            SharedObjectTransformer.load_checkpoint(args.initial_h9),
        ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.resume:
        saved = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint or saved["round"] != len(history):
            p.error("Optimizer and progress records disagree")
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_state"])
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_path = root / "actor-current.pt"
    heldout = [
        f"h16-fixedroute-heldout-{args.study_seed}-{i}"
        for i in range(args.heldout_runs)
    ]
    if not history:
        model.save(actor_path)
        baseline = collect(
            args, actor_path, seeds=heldout, horizon_lambda=0.0, search=False,
        )
        write_json(root / "baseline.json", summarize(baseline))
    for round_index in range(len(history), args.rounds):
        horizon_lambda = (
            horizon_lambda_for_round(round_index, args.horizon_tau_rounds)
            if args.hp_target == "geometric" else 0.0
        )
        model.save(actor_path)
        collect_started = time.perf_counter()
        train = collect(
            args, actor_path, horizon_lambda=horizon_lambda,
            seeds=[
                f"h16-fixedroute-train-{args.study_seed}-r{round_index}-{i}"
                for i in range(args.runs_per_round)
            ],
            search=round_index + 1 >= args.search_start_round,
        )
        collect_seconds = time.perf_counter() - collect_started
        completed = [ep for ep in train if ep.complete]
        if not completed:
            raise RuntimeError(
                "All Act trajectories censored; no valid terminal labels"
            )
        update_started = time.perf_counter()
        report = train_joint_cohort(
            model, optimizer, completed,
            horizon_lambda=horizon_lambda,
            hp_target=args.hp_target,
            auxiliary_horizon_weight=args.auxiliary_horizon_weight,
            temperature=args.temperature, minibatch=args.minibatch,
            epochs=args.epochs,
        )
        update_seconds = time.perf_counter() - update_started
        model_path = model_dir / f"joint-r{round_index + 1:04d}.pt"
        model.save(model_path)
        model.save(actor_path)
        evaluation_started = time.perf_counter()
        evaluation = collect(
            args, actor_path, seeds=heldout,
            horizon_lambda=horizon_lambda, search=False,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        simulated_transitions = sum(
            len(episode.decisions) + episode.search_transitions
            for episode in train
        )
        row = {
            "round": round_index + 1,
            "horizon_lambda": horizon_lambda,
            "timing": {
                "collection_seconds": collect_seconds,
                "update_seconds": update_seconds,
                "evaluation_seconds": evaluation_seconds,
                "approx_emulator_transitions_per_second": (
                    simulated_transitions / collect_seconds
                    if collect_seconds > 0 else None
                ),
            },
            "effective_horizon": (
                1.0 / (1.0 - horizon_lambda)
                if horizon_lambda < 1.0 else None
            ),
            "train": summarize(train), "update": report,
            "heldout": summarize(evaluation),
            "checkpoint": str(model_path),
            "checkpoint_sha256": sha256(model_path),
        }
        history.append(row)
        _checkpoint(optimizer_path, {
            "fingerprint": fingerprint,
            "round": len(history),
            "optimizer": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        })
        write_json(progress_path, {
            "fingerprint": fingerprint,
            "experiment": source,
            "baseline": str(root / "baseline.json"),
            "rounds": history,
        })
        print("[h16-act1-joint] " + json.dumps(row, sort_keys=True), flush=True)
    print("[h16-act1-joint] completed", len(history), "rounds", flush=True)


if __name__ == "__main__":
    main()
