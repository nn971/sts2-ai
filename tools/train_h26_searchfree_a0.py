#!/usr/bin/env python3
"""H26: search-free Act rollouts and combat actor-critic with tactical inputs.

No PUCT roots, fight reconstruction, or search-policy distillation occur.
Both policies have separate weights. Combat uses observed terminal fight
outcomes, bounded potential shaping and frozen macro continuation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from sts2_ai.models.h16_batch_actor import SharedH16Service
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotEpisode,
)
from sts2_ai.training.h24_joint_act_model import H24JointActPilotNet
from sts2_ai.training.h25_alternating import fit_macro_reinforce
from sts2_ai.training.h26_combat_rl import fit_search_free_combat

try:
    from tools.train_h16_joint_act_pilot import (
        _worker_init,
        _worker_run_h25,
        seed_namespace,
        sha256,
        summarize,
        write_json,
    )
except ModuleNotFoundError:
    from train_h16_joint_act_pilot import (
        _worker_init,
        _worker_run_h25,
        seed_namespace,
        sha256,
        summarize,
        write_json,
    )

H26_SCHEMA = "sts2-h26-search-free-public-threat-combat-actor-critic-v1"


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    """Stage optimizer data in the destination directory and publish atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix="." + path.name + "-",
        suffix=".tmp", delete=False,
    ) as stream:
        pending = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    os.replace(pending, path)


def prune_h25_checkpoints(
    root: Path, *, newest: int, keep_last: int = 3, archive_every: int = 25,
) -> list[str]:
    """After commit, retain recent checkpoints and periodic historical samples."""
    if newest < 1 or keep_last < 1 or archive_every < 1:
        raise ValueError("Invalid H26 checkpoint retention")
    removed: list[str] = []
    for path in (root / "models").glob("*.pt"):
        prefix, sep, suffix = path.stem.rpartition("-r")
        if not sep or prefix not in ("macro", "combat", "optimizers"):
            continue
        if not suffix.isdigit():
            continue
        number = int(suffix)
        if number >= newest - keep_last + 1 or number % archive_every == 0:
            continue
        path.unlink()
        removed.append(path.name)
    return removed


def _stable_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": H26_SCHEMA,
        "initial_h22_sha256": sha256(args.initial_h22),
        "initial_h25_macro_sha256": (
            sha256(args.initial_h25_macro) if args.initial_h25_macro else None
        ),
        "initial_h25_combat_sha256": (
            sha256(args.initial_h25_combat) if args.initial_h25_combat else None
        ),
        "emulator_revision": args.emulator_revision,
        "ascension": 0, "map": "native-overgrowth-generated-default-v111",
        "rounds": args.rounds, "runs_per_round": args.runs_per_round,
        "heldout_runs": args.heldout_runs, "study_seed": args.study_seed,
        "workers": args.workers, "combat_gamma": args.combat_gamma,
        "max_combat_samples": args.max_combat_samples,
        "macro_update_every": args.macro_update_every,
        "macro_progress_weight": args.macro_progress_weight,
        "macro_progress_loss_weight": args.macro_progress_loss_weight,
        "minibatch": args.minibatch, "epochs": args.epochs,
        "lr": args.lr, "temperature": args.temperature,
        "actor_max_batch": args.actor_max_batch,
        "actor_fixed_shape_batch_size": args.actor_fixed_shape_batch_size,
        "max_decisions": args.max_decisions,
    }


def _collect(
    args: argparse.Namespace,
    macro_path: Path, combat_path: Path, seeds: Sequence[str],
    *, stage: str, round_number: int,
) -> tuple[list[PilotEpisode], dict[str, int]]:
    """Two independently versioned, shared-CUDA actors; frozen for this cohort."""
    source = {
        "checkpoint": str(macro_path.resolve()),
        "h25_combat_checkpoint": str(combat_path.resolve()),
        "representation": "h21",
        "h24_selection": {
            "hp": True, "gold": True, "potion": True, "relic": False,
        },
        "h24_vocabulary": {"potion_ids": (), "relic_ids": ()},
        "macro_model": None, "macro_prior_weight": 0.0,
        "horizon_lambda": 0.0, "temperature": args.temperature,
        "actor_inference": "shared", "actor_device": args.device,
        "combat_credit_weight": 0.25, "loss_progress_cap": 0.2,
        "h25_hp_bias": 0.15,
        "h25_fights_per_run": 0,
        "h25_combat_simulations": 1,
        "h25_max_combat_decisions": 256,
        "h26_no_search": True,
        "h26_tactical": True,
        "max_decisions": args.max_decisions,
        "max_simulation_steps": 512,
        "rollout_transport": "fused",
        "map_profile_id": None,
        "ascension": 0, "study_seed": args.study_seed,
        "emulator_revision": args.emulator_revision,
        "emulator_root": str(args.emulator_root.resolve()),
        "ai_root": str(Path.cwd().resolve()),
        "h25_status_root": str((args.output_root / "workers").resolve()),
        "h25_round_number": round_number,
    }
    episodes: list[PilotEpisode | None] = [None] * len(seeds)
    summary: dict[str, int] = {
        "roots": 0, "transitions": 0, "examples": 0,
    }
    started = time.perf_counter()
    with ExitStack() as stack:
        macro_service = stack.enter_context(SharedH16Service(
            checkpoint=macro_path.resolve(), digest=sha256(macro_path),
            workers=args.workers, device=args.device,
            horizon_lambda=0.0, max_batch=args.actor_max_batch,
            fixed_shape_batch_size=args.actor_fixed_shape_batch_size,
            h21_collator="packed",
        ))
        combat_service = stack.enter_context(SharedH16Service(
            checkpoint=combat_path.resolve(), digest=sha256(combat_path),
            workers=args.workers, device=args.device,
            horizon_lambda=0.0, max_batch=args.actor_max_batch,
            fixed_shape_batch_size=args.actor_fixed_shape_batch_size,
            h21_collator="packed",
        ))
        source["actor_transport_handle"] = macro_service.transport
        source["h25_combat_actor_transport_handle"] = combat_service.transport
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init, initargs=(source,),
        ) as pool:
            pending = {
                pool.submit(_worker_run_h25, seed): index
                for index, seed in enumerate(seeds)
            }
            completed = 0
            def pulse() -> None:
                write_json(args.output_root / "live.json", {
                    "schema": H26_SCHEMA, "round": round_number,
                    "stage": stage, "timestamp_unix": time.time(),
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                    "completed_episodes": completed, "total_episodes": len(seeds),
                    "unfinished_episodes": len(pending),
                    "search_from_completed_episodes": dict(summary),
                    "note": "Search counters include only completed worker tasks.",
                })
            pulse()
            while pending:
                done, _ = wait(
                    pending, timeout=args.heartbeat_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    index = pending.pop(future)
                    result = future.result()
                    episodes[index] = result["episode"]
                    # A worker must never perform search under the H26 contract.
                    if result["search"]["roots"] or result["examples"]:
                        raise RuntimeError("H26 search-free rollout unexpectedly searched")
                    completed += 1
                pulse()
    return [ep for ep in episodes if ep is not None], summary


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-h22", type=Path, required=True)
    p.add_argument("--initial-h25-macro", type=Path)
    p.add_argument("--initial-h25-combat", type=Path)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--runs-per-round", type=int, default=64)
    p.add_argument("--heldout-runs", type=int, default=16)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--study-seed", type=int, default=20261120)
    p.add_argument("--combat-gamma", type=float, default=0.99)
    p.add_argument("--max-combat-samples", type=int, default=4096)
    p.add_argument("--macro-update-every", type=int, default=1)
    p.add_argument("--macro-progress-weight", type=float, default=0.25,
                   help="Weight of graded floor progress in macro policy advantage")
    p.add_argument("--macro-progress-loss-weight", type=float, default=0.5,
                   help="Weight for separate floor-progress critic MSE")
    p.add_argument("--minibatch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--actor-max-batch", type=int, default=4)
    p.add_argument("--actor-fixed-shape-batch-size", type=int, default=4)
    # Full-depth search controls belong exclusively to the H26 experiment.
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--heartbeat-seconds", type=float, default=20.0)
    p.add_argument("--keep-last-rounds", type=int, default=3,
                   help="Keep latest N immutable checkpoints for recovery")
    p.add_argument("--archive-every-rounds", type=int, default=25,
                   help="Retain periodic historical checkpoints every N rounds")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="Stop after a committed round at cumulative active wall budget")
    p.add_argument("--resume", action="store_true",
                   help="Restore both models and optimizers from the last committed round")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.runs_per_round, args.heldout_runs, args.workers,
            args.max_combat_samples, args.macro_update_every, args.minibatch,
            args.epochs,
            args.max_decisions, args.actor_max_batch,
            args.actor_fixed_shape_batch_size) <= 0
            or not 0 < args.combat_gamma <= 1
            or not 0 <= args.macro_progress_weight <= 1
            or not 0 <= args.macro_progress_loss_weight <= 1
            or not 0 < args.temperature <= 5
            or (args.initial_h25_macro is None) != (args.initial_h25_combat is None)
            or args.actor_fixed_shape_batch_size > args.actor_max_batch
            or not math.isfinite(args.heartbeat_seconds)
            or args.heartbeat_seconds <= 0 or args.lr <= 0
            or args.study_seed < 0
            or min(args.keep_last_rounds, args.archive_every_rounds) <= 0
            or (args.max_wall_seconds is not None
                and (not math.isfinite(args.max_wall_seconds)
                     or args.max_wall_seconds <= 0))):
        p.error("Invalid H26 training configuration")
    for name in (
        "initial_h22", "emulator_root", "initial_h25_macro", "initial_h25_combat",
    ):
        path = getattr(args, name)
        if path is not None and not path.exists():
            p.error(f"Missing {name}: {path}")
    return args


def main() -> None:
    args = _args()
    identity = _stable_identity(args)
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode(),
    ).hexdigest()
    print(json.dumps({
        "fingerprint": fingerprint, "experiment": identity,
        "execute": args.execute,
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("H26 needs a functioning CUDA shared actor")
    root = args.output_root
    progress_path = root / "progress.json"
    if args.resume:
        if not progress_path.is_file():
            raise ValueError("--resume requires committed progress.json")
    elif progress_path.exists() or (root / "optimizer-current.pt").exists():
        raise ValueError("Output contains a previous experiment; use --resume or a new root")
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    (root / "models").mkdir(parents=True, exist_ok=True)
    macro_path = root / "macro-current.pt"
    combat_path = root / "combat-current.pt"
    history: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    if args.resume:
        progress = json.loads(progress_path.read_text())
        if progress.get("fingerprint") != fingerprint:
            raise ValueError("H26 resume fingerprint differs from committed experiment")
        history = progress["rounds"]
        if not history or len(history) > args.rounds:
            raise ValueError("H26 resume has no valid committed round")
        committed = history[-1]
        number = len(history)
        if committed["round"] != number:
            raise ValueError("H26 progress history is not sequential")
        saved_macro = root / "models" / f"macro-r{number:04d}.pt"
        saved_combat = root / "models" / f"combat-r{number:04d}.pt"
        saved_opt = root / "models" / f"optimizers-r{number:04d}.pt"
        if any(not item.is_file() for item in (saved_macro, saved_combat, saved_opt)):
            raise ValueError("H26 committed checkpoint files are missing")
        if (committed["macro_checkpoint_sha256"] != sha256(saved_macro)
                or committed["combat_checkpoint_sha256"] != sha256(saved_combat)
                or committed["optimizer_checkpoint_sha256"] != sha256(saved_opt)):
            raise ValueError("H26 committed checkpoint digest mismatch")
        macro = JointActPilotNet.load(saved_macro, device=args.device)
        combat = JointActPilotNet.load(saved_combat, device=args.device)
        if (not getattr(macro, "h24_act_only", False)
                or not getattr(combat, "h24_act_only", False)
                or getattr(macro, "macro_progress_head", None) is None
                or getattr(combat, "macro_progress_head", None) is not None
                or not getattr(combat, "h26_tactical", False)
                or getattr(macro, "h26_tactical", False)):
            raise ValueError("H26 resume requires Act-only H24 checkpoints")
        macro_optimizer = torch.optim.AdamW(macro.parameters(), lr=args.lr)
        combat_optimizer = torch.optim.AdamW(combat.parameters(), lr=args.lr)
        saved = torch.load(saved_opt, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint or saved["round"] != number:
            raise ValueError("H26 optimizer/progress identity mismatch")
        macro_optimizer.load_state_dict(saved["macro_optimizer"])
        combat_optimizer.load_state_dict(saved["combat_optimizer"])
        torch.set_rng_state(saved["rng_state"])
        prior_elapsed = float(progress["elapsed_wall_seconds"])
        macro.save(macro_path)
        combat.save(combat_path)
        print(f"[h26] resumed from committed round {number}", flush=True)
    else:
        if args.initial_h25_macro is not None:
            macro = JointActPilotNet.load(args.initial_h25_macro, device=args.device)
            combat = JointActPilotNet.load(args.initial_h25_combat, device=args.device)
            if (not getattr(macro, "h24_act_only", False)
                    or not getattr(combat, "h24_act_only", False)
                    or getattr(macro, "macro_progress_head", None) is None
                    or getattr(combat, "macro_progress_head", None) is not None):
                raise ValueError("H26 requires H26 v2 independent Act-only checkpoints")
        else:
            source = JointActPilotNet.load(args.initial_h22, device=args.device)
            macro = H24JointActPilotNet.warmstart_h22(
                source, act_only=True, search_act_mix=1.0,
                macro_progress=True,
            ).to(args.device)
            combat = copy.deepcopy(macro)
            combat.macro_progress_head = None
            del source
        macro.h26_tactical = False
        combat.h26_tactical = True
        macro_optimizer = torch.optim.AdamW(macro.parameters(), lr=args.lr)
        combat_optimizer = torch.optim.AdamW(combat.parameters(), lr=args.lr)
        macro.save(macro_path)
        combat.save(combat_path)
    wall_started = time.perf_counter()
    namespace = seed_namespace(0, "generated")
    heldout = [
        f"{namespace}-heldout-{args.study_seed}-{i}"
        for i in range(args.heldout_runs)
    ]
    if (args.max_wall_seconds is not None
            and prior_elapsed >= args.max_wall_seconds):
        print("[h26] wall budget already reached; no further rounds", flush=True)
        return
    for round_index in range(len(history), args.rounds):
        round_number = round_index + 1
        seeds = [
            f"{namespace}-train-{args.study_seed}-r{round_index}-{i}"
            for i in range(args.runs_per_round)
        ]
        collected_at = time.perf_counter()
        train, search = _collect(
            args, macro_path, combat_path, seeds,
            stage="search_free_full_acts", round_number=round_number,
        )
        collection_seconds = time.perf_counter() - collected_at
        if any(not ep.complete for ep in train):
            # Never insert censored trajectories into macro REINFORCE.
            print("[h26] censored episodes excluded from macro updates", flush=True)
        write_json(root / "live.json", {
            "schema": H26_SCHEMA, "round": round_number,
            "stage": "training", "timestamp_unix": time.time(),
            "search": search,
        })
        combat_update = fit_search_free_combat(
            combat, macro, combat_optimizer, train,
            gamma=args.combat_gamma,
            minibatch=args.minibatch, epochs=args.epochs,
            temperature=args.temperature,
            max_samples=args.max_combat_samples,
            seed=args.study_seed + round_number,
        )
        macro_update = (
            fit_macro_reinforce(
                macro, macro_optimizer, train,
                minibatch=args.minibatch, epochs=args.epochs,
                temperature=args.temperature,
                progress_weight=args.macro_progress_weight,
                progress_loss_weight=args.macro_progress_loss_weight,
            )
            if round_number % args.macro_update_every == 0 else None
        )
        # Distinct model objects/optimizers prevent cross-phase policy drift.
        macro.save(root / "models" / f"macro-r{round_number:04d}.pt")
        combat.save(root / "models" / f"combat-r{round_number:04d}.pt")
        macro.save(macro_path)
        combat.save(combat_path)
        eval_start = time.perf_counter()
        held, _ = _collect(
            args, macro_path, combat_path, heldout,
            stage="heldout", round_number=round_number,
        )
        row = {
            "round": round_number, "train": summarize(train),
            "heldout": summarize(held),
            "combat_search": search,
            "combat_update": combat_update,
            "macro_update": macro_update,
            "timing": {
                "collection_seconds": collection_seconds,
                "evaluation_seconds": time.perf_counter() - eval_start,
            },
            "macro_checkpoint": str(root / "models" / f"macro-r{round_number:04d}.pt"),
            "combat_checkpoint": str(root / "models" / f"combat-r{round_number:04d}.pt"),
        }
        optimizer_checkpoint = root / "models" / f"optimizers-r{round_number:04d}.pt"
        # Model snapshots and optimizer are durable before progress.json commits.
        _atomic_torch_save({
            "fingerprint": fingerprint, "round": round_number,
            "macro_optimizer": macro_optimizer.state_dict(),
            "combat_optimizer": combat_optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }, optimizer_checkpoint)
        row.update({
            "macro_checkpoint_sha256": sha256(
                root / "models" / f"macro-r{round_number:04d}.pt"),
            "combat_checkpoint_sha256": sha256(
                root / "models" / f"combat-r{round_number:04d}.pt"),
            "optimizer_checkpoint_sha256": sha256(optimizer_checkpoint),
        })
        history.append(row)
        elapsed_total = prior_elapsed + time.perf_counter() - wall_started
        write_json(progress_path, {
            "fingerprint": fingerprint, "experiment": identity,
            "elapsed_wall_seconds": elapsed_total,
            "rounds": history,
        })
        # Retention runs only after the latest progress.json commit.
        pruned = prune_h25_checkpoints(
            root, newest=round_number, keep_last=args.keep_last_rounds,
            archive_every=args.archive_every_rounds,
        )
        print("[h26] " + json.dumps({
            "round": round_number,
            "train_cleared": row["train"]["cleared"],
            "heldout_cleared": row["heldout"]["cleared"],
            "combat_examples": combat_update["examples"],
            "search_transitions": 0,
            "combat_steps": combat_update["steps"],
            "macro_steps": macro_update["steps"] if macro_update else 0,
            "collection_seconds": collection_seconds,
            "elapsed_wall_seconds": round(elapsed_total, 1),
            "pruned_checkpoints": len(pruned),
        }), flush=True)
        if (args.max_wall_seconds is not None
                and elapsed_total >= args.max_wall_seconds):
            print(
                f"[h26] wall budget reached after committed round {round_number}; "
                "resume with identical experiment and --resume",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
