#!/usr/bin/env python3
"""H14 M2: native A10 macro PPO, with an independently frozen H9 combat actor.

Each episode uses one fair public emulator trajectory. H9 receives its
unchanged H7 input codec; the H14 macro actor/value network sees the new
audited persistent-state features. Macro samples from censored runs without
a certified Act-1 clear are excluded, and positive labels require actual
observed Act-1 clearance. No predicted combat outcomes or fabricated rewards.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run
from sts2_ai.evaluation import play_run
from sts2_ai.models.h14_hierarchical_agent import (
    FrozenH9CombatActor,
    H14HierarchicalAgent,
    H14MacroActionSample,
)
from sts2_ai.models.h14_macro_actor_critic import H14MacroActorCritic
from sts2_ai.models.shared_game_objects import encode_public_decision
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h14_macro_gae import macro_gae

SCHEMA = "sts2-h14-native-macro-ppo-gae-v2"
_WORKER_BACKEND: JsonlEmulatorBackend | None = None
_WORKER_H9: FrozenH9CombatActor | None = None
_WORKER_MACRO: H14MacroActorCritic | None = None
_WORKER_MACRO_SHA: str = ""


@dataclass(frozen=True)
class Episode:
    seed: str
    outcome: str
    censored: bool
    act1_cleared: bool | None
    decisions: int
    terminal_floor: int
    samples: tuple[H14MacroActionSample, ...]
    macro_history: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix="." + path.name + "-", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def close_worker() -> None:
    global _WORKER_BACKEND
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.__exit__(None, None, None)
        _WORKER_BACKEND = None


def init_worker(
    emulator_root: str, emulator_revision: str,
    h9_path: str, h9_sha: str, macro_path: str, macro_sha: str,
) -> None:
    global _WORKER_BACKEND, _WORKER_H9, _WORKER_MACRO, _WORKER_MACRO_SHA
    torch.set_num_threads(1)
    _WORKER_BACKEND = JsonlEmulatorBackend(
        emulator_root=Path(emulator_root),
        expected_emulator_revision=emulator_revision,
        build=False,
    )
    _WORKER_BACKEND.__enter__()
    atexit.register(close_worker)
    _WORKER_H9 = FrozenH9CombatActor.load_pinned(
        Path(h9_path), expected_sha256=h9_sha,
    )
    if sha256_file(Path(macro_path)) != macro_sha:
        raise ValueError("Worker H14 actor checkpoint digest mismatch")
    _WORKER_MACRO = H14MacroActorCritic.load_checkpoint(
        Path(macro_path),
    ).eval()
    _WORKER_MACRO_SHA = macro_sha


def run_episode(request: tuple[int, str, float, int]) -> tuple[int, Episode]:
    index, seed, temperature, max_decisions = request
    if _WORKER_BACKEND is None or _WORKER_H9 is None or _WORKER_MACRO is None:
        raise RuntimeError("Uninitialized H14 actor worker")
    agent = H14HierarchicalAgent(
        _WORKER_H9, _WORKER_MACRO, episode_seed=seed,
        macro_checkpoint_sha256=_WORKER_MACRO_SHA,
        temperature=temperature,
    )
    result = play_run(
        _WORKER_BACKEND, agent, seed=seed,
        policy=InformationPolicy(FAIR_POLICY_ID),
        ascension=10, max_decisions=max_decisions,
        environment=NATIVE_OVERGROWTH,
        stop_after_act1_clear=True,
    )
    certified = (
        True if result.outcome == "act1_clear"
        else False if result.outcome == "defeat" else None
    )
    return index, Episode(
        seed, result.outcome, result.censored, certified,
        result.decisions,
        (16 if certified is True else (
            result.terminal_floor if type(result.terminal_floor) is int else 0
        )),
        tuple(agent.samples) if certified is not None else (),
        tuple(agent.macro_history),
    )


def train_round(
    network: H14MacroActorCritic, optimizer: torch.optim.Optimizer,
    episodes: list[Episode], *, device: str, ppo_epochs: int,
    minibatch: int, sample_limit: int, round_index: int,
    temperature: float, gamma: float, gae_lambda: float,
    floor_reward: float,
) -> dict[str, Any]:
    dataset: list[tuple[H14MacroActionSample, float, float, float]] = []
    for episode in episodes:
        if episode.act1_cleared is None or not episode.samples:
            continue
        targets = macro_gae(
            [sample.progress_floor for sample in episode.samples],
            [sample.behavior_value for sample in episode.samples],
            final_floor=episode.terminal_floor,
            act1_cleared=episode.act1_cleared,
            gamma=gamma, gae_lambda=gae_lambda, floor_reward=floor_reward,
        )
        dataset.extend(
            (sample, target.value_target, target.advantage, target.reward)
            for sample, target in zip(episode.samples, targets)
        )
    if len(dataset) > sample_limit:
        rng = random.Random(f"h14-macro-sample-{round_index}")
        dataset = rng.sample(dataset, sample_limit)
    if not dataset:
        return {"samples": 0, "updates": 0}
    # One frozen-behavior cohort supplies advantage normalization for all
    # epochs; never normalize per tiny PPO minibatch.
    advantage_mean = sum(item[2] for item in dataset) / len(dataset)
    advantage_variance = sum(
        (item[2] - advantage_mean) ** 2 for item in dataset
    ) / len(dataset)
    advantage_std = math.sqrt(advantage_variance)
    if advantage_std < 1e-6:
        normalized = [0.0 for _ in dataset]
    else:
        normalized = [
            (item[2] - advantage_mean) / advantage_std for item in dataset
        ]
    rng = random.Random(f"h14-macro-minibatches-{round_index}")
    network.train()
    stats: list[dict[str, float]] = []
    for _ in range(ppo_epochs):
        order = list(range(len(dataset)))
        rng.shuffle(order)
        for start in range(0, len(order), minibatch):
            indices = order[start:start + minibatch]
            batch = [dataset[index] for index in indices]
            encoded = [item.decision for item, _, _, _ in batch]
            selected = torch.tensor([item.selected for item, _, _, _ in batch])
            logp = torch.tensor([
                item.behavior_log_probability for item, _, _, _ in batch
            ])
            old_values = torch.tensor([
                item.behavior_value for item, _, _, _ in batch
            ])
            returns = torch.tensor([target for _, target, _, _ in batch])
            adv = torch.tensor([normalized[index] for index in indices])
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = network.ppo_loss(
                encoded, selected_actions=selected,
                behavior_log_probabilities=logp,
                behavior_values=old_values, returns=returns, advantages=adv,
                behavior_temperature=temperature,
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite H14 macro PPO loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            stats.append(metrics)
    network.eval()
    return {
        "samples": len(dataset), "updates": len(stats),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "value_target_mean": sum(item[1] for item in dataset) / len(dataset),
        "reward_mean": sum(item[3] for item in dataset) / len(dataset),
        **{name: sum(item[name] for item in stats) / len(stats)
           for name in ("policy", "value", "entropy")},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h9-checkpoint", type=Path, required=True)
    p.add_argument("--h9-sha256", required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--runs-per-round", type=int, default=64)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--seed-prefix", default="h14-a10-macro-train-v1")
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--lr", type=float, default=0.00003)
    p.add_argument("--device", default="cuda")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--feedforward", type=int, default=1536)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch", type=int, default=8)
    p.add_argument("--ppo-sample-limit", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.runs_per_round, args.workers, args.max_decisions,
            args.width, args.layers, args.heads, args.feedforward,
            args.ppo_epochs, args.ppo_minibatch, args.ppo_sample_limit) < 1
            or args.temperature <= 0 or args.lr <= 0 or not args.seed_prefix):
        p.error("Invalid H14 training parameters")
    if not args.h9_checkpoint.is_file() or not args.emulator_dll.is_file():
        p.error("Missing pinned H9 checkpoint or emulator DLL")
    h9_digest = sha256_file(args.h9_checkpoint)
    if h9_digest != args.h9_sha256:
        p.error("H9 checkpoint SHA does not match the expected digest")
    from subprocess import check_output
    revision = check_output(
        ["git", "-C", str(args.emulator_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != args.emulator_revision:
        p.error("Emulator revision mismatch")
    cfg = H7Config(
        width=args.width, layers=args.layers, heads=args.heads,
        feedforward=args.feedforward,
    )
    provenance = {
        "h9_sha256": h9_digest,
        "emulator_revision": revision,
        "emulator_dll_sha256": sha256_file(args.emulator_dll),
        "environment": NATIVE_OVERGROWTH,
        "ascension": "10",
        "observation_policy": FAIR_POLICY_ID,
    }
    config = {
        "schema": SCHEMA, "model": asdict(cfg), "provenance": provenance,
        "seed_prefix": args.seed_prefix, "temperature": args.temperature,
        "lr": args.lr, "max_decisions": args.max_decisions,
        "runs_per_round": args.runs_per_round,
        "workers": args.workers, "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch": args.ppo_minibatch,
        "ppo_sample_limit": args.ppo_sample_limit,
    }
    print(json.dumps({"mode": "execute" if args.execute else "preflight",
                      "identity": config}, sort_keys=True), flush=True)
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=revision,
        build=False,
    ) as backend:
        probe = reset_training_run(
            backend, f"{args.seed_prefix}-r0001-run00000", NATIVE_OVERGROWTH, 10,
        )
        try:
            observed = backend.observe(probe, InformationPolicy(FAIR_POLICY_ID))
            public = json.loads(observed.payload_json)
            menu = tuple(backend.legal_actions(probe))
            encoded = encode_public_decision(public, menu)
            print("[h14-native-preflight] " + json.dumps({
                "map_profile": public.get("map_generation_profile_id"),
                "starter": [card.get("card_id") for card in public["deck"]
                            if card.get("card_id") == "proto.curse.ascenders_bane"],
                "macro_actions": len(encoded.actions),
            }, sort_keys=True), flush=True)
        finally:
            backend.release_many([probe])
    if not args.execute:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA device requested but not available")
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity = args.output_root / "identity.json"
    latest = args.output_root / "latest.json"
    if identity.exists():
        if json.loads(identity.read_text()) != config:
            p.error("Output root contains a different training identity")
        if latest.exists() and not args.resume:
            p.error("Existing committed run requires --resume")
    else:
        if args.resume:
            p.error("No training identity to resume")
        atomic_json(identity, config)
    torch.set_num_threads(1)
    torch.manual_seed(int(hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16], 16) % (2**63 - 1))
    network = H14MacroActorCritic(cfg).to(args.device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    first_round = 1
    if args.resume and latest.exists():
        info = json.loads(latest.read_text())
        first_round = int(info["round"]) + 1
        checkpoint = args.output_root / info["checkpoint"]
        if sha256_file(checkpoint) != info["checkpoint_sha256"]:
            p.error("Resume checkpoint digest mismatch")
        network = H14MacroActorCritic.load_checkpoint(
            checkpoint, map_location=args.device,
        )
        optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
        state_file = args.output_root / info["optimizer"]
        if sha256_file(state_file) != info["optimizer_sha256"]:
            p.error("Resume optimizer digest mismatch")
        optimizer.load_state_dict(
            torch.load(state_file, map_location=args.device, weights_only=True),
        )
    models = args.output_root / "models"
    models.mkdir(exist_ok=True)
    for round_index in range(first_round, args.rounds + 1):
        source = models / f"behavior-r{round_index:04d}.pt"
        network.save_checkpoint(source, provenance=provenance)
        source_sha = sha256_file(source)
        requests = [
            (index, f"{args.seed_prefix}-r{round_index:04d}-run{index:05d}",
             args.temperature, args.max_decisions)
            for index in range(args.runs_per_round)
        ]
        completed: dict[int, Episode] = {}
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn"),
            initializer=init_worker,
            initargs=(str(args.emulator_root.resolve()), revision,
                      str(args.h9_checkpoint.resolve()), h9_digest,
                      str(source.resolve()), source_sha),
        ) as executor:
            futures = {executor.submit(run_episode, item): item[0]
                       for item in requests}
            for future in as_completed(futures):
                index, episode = future.result()
                completed[index] = episode
                print(json.dumps({
                    "round": round_index, "run": index,
                    "seed": episode.seed, "outcome": episode.outcome,
                    "act1_cleared": episode.act1_cleared,
                    "censored": episode.censored,
                    "macro_samples": len(episode.samples),
                    "macro_actions": len(episode.macro_history),
                }, sort_keys=True), flush=True)
        if set(completed) != set(range(args.runs_per_round)):
            raise RuntimeError("Incomplete native H14 source-run cohort")
        episodes = [completed[i] for i in range(args.runs_per_round)]
        history_path = args.output_root / f"round-{round_index:04d}-history.jsonl"
        with history_path.open("w", encoding="utf-8") as history_file:
            for episode in episodes:
                history_file.write(json.dumps({
                    "source_run_seed": episode.seed,
                    "outcome": episode.outcome,
                    "censored": episode.censored,
                    "act1_cleared": episode.act1_cleared,
                    "macro_decisions": episode.macro_history,
                }, sort_keys=True) + "\n")
        stats = train_round(
            network, optimizer, episodes, device=args.device,
            ppo_epochs=args.ppo_epochs, minibatch=args.ppo_minibatch,
            sample_limit=args.ppo_sample_limit, round_index=round_index,
            temperature=args.temperature,
        )
        checkpoint = models / f"macro-r{round_index:04d}.pt"
        network.save_checkpoint(checkpoint, provenance=provenance)
        opt = models / f"optimizer-r{round_index:04d}.pt"
        torch.save(optimizer.state_dict(), opt)
        record = {
            "round": round_index,
            "checkpoint": str(checkpoint.relative_to(args.output_root)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer": str(opt.relative_to(args.output_root)),
            "optimizer_sha256": sha256_file(opt),
            "behavior_sha256": source_sha,
            "history_file": history_path.name,
            "history_sha256": sha256_file(history_path),
            "outcomes": {
                "act1_cleared": sum(e.act1_cleared is True for e in episodes),
                "certified_defeats": sum(e.act1_cleared is False for e in episodes),
                "unlabeled": sum(e.act1_cleared is None for e in episodes),
            },
            "training": stats,
            "source_seeds": [e.seed for e in episodes],
        }
        atomic_json(args.output_root / f"round-{round_index:04d}.json", record)
        atomic_json(latest, record)
        print("[h14-round] " + json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
