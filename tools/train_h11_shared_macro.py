#!/usr/bin/env python3
"""H11 M2: train the shared Transformer's macro heads on native A10 full runs.

The H9 combat policy and all shared object/attention weights stay bitwise
frozen. Only the existing macro policy and value heads receive PPO gradients.
An isolated full-run seed supplies each experience; no offline teacher,
fabricated victory, private state, or counterfactual combat label is used.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.models.h7_public_policy import TorchH7PublicPolicy
from sts2_ai.models.shared_game_objects import encode_public_decision
from sts2_ai.models.shared_transformer import SharedObjectTransformer
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h10_native_transformer import build_gae
from sts2_ai.training.h7_combat_ppo import H7CombatPPO, PreparedH7
from train_h7_reference_combat import atomic_torch

SCHEMA = "sts2-h11-frozen-combat-shared-macro-ppo-v1"
_WORKER_BACKEND: JsonlEmulatorBackend | None = None
_WORKER_POLICY: TorchH7PublicPolicy | None = None


@dataclass(frozen=True)
class RunRollout:
    seed: str
    outcome: str
    censored: bool
    act_clear: bool
    frontier_progress: float
    decisions: int
    macro_decisions: int
    samples: tuple[PreparedH7, ...]


def _macro_trainable(name: str) -> bool:
    return name.startswith("policy_heads.1.") or name.startswith("value_heads.1.")


def freeze_for_macro(model: SharedObjectTransformer) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_macro_trainable(name))
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("H11 found no macro parameters to optimize")


class FrozenCombatSampledMacro:
    """One public-only shared checkpoint, greedy combat and sampled macro."""

    def __init__(self, policy: TorchH7PublicPolicy, *,
                 seed: str, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("Invalid H11 macro temperature")
        self.policy = policy
        self.temperature = temperature
        self.rng = random.Random(int(canonical_sha256([
            "h11-macro", policy.model_id, seed, temperature,
        ])[:16], 16))
        self.policy_id = "h11-frozen-combat-macro-" + policy.model_id
        self.entries: list[tuple[Any, int, float, float]] = []

    def choose(self, observation, legal_actions) -> Decision:
        if observation.policy_id != FAIR_POLICY_ID or not legal_actions:
            raise ValueError("H11 requires fair public observation and legal menu")
        state = json.loads(observation.payload_json)
        if not isinstance(state, dict):
            raise ValueError("Malformed H11 public observation")
        scores = self.policy.evaluate(observation, legal_actions)
        if len(scores.action_logits) != len(legal_actions):
            raise ValueError("H11 action logits are not aligned with legal menu")
        if isinstance(state.get("combat"), dict):
            index = max(range(len(legal_actions)),
                        key=lambda i: (scores.action_logits[i],
                                       legal_actions[i].action_id))
        else:
            encoded = encode_public_decision(state, legal_actions)
            # Match the CPU singleton policy's fp32 behavior distribution.
            logits = torch.tensor(scores.action_logits, dtype=torch.float32)
            logp = torch.log_softmax(logits / self.temperature, dim=0)
            index = self.rng.choices(
                range(len(legal_actions)), weights=logp.exp().tolist(), k=1,
            )[0]
            self.entries.append((
                encoded, index, float(logp[index]), float(scores.value),
            ))
        return Decision(legal_actions[index], policy_name=self.policy_id)


def _worker_cleanup() -> None:
    global _WORKER_BACKEND
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.__exit__(None, None, None)
        _WORKER_BACKEND = None


def _worker_init(repo_root: str, revision: str, checkpoint: str) -> None:
    global _WORKER_BACKEND, _WORKER_POLICY
    torch.set_num_threads(1)
    _WORKER_BACKEND = JsonlEmulatorBackend(
        emulator_root=Path(repo_root),
        expected_emulator_revision=revision,
        build=False,
    )
    _WORKER_BACKEND.__enter__()
    atexit.register(_worker_cleanup)
    model = SharedObjectTransformer.load_checkpoint(Path(checkpoint)).eval()
    _WORKER_POLICY = TorchH7PublicPolicy(
        model, model_id=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    )


def _worker_run(request: tuple[int, str, float, int]) -> tuple[int, RunRollout]:
    index, seed, temperature, max_decisions = request
    if _WORKER_BACKEND is None or _WORKER_POLICY is None:
        raise RuntimeError("H11 worker was not initialized")
    agent = FrozenCombatSampledMacro(
        _WORKER_POLICY, seed=seed, temperature=temperature,
    )
    result = play_run(
        _WORKER_BACKEND, agent, seed=seed,
        policy=InformationPolicy(FAIR_POLICY_ID),
        ascension=10, max_decisions=max_decisions,
        environment=NATIVE_OVERGROWTH,
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
    )
    if result.censored:
        # A capped/unsupported outcome is never relabelled as a defeat.
        samples: tuple[PreparedH7, ...] = ()
    elif result.outcome in ("victory", "defeat"):
        # Act-clear dominates all defeats; progress is only a weak auxiliary
        # signal while the initial macro actor has very few successful runs.
        score = (2.0 if result.outcome == "victory"
                 else 0.2 * min(1.0, max(0.0, result.frontier_progress / 16.0)))
        samples = (build_gae(agent.entries, reward=score)
                   if agent.entries else ())
    else:
        raise ValueError(f"Unexpected native run outcome: {result.outcome}")
    return index, RunRollout(
        seed, result.outcome, result.censored,
        result.act1_cleared is True,
        result.frontier_progress, result.decisions,
        len(agent.entries), samples,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix="." + path.name + "-", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialize-from-model", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--episodes-per-round", type=int, default=64)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--seed-prefix", default="h11-shared-macro-a10-train-v1")
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--lr", type=float, default=0.00003)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch", type=int, default=64)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.episodes_per_round, args.workers,
            args.max_decisions, args.ppo_epochs, args.ppo_minibatch,
            args.ppo_sample_limit, args.checkpoint_every) < 1
            or args.temperature <= 0 or args.lr <= 0 or not args.seed_prefix):
        p.error("Invalid H11 configuration")
    if args.emulator_root.resolve() == Path.cwd().resolve():
        p.error("Emulator root must be the sts2-emulator checkout")
    if not args.emulator_dll.is_file():
        p.error("Missing pinned emulator DLL")
    from subprocess import check_output
    revision = check_output(
        ["git", "-C", str(args.emulator_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != args.emulator_revision:
        p.error("Emulator revision mismatch")
    config = {
        "schema": SCHEMA, "base_sha256": file_sha256(args.initialize_from_model),
        "emulator_revision": revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "arguments": {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()
                      if k not in ("execute", "resume", "output_root")},
    }
    fingerprint = canonical_sha256(config)
    print(json.dumps({"fingerprint": fingerprint,
                      "macro_only": True, "execute": args.execute,
                      "rounds": args.rounds}, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is not available")
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    model = SharedObjectTransformer.load_checkpoint(args.initialize_from_model)
    freeze_for_macro(model)
    learner = H7CombatPPO(
        model, device=args.device, lr=args.lr,
        temperature=args.temperature, epochs=args.ppo_epochs,
        minibatch=args.ppo_minibatch, sample_limit=args.ppo_sample_limit,
    )
    out = args.output_root
    models = out / "models"
    checkpoint = out / "optimizer-checkpoint.pt"
    history: list[dict[str, Any]] = []
    if args.resume:
        if not checkpoint.exists():
            p.error("Missing H11 optimizer checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["schema"] != SCHEMA or saved["fingerprint"] != fingerprint:
            p.error("H11 resume fingerprint mismatch")
        if len(saved["history"]) > args.rounds:
            p.error("H11 checkpoint exceeds requested rounds")
        learner.model.load_state_dict(saved["model"], strict=True)
        learner.optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif checkpoint.exists() or (models.exists() and any(models.iterdir())):
        p.error("Existing output must use --resume or another output path")
    models.mkdir(parents=True, exist_ok=True)
    actor_file = out / "actor-current.pt"

    for index in range(len(history), args.rounds):
        begun = time.monotonic()
        learner.model.save_checkpoint(actor_file)
        inputs = [(i, f"{args.seed_prefix}-r{index+1}-e{i}",
                   args.temperature, args.max_decisions)
                  for i in range(args.episodes_per_round)]
        results: list[RunRollout | None] = [None] * len(inputs)
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn"),
            initializer=_worker_init,
            initargs=(str(args.emulator_root), revision,
                      str(actor_file.resolve())),
        ) as pool:
            futures = [pool.submit(_worker_run, request) for request in inputs]
            for future in as_completed(futures):
                at, rollout = future.result()
                results[at] = rollout
        if any(item is None for item in results):
            raise RuntimeError("H11 native cohort has missing worker results")
        cohort = [item for item in results if item is not None]
        completed = [item for item in cohort if not item.censored]
        samples = [sample for item in completed for sample in item.samples]
        diagnostics: dict[str, Any] = {}
        if len(samples) >= 2:
            diagnostics = learner.update_prepared(
                samples, seed=20261029 + index, allow_native_cpu_actor=True,
            )
        row: dict[str, Any] = {
            "round": index + 1, "source_runs": len(cohort),
            "complete": len(completed),
            "censored": sum(item.censored for item in cohort),
            "wins": sum(item.act_clear for item in cohort),
            "macro_samples": len(samples),
            "mean_progress": sum(item.frontier_progress for item in cohort) / len(cohort),
            "elapsed_seconds": time.monotonic() - begun,
            "optimizer": diagnostics,
        }
        history.append(row)
        atomic_torch(checkpoint, {
            "schema": SCHEMA, "fingerprint": fingerprint,
            "model": learner.model.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if args.device.startswith("cuda") else None,
        })
        if (index + 1) % args.checkpoint_every == 0 or index + 1 == args.rounds:
            learner.model.save_checkpoint(models / f"player-r{index+1:04d}.pt")
        _atomic_json(out / "progress.json", {
            "schema": SCHEMA, "fingerprint": fingerprint,
            "config": config, "history": history,
        })
        print("[h11-macro] " + json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
