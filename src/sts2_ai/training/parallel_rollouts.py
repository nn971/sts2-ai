"""Process-isolated public-only rollout workers for real JSONL emulator runs.

The learner stays on one process and sends a frozen portable policy to each
worker. Every worker owns a separate .NET emulator process; no hidden state or
actual game RNG is transmitted between workers and the trainer.
"""
from __future__ import annotations

import atexit
import hashlib
import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.models.neural import NeuralPolicyValueModel

if TYPE_CHECKING:
    from sts2_ai.training.selfplay import Episode

_WORKER_BACKEND: JsonlEmulatorBackend | None = None


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    model_payload: dict[str, Any]
    seed: str
    actor_seed: int
    max_decisions: int
    policy_id: str
    environment: str


def episode_actor_seed(base_seed: int, run_seed: str) -> int:
    """Stable per-episode actor RNG, independent of worker scheduling."""
    data = f"sts2-actor-v1:{base_seed}:{run_seed}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], byteorder="big")


def _worker_init(
    repo_root: str, emulator_root: str, revision: str
) -> None:
    global _WORKER_BACKEND
    if _WORKER_BACKEND is not None:
        raise RuntimeError("Rollout worker initialized twice")
    backend = JsonlEmulatorBackend(
        repo_root=Path(repo_root),
        emulator_root=Path(emulator_root),
        expected_emulator_revision=revision,
        build=False,
    )
    _WORKER_BACKEND = backend
    atexit.register(backend.close)


def _worker_collect(request: EpisodeRequest) -> Episode:
    from sts2_ai.training.selfplay import collect_public_episode

    backend = _WORKER_BACKEND
    if backend is None:
        raise RuntimeError("Rollout worker has no private emulator instance")
    return collect_public_episode(
        backend,
        NeuralPolicyValueModel.from_dict(request.model_payload),
        seed=request.seed,
        actor_rng=random.Random(request.actor_seed),
        max_decisions=request.max_decisions,
        policy_id=request.policy_id,
        environment=request.environment,
    )


def open_worker_pool(backend: JsonlEmulatorBackend, workers: int) -> ProcessPoolExecutor:
    if type(workers) is not int or workers <= 1:
        raise ValueError("Multiprocess worker count must be at least 2")
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_worker_init,
        initargs=(
            str(backend._repo_root),
            str(backend._emulator_root),
            backend.emulator_revision,
        ),
    )


def collect_parallel(
    pool: ProcessPoolExecutor,
    model: NeuralPolicyValueModel,
    run_seeds: Sequence[str],
    *,
    base_seed: int,
    max_decisions: int,
    policy_id: str,
    environment: str,
) -> tuple[Episode, ...]:
    if not run_seeds:
        return ()
    if len(set(run_seeds)) != len(run_seeds):
        raise ValueError("Each rollout seed must be unique within a cohort")
    payload = model.to_dict()
    requests = [
        EpisodeRequest(
            payload, seed, episode_actor_seed(base_seed, seed),
            max_decisions, policy_id, environment,
        )
        for seed in run_seeds
    ]
    # Executor.map retains caller order, regardless of worker completion order.
    return tuple(pool.map(_worker_collect, requests))
