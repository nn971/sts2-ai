"""Authentic Act-1 snapshot curriculum and public-only combat rollouts.

Opt-in support for training the v15 combat residual without replaying entire
macro-policy continuations. Source-run provenance is preserved: each recipe is
restored exactly, verified, and forked before public-only tactical sampling.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator import EmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run
from sts2_ai.evaluation.combat_snapshots import (
    POLICY,
    CombatSnapshotRecipe,
    _advance,
    _verify,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.combat_outcomes import CombatOutcomeRecorder
from sts2_ai.training.parallel_rollouts import episode_actor_seed
from sts2_ai.training.selfplay import (
    Episode,
    PublicDecision,
    _public_metrics,
    sample_public_action,
)

SourceKey = tuple[str, int, str]


def source_key(recipe: CombatSnapshotRecipe) -> SourceKey:
    return recipe.seed, recipe.ascension, recipe.source_policy


def split_snapshot_sources(
    recipes: Sequence[CombatSnapshotRecipe], *,
    holdout_fraction: float = 0.2, salt: str = "sts2-v15m-snapshot-v1",
) -> tuple[tuple[CombatSnapshotRecipe, ...], tuple[CombatSnapshotRecipe, ...]]:
    """Stable group split; all entries from one source run share a partition.

    The holdout is a within-corpus diagnostic split. A previously inspected
    corpus remains development data regardless of this partition.
    """
    if not 0.0 < holdout_fraction < 1.0 or not math.isfinite(holdout_fraction):
        raise ValueError("Holdout fraction must be strictly between zero and one")
    if not salt:
        raise ValueError("Split salt must be nonempty")
    grouped: dict[SourceKey, list[CombatSnapshotRecipe]] = defaultdict(list)
    seen: set[tuple[SourceKey, str]] = set()
    for recipe in recipes:
        recipe.validate()
        identity = (source_key(recipe), recipe.exact_hash)
        if identity in seen:
            raise ValueError("Duplicated exact snapshot entry")
        seen.add(identity)
        grouped[source_key(recipe)].append(recipe)
    if len(grouped) < 2:
        raise ValueError("Source-run split requires at least two independent runs")

    train: list[CombatSnapshotRecipe] = []
    holdout: list[CombatSnapshotRecipe] = []
    for key in sorted(grouped):
        raw = json.dumps((salt, key), separators=(",", ":")).encode()
        u = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / (1 << 64)
        target = holdout if u < holdout_fraction else train
        target.extend(sorted(grouped[key], key=lambda r: (len(r.action_ids), r.exact_hash)))
    if not train or not holdout:
        raise ValueError(
            "Split contains an empty partition; collect more independent source runs"
        )
    return tuple(train), tuple(holdout)


def sample_snapshot_curriculum(
    recipes: Sequence[CombatSnapshotRecipe], *, count: int, seed: int,
    mode: str = "balanced",
) -> tuple[CombatSnapshotRecipe, ...]:
    """Deterministic sample with replacement when an encounter stratum is small.

    'balanced' cycles over observed (progress, tier) pairs, preserving native
    progress-to-build matching. 'natural' cycles through the original corpus.
    Neither constructs synthetic combinations or changes hidden RNG.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("Snapshot sample count must be positive")
    if mode not in ("balanced", "natural"):
        raise ValueError("Snapshot sampling must be balanced or natural")
    if not recipes:
        raise ValueError("Snapshot sampling requires a nonempty corpus")
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[CombatSnapshotRecipe]] = defaultdict(list)
    seen: set[tuple[SourceKey, str]] = set()
    for recipe in recipes:
        recipe.validate()
        identity = (source_key(recipe), recipe.exact_hash)
        if identity in seen:
            raise ValueError("Duplicated exact snapshot entry")
        seen.add(identity)
        strata[(recipe.progress, recipe.tier)].append(recipe)

    if mode == "natural":
        buckets = {("all", "all"): list(recipes)}
    else:
        buckets = dict(strata)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for pool in buckets.values():
        pool.sort(key=lambda r: (source_key(r), len(r.action_ids), r.exact_hash))
        rng.shuffle(pool)
    cursors = dict.fromkeys(keys, 0)
    selected: list[CombatSnapshotRecipe] = []
    for i in range(count):
        key = keys[i % len(keys)]
        pool = buckets[key]
        at = cursors[key]
        if at == len(pool):
            rng.shuffle(pool)
            at = 0
        selected.append(pool[at])
        cursors[key] = at + 1
    return tuple(selected)


def _snapshot_episode(
    backend: EmulatorBackend, state: str, model: PhaseSplitNeuralModel, *,
    episode_seed: str, actor_seed: int, temperature: float,
    max_decisions: int,
) -> Episode:
    """Consume one fork. Never pass its exact handle to the tactical policy."""
    recorder = CombatOutcomeRecorder()
    decisions: list[PublicDecision] = []
    rng = random.Random(episode_actor_seed(actor_seed, episode_seed))
    handle = state
    try:
        while True:
            frame = backend.observe(handle, POLICY)
            public = json.loads(frame.payload_json)
            if not isinstance(public, dict):
                raise ValueError("Invalid public combat observation")
            recorder.observe(public, len(decisions))
            active = isinstance(public.get("combat"), dict)
            terminal = backend.is_terminal(handle)
            if terminal or not active:
                outcomes = recorder.outcomes
                if len(outcomes) != 1:
                    raise ValueError("Resolved snapshot must yield exactly one combat")
                outcome = outcomes[0].result
                act, floor, hp_ratio = _public_metrics(frame)
                return Episode(
                    episode_seed, tuple(decisions), outcome, act, floor, hp_ratio,
                    True, NATIVE_OVERGROWTH, combat_outcomes=outcomes,
                )
            if len(decisions) >= max_decisions:
                act, floor, hp_ratio = _public_metrics(frame)
                return Episode(
                    episode_seed, tuple(decisions), "truncated", act, floor,
                    hp_ratio, False, NATIVE_OVERGROWTH,
                    combat_outcomes=recorder.outcomes,
                )
            legal = tuple(backend.legal_actions(handle))
            if not legal:
                raise ValueError("Live combat snapshot has no legal actions")
            selected = sample_public_action(
                model, frame, legal, rng=rng, temperature=temperature,
            )
            decisions.append(PublicDecision(frame, legal, selected, "combat"))
            recorder.selected_action(legal[selected])
            child = backend.step(handle, legal[selected]).child
            backend.release_many([handle])
            handle = child
    finally:
        backend.release_many([handle])


IndexedRecipe = tuple[int, CombatSnapshotRecipe]
SourceGroup = tuple[SourceKey, tuple[IndexedRecipe, ...]]


@dataclass(frozen=True, slots=True)
class SnapshotWorkerBatch:
    """One frozen policy per process task; source runs remain atomic."""

    groups: tuple[SourceGroup, ...]
    model_payload: dict[str, Any]
    round_index: int
    actor_seed: int
    temperature: float
    max_decisions: int


def plan_snapshot_batches(
    grouped: dict[SourceKey, list[IndexedRecipe]], *,
    workers: int, max_decisions: int,
) -> tuple[tuple[SourceGroup, ...], ...]:
    """Greedy deterministic grouping of whole source runs into worker batches.

    One serialized v15 model is sent per batch (at most one per worker).
    Prefix length and anticipated combat actions approximate source work.
    This scheduling heuristic has no effect on the chosen episodes/RNG.
    """
    if type(workers) is not int or workers <= 0 or max_decisions <= 0:
        raise ValueError("Invalid snapshot batch worker count or combat cap")
    if not grouped:
        return ()
    ranks = sorted(
        grouped.items(),
        key=lambda kv: (
            -(max(len(recipe.action_ids) for _, recipe in kv[1])
              + min(32, max_decisions) * len(kv[1])),
            kv[0],
        ),
    )
    bucket_count = min(workers, len(ranks))
    buckets: list[list[SourceGroup]] = [[] for _ in range(bucket_count)]
    loads = [0] * bucket_count
    for key, rows in ranks:
        ordered = tuple(sorted(rows, key=lambda item: (
            len(item[1].action_ids), item[0],
        )))
        estimated = (
            max(len(recipe.action_ids) for _, recipe in ordered)
            + min(32, max_decisions) * len(ordered)
        )
        destination = min(range(bucket_count), key=lambda i: (loads[i], i))
        buckets[destination].append((key, ordered))
        loads[destination] += estimated
    return tuple(tuple(bucket) for bucket in buckets)


def _collect_source_group(
    backend: EmulatorBackend, model: PhaseSplitNeuralModel,
    group: SourceGroup, *,
    round_index: int, actor_seed: int, temperature: float,
    max_decisions: int,
) -> tuple[tuple[int, Episode], ...]:
    (seed, ascension, _source), items = group
    handle = reset_training_run(backend, seed, NATIVE_OVERGROWTH, ascension)
    prefix: tuple[str, ...] = ()
    results: list[tuple[int, Episode]] = []
    try:
        for index, recipe in items:
            if (recipe.emulator_revision != backend.emulator_revision
                    or source_key(recipe) != group[0]):
                raise ValueError("Snapshot source/revision mismatch")
            if recipe.action_ids[:len(prefix)] != prefix:
                raise ValueError("Inconsistent source-run snapshot prefixes")
            handle = _advance(backend, handle, recipe.action_ids[len(prefix):])
            prefix = recipe.action_ids
            _verify(backend, handle, recipe)
            episode_seed = f"snapshot-{round_index}-{index}-{seed}-{recipe.exact_hash}"
            episode = _snapshot_episode(
                backend, backend.fork(handle), model,
                episode_seed=episode_seed, actor_seed=actor_seed,
                temperature=temperature, max_decisions=max_decisions,
            )
            results.append((index, episode))
    finally:
        backend.release_many([handle])
    return tuple(results)


def _worker_collect_snapshot_batch(
    batch: SnapshotWorkerBatch,
) -> tuple[tuple[int, Episode], ...]:
    """Process-isolated backend from the existing pinned rollout pool."""
    from sts2_ai.training import parallel_rollouts

    backend = parallel_rollouts._WORKER_BACKEND
    if backend is None:
        raise RuntimeError("Snapshot worker has no independent emulator")
    model = PhaseSplitNeuralModel.from_dict(batch.model_payload)
    results: list[tuple[int, Episode]] = []
    for group in batch.groups:
        results.extend(_collect_source_group(
            backend, model, group,
            round_index=batch.round_index,
            actor_seed=batch.actor_seed,
            temperature=batch.temperature,
            max_decisions=batch.max_decisions,
        ))
    return tuple(results)


def collect_snapshot_cohort(
    backend: EmulatorBackend, model: PhaseSplitNeuralModel,
    selected: Sequence[CombatSnapshotRecipe], *, round_index: int,
    actor_seed: int, temperature: float, max_decisions: int = 256,
    pool: ProcessPoolExecutor | None = None, workers: int = 1,
) -> tuple[Episode, ...]:
    """Replay verified sources, fork combat and return ordered on-policy episodes.

    With a pool, each task owns complete source-run groups; worker completion
    order cannot alter episode order or sampled actions. Exact hidden handles
    never leave the owning emulator process. Identical recipes deliberately
    share initial RNG even if repeated under different action samples.
    """
    if not selected or max_decisions <= 0:
        raise ValueError("Require selected snapshots and a positive decision cap")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid snapshot rollout temperature")
    if type(round_index) is not int or round_index < 0:
        raise ValueError("Invalid snapshot round index")
    if type(workers) is not int or workers <= 0:
        raise ValueError("Invalid snapshot worker count")
    if (pool is None and workers != 1) or (pool is not None and workers <= 1):
        raise ValueError("Multiprocess snapshots require a pool and workers > 1")
    grouped: dict[SourceKey, list[IndexedRecipe]] = defaultdict(list)
    for index, recipe in enumerate(selected):
        recipe.validate()
        if recipe.emulator_revision != backend.emulator_revision:
            raise ValueError("Snapshot emulator revision mismatch")
        grouped[source_key(recipe)].append((index, recipe))
    batches = plan_snapshot_batches(
        grouped, workers=workers, max_decisions=max_decisions,
    )
    results: list[Episode | None] = [None] * len(selected)
    batches_returned: Iterable[tuple[tuple[int, Episode], ...]]
    if pool is None:
        batches_returned = (
            tuple(
                result for group in batch
                for result in _collect_source_group(
                    backend, model, group,
                    round_index=round_index, actor_seed=actor_seed,
                    temperature=temperature, max_decisions=max_decisions,
                )
            )
            for batch in batches
        )
    else:
        payload = model.to_dict()
        requests = (
            SnapshotWorkerBatch(
                batch, payload, round_index, actor_seed, temperature,
                max_decisions,
            )
            for batch in batches
        )
        batches_returned = pool.map(_worker_collect_snapshot_batch, requests)
    for group_results in batches_returned:
        for index, episode in group_results:
            if (type(index) is not int or not 0 <= index < len(results)
                    or results[index] is not None):
                raise ValueError("Snapshot worker returned duplicate/invalid result")
            results[index] = episode
    if any(episode is None for episode in results):
        raise ValueError("Incomplete snapshot cohort")
    return tuple(episode for episode in results if episode is not None)
