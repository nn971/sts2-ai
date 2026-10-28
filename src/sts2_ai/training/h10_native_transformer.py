"""On-policy H9 Transformer rollouts from exact, source-run-native A10 combat entries.

The native player only observes FAIR_POLICY_ID public states/legal menus. The
replay verifier alone accesses exact hashes. Training uses source-run grouped
forks and the already tested H7 PPO ratio/critic checks.
"""
from __future__ import annotations

import atexit
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from sts2_ai.emulator import EmulatorBackend, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run
from sts2_ai.evaluation.combat_snapshots import (
    POLICY, CombatSnapshotRecipe, _advance, _verify, load_recipes,
)
from sts2_ai.models.h7_public_policy import H7PlayerSpec, TorchH7PublicPolicy
from sts2_ai.models.shared_game_objects import encode_public_decision
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h7_combat_ppo import PreparedH7
from sts2_ai.training.snapshot_curriculum import (
    sample_snapshot_curriculum, source_key,
)

NATIVE_REWARD = "h10-native-win-and-exit-hp-v1"
_WORKER_BACKEND: JsonlEmulatorBackend | None = None
_WORKER_POLICY: TorchH7PublicPolicy | None = None
_WORKER_SPEC: H7PlayerSpec | None = None


@dataclass(frozen=True)
class NativeRollout:
    source_seed: str
    entry_hash: str
    tier: str
    outcome: str
    reward: float
    exit_hp: int
    decisions: int
    samples: tuple[PreparedH7, ...]


def native_reward(outcome: str, exit_hp: int, max_hp: int) -> float:
    """No fabricated healing/summon progress: defeats score zero.

    This first native objective is deliberately distinct from synthetic H9
    total-nonminion defeat-progress reward. Both lie in [0, 2].
    """
    if type(exit_hp) is not int or type(max_hp) is not int or not (
        max_hp > 0 and 0 <= exit_hp <= max_hp
    ):
        raise ValueError("Invalid public native exit HP")
    if outcome == "victory":
        return 1.0 + exit_hp / max_hp
    if outcome == "defeat":
        return 0.0
    raise ValueError("Native PPO requires a resolved terminal combat")


def choose_sampled(logits: Sequence[float], *, temperature: float,
                   rng: random.Random) -> tuple[int, float]:
    if (not logits or not math.isfinite(temperature) or temperature <= 0
            or any(not math.isfinite(float(v)) for v in logits)):
        raise ValueError("Invalid native actor distribution")
    logp = torch.log_softmax(
        torch.tensor(logits, dtype=torch.float32) / temperature, dim=0,
    )
    chosen = rng.choices(range(len(logits)), weights=logp.exp().tolist(), k=1)[0]
    return chosen, float(logp[chosen])


def build_gae(
    steps: Sequence[tuple[object, int, float, float]], *,
    reward: float, gamma: float = 1.0, gae_lambda: float = 0.98,
) -> tuple[PreparedH7, ...]:
    if not steps or not 0 <= reward <= 2 or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Invalid native PPO trajectory")
    gae = 0.0
    prepared: list[PreparedH7] = []
    for i in reversed(range(len(steps))):
        encoded, chosen, logp, value = steps[i]
        next_value = steps[i + 1][3] if i + 1 < len(steps) else 0.0
        r = reward / 2 if i + 1 == len(steps) else 0.0
        delta = r + gamma * next_value - value
        gae = delta + gamma * gae_lambda * gae
        prepared.append(PreparedH7(
            encoded, chosen, logp, value, gae, gae + value,
        ))
    return tuple(reversed(prepared))


def play_native_fork(
    backend: EmulatorBackend, handle: str, policy: TorchH7PublicPolicy,
    spec: H7PlayerSpec, recipe: CombatSnapshotRecipe, *,
    round_index: int, occurrence: int, max_decisions: int,
) -> NativeRollout:
    """Own and always release the fork; public-only actor and fresh actions."""
    rng = random.Random(spec.episode_seed(
        fixture_seed=f"{recipe.seed}:{round_index}:{occurrence}",
        fixture_id=recipe.exact_hash,
    ))
    entries: list[tuple[object, int, float, float]] = []
    current = handle
    try:
        while True:
            observation = backend.observe(current, POLICY)
            state = json.loads(observation.payload_json)
            if not isinstance(state, dict):
                raise ValueError("Malformed fair public native combat observation")
            combat = state.get("combat")
            if backend.is_terminal(current) or not isinstance(combat, dict):
                hp = state.get("hp")
                if type(hp) is not int:
                    raise ValueError("Native terminal missing public player HP")
                if hp < 0:
                    raise ValueError("Invalid terminal HP")
                outcome = "defeat" if hp == 0 or (
                    state.get("terminal_outcome") == "defeat"
                ) else "victory"
                score = native_reward(outcome, hp, recipe.max_hp)
                if not entries:
                    raise ValueError("Native snapshot completed with zero player decisions")
                return NativeRollout(
                    recipe.seed, recipe.exact_hash, recipe.tier, outcome, score,
                    hp, len(entries), build_gae(entries, reward=score),
                )
            if len(entries) >= max_decisions:
                return NativeRollout(
                    recipe.seed, recipe.exact_hash, recipe.tier,
                    "censored", 0.0, int(state.get("hp", 0)), len(entries), (),
                )
            legal = tuple(backend.legal_actions(current))
            if not legal:
                raise ValueError("Active native combat has no legal actions")
            estimate = policy.evaluate(observation, legal)
            encoded = encode_public_decision(state, legal)
            if encoded.phase != "combat":
                raise ValueError("Native PPO received noncombat legal menu")
            chosen, logp = choose_sampled(
                estimate.action_logits, temperature=spec.temperature, rng=rng,
            )
            entries.append((encoded, chosen, logp, float(estimate.value)))
            child = backend.step(current, legal[chosen]).child
            backend.release_many([current])
            current = child
    finally:
        backend.release_many([current])


def collect_source_group(
    backend: EmulatorBackend, policy: TorchH7PublicPolicy, spec: H7PlayerSpec,
    indexed: Sequence[tuple[int, CombatSnapshotRecipe]], *,
    round_index: int, max_decisions: int,
) -> tuple[tuple[int, NativeRollout], ...]:
    """Replay each source prefix once, verifying every exact combat entry."""
    if not indexed:
        return ()
    indexed = sorted(indexed, key=lambda pair: (
        len(pair[1].action_ids), pair[1].exact_hash, pair[0],
    ))
    key = source_key(indexed[0][1])
    seed, ascension, _ = key
    current = reset_training_run(backend, seed, NATIVE_OVERGROWTH, ascension)
    prefix: tuple[str, ...] = ()
    results: list[tuple[int, NativeRollout]] = []
    try:
        for occurrence, (ordinal, recipe) in enumerate(indexed):
            if (recipe.emulator_revision != backend.emulator_revision
                    or source_key(recipe) != key
                    or recipe.action_ids[:len(prefix)] != prefix):
                raise ValueError("Native snapshot provenance/prefix mismatch")
            current = _advance(backend, current, recipe.action_ids[len(prefix):])
            prefix = recipe.action_ids
            _verify(backend, current, recipe)
            results.append((ordinal, play_native_fork(
                backend, backend.fork(current), policy, spec, recipe,
                round_index=round_index, occurrence=ordinal,
                max_decisions=max_decisions,
            )))
    finally:
        backend.release_many([current])
    return tuple(results)


def verify_corpora(
    train: Sequence[CombatSnapshotRecipe], heldout: Sequence[CombatSnapshotRecipe],
    *, emulator_revision: str,
) -> None:
    if not train or not heldout:
        raise ValueError("Native train and held-out corpora must both contain entries")
    for records in (train, heldout):
        identities = set()
        for recipe in records:
            recipe.validate()
            if (recipe.emulator_revision != emulator_revision or recipe.ascension != 10
                    or (recipe.seed, recipe.exact_hash) in identities):
                raise ValueError("Native corpus revision, ascension or uniqueness mismatch")
            identities.add((recipe.seed, recipe.exact_hash))
    train_keys = {source_key(recipe) for recipe in train}
    heldout_keys = {source_key(recipe) for recipe in heldout}
    if train_keys & heldout_keys or {r.exact_hash for r in train} & {
        r.exact_hash for r in heldout
    }:
        raise ValueError("Train and holdout native source runs must be disjoint")


def select_native(
    records: Sequence[CombatSnapshotRecipe], *, count: int,
    study_seed: int, round_index: int,
) -> tuple[CombatSnapshotRecipe, ...]:
    return sample_snapshot_curriculum(
        records, count=count,
        seed=int(canonical_sha256(["h10-native-train", study_seed, round_index])[:16], 16),
        mode="balanced",
    )


def _worker_init(repo_root: str, emulator_root: str, revision: str,
                 model_path: str, model_sha: str, actor_seed: int,
                 temperature: float) -> None:
    global _WORKER_BACKEND, _WORKER_POLICY, _WORKER_SPEC
    _WORKER_BACKEND = JsonlEmulatorBackend(
        repo_root=Path(repo_root), emulator_root=Path(emulator_root),
        expected_emulator_revision=revision, build=False,
    )
    atexit.register(_WORKER_BACKEND.close)
    _WORKER_SPEC = H7PlayerSpec(
        model_path, model_sha, actor_seed, temperature=temperature,
    )
    _WORKER_POLICY = _WORKER_SPEC.load_pinned()


def _worker_group(task: tuple[int, list[tuple[int, CombatSnapshotRecipe]], int]
                  ) -> tuple[tuple[int, NativeRollout], ...]:
    if _WORKER_BACKEND is None or _WORKER_POLICY is None or _WORKER_SPEC is None:
        raise RuntimeError("Native PPO worker not initialized")
    round_index, rows, max_decisions = task
    return collect_source_group(
        _WORKER_BACKEND, _WORKER_POLICY, _WORKER_SPEC, rows,
        round_index=round_index, max_decisions=max_decisions,
    )


def native_cohort(
    *, selected: Sequence[CombatSnapshotRecipe], round_index: int,
    spec: H7PlayerSpec, emulator_root: Path, emulator_revision: str,
    workers: int, max_decisions: int,
) -> tuple[NativeRollout, ...]:
    """Fixed ordering regardless of worker completion, one pinned actor per process."""
    if workers < 1 or max_decisions < 1 or not selected:
        raise ValueError("Invalid native PPO cohort arguments")
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    grouped: dict[tuple[str, int, str], list[tuple[int, CombatSnapshotRecipe]]] = defaultdict(list)
    for i, recipe in enumerate(selected):
        grouped[source_key(recipe)].append((i, recipe))
    ordered_groups = [grouped[key] for key in sorted(grouped)]
    result: list[NativeRollout | None] = [None] * len(selected)
    with JsonlEmulatorBackend(
        emulator_root=emulator_root,
        expected_emulator_revision=emulator_revision, build=False,
    ) as backend:
        initargs = (
            str(backend._repo_root), str(backend._emulator_root),
            emulator_revision, spec.model_path, spec.model_sha256,
            spec.actor_seed, spec.temperature,
        )
        if workers == 1:
            policy = spec.load_pinned()
            batches = [collect_source_group(
                backend, policy, spec, rows, round_index=round_index,
                max_decisions=max_decisions,
            ) for rows in ordered_groups]
        else:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(ordered_groups)),
                mp_context=get_context("spawn"),
                initializer=_worker_init, initargs=initargs,
            ) as pool:
                batches = list(pool.map(
                    _worker_group,
                    [(round_index, rows, max_decisions) for rows in ordered_groups],
                ))
        for batch in batches:
            for index, episode in batch:
                if result[index] is not None:
                    raise ValueError("Native PPO source returned duplicate entry")
                result[index] = episode
    if any(row is None for row in result):
        raise ValueError("Incomplete native PPO source cohort")
    return tuple(row for row in result if row is not None)


def summarize_native(episodes: Sequence[NativeRollout]) -> dict[str, object]:
    if not episodes:
        raise ValueError("No native combat outcomes")
    counts = Counter(ep.outcome for ep in episodes)
    tiers = Counter(ep.tier for ep in episodes)
    return {
        "combats": len(episodes), "outcomes": dict(counts),
        "by_tier": dict(tiers),
        "mean_reward": sum(ep.reward for ep in episodes) / len(episodes),
        "mean_exit_hp": sum(ep.exit_hp for ep in episodes) / len(episodes),
        "sampled_decisions": sum(ep.decisions for ep in episodes),
        "censored": counts["censored"],
    }
