"""v15c combat-only, on-policy PPO with a frozen v8/v13 macro policy.

This is intentionally separate from the existing phase_split_ppo_batch path:
the v15 pooled control is an independent residual model, not a reinterpretation
of v8 flattened feature tensors. Relational v15d PPO uses padded, masked,
batched object/action attention; the pooled v15c control remains unchanged.

Only resolved *public combat segments* supply terminal HP-preservation GAE.
An unfinished combat in a decision-capped run has no target and is excluded.
No hidden RNG, deck order, scripted moves, or fake terminal rewards are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import EmulatorBackend, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import (
    PROTOTYPE_THREE_ACT_GOAL,
    require_episode_goal,
)
from sts2_ai.emulator.run_environment import LEGACY, require_environment
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.models.object_embeddings import CompiledObjectFeatures, ObjectVocabulary
from sts2_ai.models.object_instances import (
    ActionBinding,
    ObjectRef,
    ObjectSet,
    extract_actionable_combat_objects,
    legal_action_object_refs,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.pooled_objects import PooledObjectTorch
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.models.relational_objects import (
    RelationalObjectTorch,
    RelationalTopology,
    _action_tokens,
    prepare_relational_topology,
)
from sts2_ai.training.parallel_rollouts import (
    collect_parallel,
    episode_actor_seed,
    open_worker_pool,
)
from sts2_ai.training.phase_split_ppo import gae_terminal
from sts2_ai.training.phase_split_selfplay import _hp_preservation_target
from sts2_ai.training.selfplay import (
    Episode,
    collect_public_episode,
    temperature_for_round,
)

POOLED_PPO_VERSION = "sts2-v15c-frozen-strategy-combat-ppo-v1"
POOLED_PPO_CHECKPOINT = "sts2-v15c-pooled-combat-checkpoint-v1"
RELATIONAL_PPO_CHECKPOINT = "sts2-v15d-relational-combat-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class PooledPpoSample:
    objects: ObjectSet
    action_refs: tuple[tuple[ObjectRef, ...], ...]
    baseline: PolicyValueEstimate
    chosen_index: int
    old_logprob: float
    advantage: float
    target: float
    action_bindings: tuple[tuple[str, tuple[ActionBinding, ...]], ...] | None = None
    relational_topology: RelationalTopology | None = None
    compiled_features: CompiledObjectFeatures | None = None


@dataclass(frozen=True, slots=True)
class PooledPpoMetrics:
    samples: int
    steps: int
    loss: float | None
    mean_policy_loss: float | None
    mean_value_loss: float | None
    mean_entropy: float | None


@dataclass(frozen=True, slots=True)
class PooledTrainingRound:
    round_index: int
    played: int
    completed: int
    censored: int
    wins: int
    resolved_combats: int
    combat_decisions: int
    optimizer_steps: int
    mean_loss: float | None
    mean_combat_target: float | None
    rollout_seconds: float
    optimizer_seconds: float


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def collect_public_vocabulary(
    backend: EmulatorBackend, baseline: PhaseSplitNeuralModel, *,
    runs: int = 32, seed: int = 19, seed_prefix: str = "v15-vocab",
    max_decisions: int = 4096, environment: str = LEGACY,
    episode_goal_version: str = PROTOTYPE_THREE_ACT_GOAL,
    temperature: float = 0.7,
    workers: int = 1,
) -> ObjectVocabulary:
    """Discover a *frozen* v15 vocabulary from disjoint public policy rollouts.

    This consumes training-side sampling seeds, never held-out evaluation seeds.
    OOV remains defined for objects absent from this cohort. The vocabulary
    must be saved and reused identically across runs and their controls.
    """
    if min(runs, workers, max_decisions) <= 0:
        raise ValueError("Vocabulary runs/workers/cap must be positive")
    if not (0 < temperature and math.isfinite(temperature)):
        raise ValueError("Invalid vocabulary collection temperature")
    require_episode_goal(episode_goal_version, environment)
    require_environment(backend, environment)
    if workers > 1 and not isinstance(backend, JsonlEmulatorBackend):
        raise ValueError("Parallel collection requires the JSONL emulator")
    seeds = [f"{seed_prefix}-{i}" for i in range(runs)]
    with ExitStack() as scope:
        pool = (
            scope.enter_context(open_worker_pool(backend, workers))
            if isinstance(backend, JsonlEmulatorBackend) and workers > 1
            else None
        )
        if pool is not None:
            cohort = collect_parallel(
                pool, baseline, seeds, base_seed=seed,
                max_decisions=max_decisions, policy_id="prototype-fair-v0",
                environment=environment, temperature=temperature,
                episode_goal_version=episode_goal_version,
            )
        else:
            cohort = tuple(collect_public_episode(
                backend, baseline, seed=run_seed,
                actor_rng=random.Random(episode_actor_seed(seed, run_seed)),
                max_decisions=max_decisions, environment=environment,
                temperature=temperature, episode_goal_version=episode_goal_version,
            ) for run_seed in seeds)
    observations: list[ObjectSet] = []
    for episode in cohort:
        for decision in episode.decisions:
            if decision.phase == "combat":
                state = json.loads(decision.observation.payload_json)
                if not isinstance(state, dict):
                    raise ValueError("Invalid public observation")
                observations.append(extract_actionable_combat_objects(
                    state,
                    [(a.kind, a.payload_json) for a in decision.legal_actions],
                ))
    if not observations:
        raise ValueError("No public combat observations available for vocabulary")
    # The pinned JSONL backend already validates the fair public capability
    # manifest at handshake. Use only its declared static IDs; never parse
    # private engine state or scrape internal sources for vocabulary labels.
    manifest = (
        backend.capability_manifest
        if isinstance(backend, JsonlEmulatorBackend) else None
    )
    return ObjectVocabulary.from_object_sets(observations, manifest=manifest)


def _prepared_decision(
    decision: Any, base: NeuralPolicyValueModel,
) -> tuple[ObjectSet, tuple[tuple[ObjectRef, ...], ...], PolicyValueEstimate]:
    state = json.loads(decision.observation.payload_json)
    if not isinstance(state, dict):
        raise ValueError("Malformed public combat observation")
    objects = extract_actionable_combat_objects(
        state,
        [(a.kind, a.payload_json) for a in decision.legal_actions],
    )
    refs = tuple(
        legal_action_object_refs(a.kind, a.payload_json, objects)
        for a in decision.legal_actions
    )
    baseline = base.evaluate(decision.observation, decision.legal_actions)
    if len(refs) != len(baseline.action_logits):
        raise ValueError("Baseline policy returned wrong legal-action count")
    return objects, refs, baseline


def prepare_pooled_samples(
    cohort: Sequence[Episode], base: NeuralPolicyValueModel,
    head: PooledObjectTorch | RelationalObjectTorch, *,
    temperature: float, gamma: float,
    gae_lambda: float,
) -> tuple[list[PooledPpoSample], list[float]]:
    """Cache behavior probabilities and GAE before *any* optimizer update.

    Completed combats within censored runs are eligible; unfinished segments
    are excluded. Every PPO action is from the policy that collected it.
    """
    torch = head.torch
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid PPO behavior temperature")
    samples: list[PooledPpoSample] = []
    rewards: list[float] = []
    with torch.no_grad():
        for episode in cohort:
            visited: set[int] = set()
            for segment in episode.combat_outcomes:
                indices = list(range(segment.start_decision, segment.end_decision))
                if (not indices or indices[0] < 0
                        or indices[-1] >= len(episode.decisions)):
                    raise ValueError("Invalid/empty terminal combat decision span")
                if any(i in visited for i in indices):
                    raise ValueError("Overlapping combat segments")
                visited.update(indices)
                target = _hp_preservation_target(segment)
                prepared = []
                values: list[float] = []
                for i in indices:
                    decision = episode.decisions[i]
                    if decision.phase != "combat":
                        raise ValueError("Combat segment contains a strategy decision")
                    if not 0 <= decision.chosen_index < len(decision.legal_actions):
                        raise ValueError("Chosen action absent from legal menu")
                    objects, refs, baseline = _prepared_decision(decision, base)
                    bindings = (
                        _action_tokens(decision.legal_actions, objects)
                        if isinstance(head, RelationalObjectTorch) else None
                    )
                    v, logits = head.forward_prepared(
                        objects, bindings if bindings is not None else refs, baseline,
                    )
                    old = torch.nn.functional.log_softmax(logits / temperature, dim=0)
                    logprob = float(old[decision.chosen_index].item())
                    value = float(v.item())
                    if not math.isfinite(logprob) or not math.isfinite(value):
                        raise ValueError("Non-finite PPO behavior output")
                    topology = (
                        prepare_relational_topology(objects, bindings)
                        if bindings is not None else None
                    )
                    features = (
                        head.embedding.prepare(objects)
                        if isinstance(head, RelationalObjectTorch) else None
                    )
                    prepared.append((
                        objects, refs, bindings, baseline, decision, logprob,
                        topology, features,
                    ))
                    values.append(value)
                adv, returns = gae_terminal(
                    values, target, gamma=gamma, lam=gae_lambda,
                )
                for (
                    objects, refs, bindings, baseline, decision,
                    logprob, topology, features
                ), a, ret in zip(prepared, adv, returns, strict=True):
                    samples.append(PooledPpoSample(
                        objects, refs, baseline, decision.chosen_index,
                        logprob, a, ret, bindings, topology, features,
                    ))
                rewards.append(target)
            if episode.completed and any(
                d.phase == "combat" and i not in visited
                for i, d in enumerate(episode.decisions)
            ):
                raise ValueError("Completed episode has an unrecorded combat segment")
    return samples, rewards



def batched_ppo_loss(
    torch: Any,
    values: Any,
    logits: Sequence[Any],
    *,
    chosen_indices: Sequence[int],
    old_logprobs: Sequence[float],
    advantages: Sequence[float],
    targets: Sequence[float],
    temperature: float,
    clip_epsilon: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.002,
) -> tuple[Any, Any, Any, Any]:
    """One masked PPO loss graph for a batch of variable legal-action menus.

    Return (combined, policy, value, entropy) *means*. In contrast to
    a Python loop issuing one log_softmax and gradient graph per decision,
    this pads the variable menu once, runs a single softmax, and masks
    padded entries from entropy. Padding never changes probabilities
    or gradients of the real actions.
    """
    b = len(logits)
    if not b or values.ndim != 1 or values.shape[0] != b:
        raise ValueError("Batched PPO values and action menus mismatch")
    if any(len(rows) != b for rows in (
        chosen_indices, old_logprobs, advantages, targets,
    )):
        raise ValueError("Batched PPO metadata and action menus mismatch")
    sizes = [int(menu.numel()) for menu in logits]
    if any(length <= 0 or not 0 <= chosen_indices[i] < length
           for i, length in enumerate(sizes)):
        raise ValueError("Batched PPO requires nonempty legal menus and valid actions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid PPO temperature")
    if not 0 < clip_epsilon < 1:
        raise ValueError("Invalid PPO clipping")
    device, dtype = values.device, values.dtype
    lengths = torch.tensor(sizes, device=device, dtype=torch.long)
    valid = (
        torch.arange(max(sizes), device=device)[None, :]
        < lengths[:, None]
    )
    padded = torch.nn.utils.rnn.pad_sequence(
        logits, batch_first=True, padding_value=torch.finfo(dtype).min,
    )
    log_prob = torch.nn.functional.log_softmax(padded / temperature, dim=-1)
    indices = torch.tensor(chosen_indices, device=device, dtype=torch.long)
    selected = log_prob.gather(1, indices.unsqueeze(1)).squeeze(1)
    old = torch.tensor(old_logprobs, device=device, dtype=dtype)
    adv = torch.tensor(advantages, device=device, dtype=dtype)
    target = torch.tensor(targets, device=device, dtype=dtype)
    ratio = (selected - old).clamp(-20.0, 20.0).exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * adv, clipped * adv)
    value_loss = (values - target).square()
    # Replace padded log probabilities before multiplication to avoid
    # 0 * (-inf) under aggressive clipping or mixed-precision settings.
    entropy = -(
        log_prob.exp() * log_prob.masked_fill(~valid, 0.0)
    ).sum(dim=1)
    return (
        (policy_loss + value_weight * value_loss
         - entropy_weight * entropy).mean(),
        policy_loss.mean(), value_loss.mean(), entropy.mean(),
    )


def ppo_update_pooled(
    head: PooledObjectTorch | RelationalObjectTorch,
    samples: Sequence[PooledPpoSample],
    optimizer: Any, *, temperature: float, epochs: int = 3,
    batch_size: int = 32, sample_limit: int = 2048,
    clip_epsilon: float = 0.2, value_weight: float = 0.5,
    entropy_weight: float = 0.002, seed: int = 19,
) -> PooledPpoMetrics:
    """Minibatched clipped PPO; explicitly variable legal menus, no padding.

    Gradients flow only through v15c's residual/embedding module. The original
    base strategy and combat heads never receive an optimizer or gradient.
    """
    torch = head.torch
    if min(epochs, batch_size, sample_limit) <= 0:
        raise ValueError("Invalid PPO minibatch configuration")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("Invalid PPO clipping")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid PPO temperature")
    if any(not math.isfinite(s.target) or not math.isfinite(s.advantage)
           or not math.isfinite(s.old_logprob) for s in samples):
        raise ValueError("Non-finite PPO training sample")
    if not samples:
        return PooledPpoMetrics(0, 0, None, None, None, None)
    rng = random.Random(seed)
    rows = list(samples)
    if len(rows) > sample_limit:
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), sample_limit))]
    avg = sum(s.advantage for s in rows) / len(rows)
    var = sum((s.advantage - avg) ** 2 for s in rows) / len(rows)
    denom = math.sqrt(var + 1e-8)
    advantages = [
        max(-5.0, min(5.0, (s.advantage - avg) / denom))
        for s in rows
    ]
    nsteps = 0
    sums = [0.0, 0.0, 0.0, 0.0]
    seen = 0
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            if isinstance(head, RelationalObjectTorch):
                if any(rows[k].action_bindings is None for k in chosen):
                    raise ValueError("Missing v15d action-role bindings")
                values, logits_batch = head.forward_batch_prepared(
                    [rows[k].objects for k in chosen],
                    [rows[k].action_bindings for k in chosen],
                    [rows[k].baseline for k in chosen],
                    topologies=[
                        rows[k].relational_topology
                        if rows[k].relational_topology is not None
                        else prepare_relational_topology(
                            rows[k].objects, rows[k].action_bindings,
                        )
                        for k in chosen
                    ],
                    features=[
                        rows[k].compiled_features
                        if rows[k].compiled_features is not None
                        else head.embedding.prepare(rows[k].objects)
                        for k in chosen
                    ],
                )
                loss, actor_mean, critic_mean, entropy_mean = batched_ppo_loss(
                    torch, values, logits_batch,
                    chosen_indices=[rows[k].chosen_index for k in chosen],
                    old_logprobs=[rows[k].old_logprob for k in chosen],
                    advantages=[advantages[k] for k in chosen],
                    targets=[rows[k].target for k in chosen],
                    temperature=temperature, clip_epsilon=clip_epsilon,
                    value_weight=value_weight, entropy_weight=entropy_weight,
                )
            else:
                losses = []
                policy_losses = []
                value_losses = []
                entropies = []
                for k in chosen:
                    sample = rows[k]
                    value, logits = head.forward_prepared(
                        sample.objects, sample.action_refs, sample.baseline,
                    )
                    logp = torch.nn.functional.log_softmax(
                        logits / temperature, dim=0,
                    )
                    selected = logp[sample.chosen_index]
                    ratio = (selected - sample.old_logprob).clamp(-20.0, 20.0).exp()
                    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
                    actor = -torch.minimum(
                        ratio * advantages[k], clipped * advantages[k],
                    )
                    critic = (value - sample.target).square()
                    entropy = -(logp.exp() * logp).sum()
                    losses.append(
                        actor + value_weight * critic - entropy_weight * entropy
                    )
                    policy_losses.append(actor)
                    value_losses.append(critic)
                    entropies.append(entropy)
                loss = torch.stack(losses).mean()
                actor_mean = torch.stack(policy_losses).mean()
                critic_mean = torch.stack(value_losses).mean()
                entropy_mean = torch.stack(entropies).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.module.parameters(), 1.0)
            optimizer.step()
            for i, quantity in enumerate((
                loss.detach(), actor_mean.detach(), critic_mean.detach(),
                entropy_mean.detach(),
            )):
                sums[i] += float(quantity) * len(chosen)
            seen += len(chosen)
            nsteps += 1
    return PooledPpoMetrics(
        len(rows), nsteps, sums[0] / seen, sums[1] / seen,
        sums[2] / seen, sums[3] / seen,
    )


def _checkpoint_save(
    path: Path, *, torch: Any,
    head: PooledObjectTorch | RelationalObjectTorch, optimizer: Any,
    fingerprint: str, emulator_revision: str,
    rows: list[PooledTrainingRound],
    checkpoint_format: str = POOLED_PPO_CHECKPOINT,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": checkpoint_format,
        "fingerprint": fingerprint,
        "emulator_revision": emulator_revision,
        "module": {name: x.detach().cpu().clone()
                   for name, x in head.module.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "rows": [asdict(row) for row in rows],
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, suffix=".pending", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_pooled_combat_ppo(
    backend: EmulatorBackend, base: PhaseSplitNeuralModel,
    vocabulary: ObjectVocabulary, *,
    rounds: int, episodes_per_round: int = 32,
    width: int = 64, workers: int = 1, seed: int = 19,
    seed_prefix: str = "v15-pooled-train",
    environment: str = LEGACY,
    episode_goal_version: str = PROTOTYPE_THREE_ACT_GOAL,
    max_decisions: int = 4096, temperature_start: float = 0.85,
    temperature_end: float = 0.5, temperature_decay_rounds: int = 200,
    learning_rate: float = 0.0003, epochs: int = 3,
    batch_size: int = 32, sample_limit: int = 2048,
    clip_epsilon: float = 0.2, gamma: float = 0.995,
    gae_lambda: float = 0.98, value_weight: float = 0.5,
    entropy_weight: float = 0.002,
    checkpoint: Path | None = None, resume: bool = False,
    progress: Callable[[PooledTrainingRound], None] | None = None,
    architecture: str = "pooled",
) -> tuple[PhaseSplitNeuralModel, tuple[PooledTrainingRound, ...]]:
    """Train ONLY the combat residual, from a frozen phase-split checkpoint.

    `rounds` is the TOTAL desired number of rounds (not additional rounds).
    A fingerprint freezes every behavior parameter except that target count.
    """
    import torch

    if architecture not in ("pooled", "relational"):
        raise ValueError("Unknown v15 combat architecture")
    if min(rounds, episodes_per_round, workers, max_decisions, width) <= 0:
        raise ValueError("Expected positive training sizes")
    if not 0 < learning_rate < 1:
        raise ValueError("Invalid learning rate")
    if resume and checkpoint is None:
        raise ValueError("Resume requires a checkpoint")
    if workers > 1 and not isinstance(backend, JsonlEmulatorBackend):
        raise ValueError("Parallel training requires the JSONL emulator")
    if not isinstance(base.combat, NeuralPolicyValueModel):
        raise ValueError("v15c PPO requires a frozen v8/v13 base combat head")
    if base.combat_value_objective != "hp_preservation":
        raise ValueError("Frozen tactical critic must use hp_preservation objective")
    if min(epochs, batch_size, sample_limit) <= 0:
        raise ValueError("Invalid PPO minibatch sizes")
    if not (0.0 < gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError("Invalid GAE parameters")
    if not (math.isfinite(temperature_start) and temperature_start > 0
            and math.isfinite(temperature_end) and temperature_end > 0):
        raise ValueError("Invalid training temperatures")
    if temperature_decay_rounds <= 0:
        raise ValueError("Invalid temperature decay")
    require_episode_goal(episode_goal_version, environment)
    require_environment(backend, environment)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    head = (
        RelationalObjectTorch(vocabulary, width=width)
        if architecture == "relational"
        else PooledObjectTorch(vocabulary, width=width)
    )
    checkpoint_format = (
        RELATIONAL_PPO_CHECKPOINT if architecture == "relational"
        else POOLED_PPO_CHECKPOINT
    )
    optimizer = torch.optim.AdamW(head.module.parameters(), lr=learning_rate)
    revision = str(getattr(backend, "emulator_revision", "test-backend"))
    config = {
        "version": POOLED_PPO_VERSION, "emulator_revision": revision,
        "base_sha256": _hash_json(base.to_dict()),
        "vocabulary_sha256": _hash_json(vocabulary.to_dict()),
        "episodes_per_round": episodes_per_round, "width": width,
        "workers": workers, "seed": seed, "seed_prefix": seed_prefix,
        "environment": environment, "episode_goal_version": episode_goal_version,
        "max_decisions": max_decisions,
        "temperature_start": temperature_start, "temperature_end": temperature_end,
        "temperature_decay_rounds": temperature_decay_rounds,
        "learning_rate": learning_rate, "epochs": epochs,
        "batch_size": batch_size, "sample_limit": sample_limit,
        "clip_epsilon": clip_epsilon, "gamma": gamma, "gae_lambda": gae_lambda,
        "value_weight": value_weight, "entropy_weight": entropy_weight,
    }
    if architecture != "pooled":
        config["architecture"] = architecture
    fingerprint = _hash_json(config)
    rows: list[PooledTrainingRound] = []
    if resume:
        assert checkpoint is not None
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("format") != checkpoint_format:
            raise ValueError("Wrong v15c PPO checkpoint format")
        if (saved.get("fingerprint") != fingerprint
                or saved.get("emulator_revision") != revision):
            raise ValueError("v15c PPO checkpoint configuration/emulator mismatch")
        old = saved["module"]
        expected = head.module.state_dict()
        if set(old) != set(expected):
            raise ValueError("v15c PPO checkpoint parameter mismatch")
        if any(old[k].shape != expected[k].shape for k in expected):
            raise ValueError("v15c PPO checkpoint tensor shape mismatch")
        head.module.load_state_dict(old)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        rows = [PooledTrainingRound(**row) for row in saved["rows"]]
        if any(row.round_index != i for i, row in enumerate(rows)):
            raise ValueError("v15c PPO checkpoint rounds are not contiguous")
        if len(rows) > rounds:
            raise ValueError("Checkpoint contains more rounds than requested")

    with ExitStack() as scope:
        pool = (
            scope.enter_context(open_worker_pool(backend, workers))
            if workers > 1 and isinstance(backend, JsonlEmulatorBackend)
            else None
        )
        for i in range(len(rows), rounds):
            policy = PhaseSplitNeuralModel(
                base.strategy, head.export(base.combat),
                f"v15-{architecture}-before-round-{i}",
                combat_value_objective="hp_preservation",
            )
            temperature = temperature_for_round(
                i, start=temperature_start, end=temperature_end,
                decay_rounds=temperature_decay_rounds,
            )
            seeds = [f"{seed_prefix}-{i}-{j}" for j in range(episodes_per_round)]
            start = time.perf_counter()
            if pool is not None:
                cohort = collect_parallel(
                    pool, policy, seeds, base_seed=seed,
                    max_decisions=max_decisions, policy_id="prototype-fair-v0",
                    environment=environment, temperature=temperature,
                    episode_goal_version=episode_goal_version,
                )
            else:
                cohort = tuple(collect_public_episode(
                    backend, policy, seed=episode_seed,
                    actor_rng=random.Random(episode_actor_seed(seed, episode_seed)),
                    max_decisions=max_decisions, environment=environment,
                    temperature=temperature, episode_goal_version=episode_goal_version,
                ) for episode_seed in seeds)
            rollout_time = time.perf_counter() - start
            start = time.perf_counter()
            samples, targets = prepare_pooled_samples(
                cohort, base.combat, head,
                temperature=temperature, gamma=gamma, gae_lambda=gae_lambda,
            )
            result = ppo_update_pooled(
                head, samples, optimizer,
                temperature=temperature, epochs=epochs, batch_size=batch_size,
                sample_limit=sample_limit, clip_epsilon=clip_epsilon,
                value_weight=value_weight, entropy_weight=entropy_weight,
                seed=seed + i * 1000003,
            )
            row = PooledTrainingRound(
                round_index=i, played=len(cohort),
                completed=sum(ep.completed for ep in cohort),
                censored=sum(not ep.completed for ep in cohort),
                wins=sum(ep.won for ep in cohort),
                resolved_combats=sum(len(ep.combat_outcomes) for ep in cohort),
                combat_decisions=len(samples),
                optimizer_steps=result.steps,
                mean_loss=result.loss,
                mean_combat_target=sum(targets) / len(targets) if targets else None,
                rollout_seconds=rollout_time,
                optimizer_seconds=time.perf_counter() - start,
            )
            rows.append(row)
            if checkpoint is not None:
                _checkpoint_save(
                    checkpoint, torch=torch, head=head, optimizer=optimizer,
                    fingerprint=fingerprint, emulator_revision=revision, rows=rows,
                    checkpoint_format=checkpoint_format,
                )
            print(
                f"[v15-{architecture}] round={i + 1}/{rounds} "
                f"runs={row.played} complete={row.completed} wins={row.wins} "
                f"resolved_combats={row.resolved_combats} "
                f"combat_decisions={row.combat_decisions} "
                f"updates={row.optimizer_steps} loss={row.mean_loss!r} "
                f"rollout_s={row.rollout_seconds:.2f} "
                f"optimizer_s={row.optimizer_seconds:.2f}",
                flush=True,
            )
            if progress is not None:
                progress(row)
    final = PhaseSplitNeuralModel(
        base.strategy, head.export(base.combat),
        f"v15-{architecture}-{fingerprint[:12]}-{rounds}",
        combat_value_objective="hp_preservation",
    )
    return final, tuple(rows)
