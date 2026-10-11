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
from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe
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
from sts2_ai.models.pooled_objects import PooledObjectTorch, PortablePooledObjectModel
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.models.relational_objects import (
    PortableRelationalObjectModel,
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
from sts2_ai.training.snapshot_curriculum import (
    collect_snapshot_cohort,
    sample_snapshot_curriculum,
)
from sts2_ai.training.snapshot_importance import (
    build_importance_distribution,
    importance_diagnostics,
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
    importance_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PooledPpoMetrics:
    samples: int
    steps: int
    loss: float | None
    mean_policy_loss: float | None
    mean_value_loss: float | None
    mean_entropy: float | None
    # Opt-in v15p diagnostics; absent from old optimizer checkpoints.
    mean_preclip_grad_norm: float | None = None
    max_preclip_grad_norm: float | None = None
    gradient_clip_fraction: float | None = None
    raw_advantage_std: float | None = None
    decision_weight_ess_fraction: float | None = None


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
    # Optional for backwards compatibility with v15j's 5-round checkpoint.
    ppo_samples_used: int | None = None
    mean_policy_loss: float | None = None
    mean_value_loss: float | None = None
    mean_entropy: float | None = None
    rollout_source: str = "full_run"
    importance_ess: float | None = None
    importance_weight_max: float | None = None
    mean_preclip_grad_norm: float | None = None
    max_preclip_grad_norm: float | None = None
    gradient_clip_fraction: float | None = None
    raw_advantage_std: float | None = None
    decision_weight_ess_fraction: float | None = None


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
    episode_weights: Sequence[float] | None = None,
) -> tuple[list[PooledPpoSample], list[float]]:
    """Cache behavior probabilities and GAE before *any* optimizer update.

    Completed combats within censored runs are eligible; unfinished segments
    are excluded. Every PPO action is from the policy that collected it.
    """
    torch = head.torch
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid PPO behavior temperature")
    if episode_weights is not None:
        if len(episode_weights) != len(cohort):
            raise ValueError("Importance weights must match snapshot episodes")
        if any(not math.isfinite(w) or w <= 0 for w in episode_weights):
            raise ValueError("Importance weights must be finite and positive")
        if any(not ep.completed or len(ep.combat_outcomes) != 1 for ep in cohort):
            raise ValueError("Importance snapshots require one resolved combat per draw")
    samples: list[PooledPpoSample] = []
    rewards: list[float] = []
    with torch.no_grad():
        for episode_index, episode in enumerate(cohort):
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
                        (episode_weights[episode_index] / len(indices))
                        if episode_weights is not None else 1.0,
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
    sample_weights: Sequence[float] | None = None,
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
    )) or (sample_weights is not None and len(sample_weights) != b):
        raise ValueError("Batched PPO metadata and action menus mismatch")
    sizes = [int(menu.numel()) for menu in logits]
    if any(length <= 0 or not 0 <= chosen_indices[i] < length
           for i, length in enumerate(sizes)):
        raise ValueError("Batched PPO requires nonempty legal menus and valid actions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid PPO temperature")
    if not 0 < clip_epsilon < 1:
        raise ValueError("Invalid PPO clipping")
    if sample_weights is not None and any(
        not math.isfinite(w) or w <= 0 for w in sample_weights
    ):
        raise ValueError("PPO importance weights must be finite and positive")
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
    if sample_weights is None:
        def reduce(x: Any) -> Any:
            return x.mean()
    else:
        weights = torch.tensor(sample_weights, device=device, dtype=dtype)

        def reduce(x: Any) -> Any:
            return (weights * x).sum() / weights.sum()
    return (
        reduce(policy_loss + value_weight * value_loss
               - entropy_weight * entropy),
        reduce(policy_loss), reduce(value_loss), reduce(entropy),
    )


def ppo_update_pooled(
    head: PooledObjectTorch | RelationalObjectTorch,
    samples: Sequence[PooledPpoSample],
    optimizer: Any, *, temperature: float, epochs: int = 3,
    batch_size: int = 32, sample_limit: int = 2048,
    clip_epsilon: float = 0.2, value_weight: float = 0.5,
    entropy_weight: float = 0.002, seed: int = 19,
    diagnostics: bool = False,
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
           or not math.isfinite(s.old_logprob)
           or not math.isfinite(s.importance_weight) or s.importance_weight <= 0
           for s in samples):
        raise ValueError("Non-finite PPO training sample")
    if not samples:
        return PooledPpoMetrics(0, 0, None, None, None, None)
    rng = random.Random(seed)
    rows = list(samples)
    if len(rows) > sample_limit:
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), sample_limit))]
    importance_mode = any(s.importance_weight != 1.0 for s in rows)
    if importance_mode:
        weight_sum = math.fsum(s.importance_weight for s in rows)
        avg = math.fsum(
            s.importance_weight * s.advantage for s in rows
        ) / weight_sum
        var = math.fsum(
            s.importance_weight * (s.advantage - avg) ** 2 for s in rows
        ) / weight_sum
    else:
        avg = sum(s.advantage for s in rows) / len(rows)
        var = sum((s.advantage - avg) ** 2 for s in rows) / len(rows)
    denom = math.sqrt(var + 1e-8)
    weights_for_ess = [s.importance_weight for s in rows]
    weight_total = math.fsum(weights_for_ess)
    decision_weight_ess_fraction = (
        weight_total ** 2 / math.fsum(w * w for w in weights_for_ess) / len(rows)
        if diagnostics else None
    )
    advantages = [
        max(-5.0, min(5.0, (s.advantage - avg) / denom))
        for s in rows
    ]
    nsteps = 0
    sums = [0.0, 0.0, 0.0, 0.0]
    seen = 0
    grad_norms: list[float] = []
    clipped_gradients = 0
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
                    sample_weights=(
                        [rows[k].importance_weight for k in chosen]
                        if importance_mode else None
                    ),
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
                if importance_mode:
                    weights = torch.tensor(
                        [rows[k].importance_weight for k in chosen],
                        dtype=losses[0].dtype, device=losses[0].device,
                    )
                    weights = weights / weights.sum()
                    loss = (weights * torch.stack(losses)).sum()
                    actor_mean = (weights * torch.stack(policy_losses)).sum()
                    critic_mean = (weights * torch.stack(value_losses)).sum()
                    entropy_mean = (weights * torch.stack(entropies)).sum()
                else:
                    loss = torch.stack(losses).mean()
                    actor_mean = torch.stack(policy_losses).mean()
                    critic_mean = torch.stack(value_losses).mean()
                    entropy_mean = torch.stack(entropies).mean()
            optimizer.zero_grad()
            loss.backward()
            preclip = torch.nn.utils.clip_grad_norm_(head.module.parameters(), 1.0)
            if diagnostics:
                gradient_norm = float(preclip.detach().item())
                if not math.isfinite(gradient_norm):
                    raise ValueError("Non-finite v15p PPO gradient")
                grad_norms.append(gradient_norm)
                clipped_gradients += int(gradient_norm > 1.0)
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
        mean_preclip_grad_norm=(
            math.fsum(grad_norms) / len(grad_norms) if diagnostics else None
        ),
        max_preclip_grad_norm=max(grad_norms) if diagnostics else None,
        gradient_clip_fraction=(
            clipped_gradients / len(grad_norms) if diagnostics else None
        ),
        raw_advantage_std=math.sqrt(var) if diagnostics else None,
        decision_weight_ess_fraction=decision_weight_ess_fraction,
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



def initialize_combat_head(
    head: PooledObjectTorch | RelationalObjectTorch,
    base: PhaseSplitNeuralModel, initial: PhaseSplitNeuralModel,
) -> None:
    """Warm-start trainable v15 weights from an exact portable public model.

    The base (both frozen heads), vocabulary and architecture must match.
    Portable weights are copied to fresh Torch parameters; optimizer moments
    are intentionally reset for a new experiment, never used for --resume.
    """
    import torch

    combat = initial.combat
    if not isinstance(base.combat, NeuralPolicyValueModel):
        raise ValueError("Initialization requires a frozen legacy combat base")
    if initial.strategy.to_dict() != base.strategy.to_dict():
        raise ValueError("Initialization strategy differs from frozen base")
    if initial.combat_value_objective != base.combat_value_objective:
        raise ValueError("Initialization combat objective mismatch")
    if isinstance(head, RelationalObjectTorch):
        if not isinstance(combat, PortableRelationalObjectModel):
            raise ValueError("Initialization requires relational v15 weights")
    elif not isinstance(combat, PortablePooledObjectModel):
        raise ValueError("Initialization requires pooled v15 weights")
    if combat.base.to_dict() != base.combat.to_dict():
        raise ValueError("Initialization combat base mismatch")
    if (combat.width != head.width
            or combat.vocabulary.to_dict() != head.vocabulary.to_dict()):
        raise ValueError("Initialization width/vocabulary mismatch")
    reference = head.module.state_dict()
    if set(combat.weights) != set(reference):
        raise ValueError("Initialization tensor names mismatch")
    converted = {}
    for name, existing in reference.items():
        tensor = torch.as_tensor(combat.weights[name], dtype=existing.dtype)
        if tensor.shape != existing.shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Initialization invalid tensor: {name}")
        converted[name] = tensor
    head.module.load_state_dict(converted)


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
    snapshot_every: int = 0,
    snapshot_recipes: Sequence[CombatSnapshotRecipe] | None = None,
    snapshot_sampling: str = "balanced",
    snapshot_target_balance: float = 0.5,
    snapshot_priority_mix: float = 0.7,
    diagnose_ppo: bool = False,
    initialize_from: PhaseSplitNeuralModel | None = None,
    on_round: Callable[
        [tuple[PooledTrainingRound, ...], PhaseSplitNeuralModel | None], None
    ] | None = None,
) -> tuple[PhaseSplitNeuralModel, tuple[PooledTrainingRound, ...]]:
    """Train ONLY the combat residual, from a frozen phase-split checkpoint.

    `rounds` is the TOTAL desired number of rounds (not additional rounds).
    A fingerprint freezes every behavior parameter except that target count.
    """
    import torch

    if architecture not in ("pooled", "relational"):
        raise ValueError("Unknown v15 combat architecture")
    if snapshot_every < 0:
        raise ValueError("Snapshot interval must be nonnegative")
    if min(rounds, episodes_per_round, workers, max_decisions, width) <= 0:
        raise ValueError("Expected positive training sizes")
    if not 0 < learning_rate < 1:
        raise ValueError("Invalid learning rate")
    if resume and checkpoint is None:
        raise ValueError("Resume requires a checkpoint")
    if snapshot_recipes is not None:
        if not snapshot_recipes:
            raise ValueError("Snapshot training requires at least one recipe")
        if environment != "native-overgrowth":
            raise ValueError("Snapshots require the native Overgrowth environment")
        if snapshot_sampling not in ("balanced", "natural", "importance"):
            raise ValueError("Invalid snapshot sampling mode")
        for recipe in snapshot_recipes:
            recipe.validate()
            if recipe.emulator_revision != backend.emulator_revision:
                raise ValueError("Snapshot corpus emulator revision mismatch")
    importance_sampler = (
        build_importance_distribution(
            snapshot_recipes, target_balance=snapshot_target_balance,
            priority_mix=snapshot_priority_mix,
        )
        if snapshot_recipes is not None and snapshot_sampling == "importance"
        else None
    )
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
    if initialize_from is not None:
        initialize_combat_head(head, base, initialize_from)
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
    if initialize_from is not None:
        config["initialize_from_sha256"] = _hash_json(initialize_from.to_dict())
    if diagnose_ppo:
        config["diagnose_ppo"] = True
    if snapshot_recipes is not None:
        # This changes the fingerprint only for the new opt-in mode.
        # Existing v15l checkpoint fingerprints and resumption remain intact.
        config["snapshot_curriculum"] = {
            "sampling": snapshot_sampling,
            "recipe_sha256": _hash_json([asdict(r) for r in snapshot_recipes]),
        }
        if importance_sampler is not None:
            config["snapshot_curriculum"]["target_balance"] = snapshot_target_balance
            config["snapshot_curriculum"]["priority_mix"] = snapshot_priority_mix
            config["snapshot_curriculum"]["priority_version"] = "hp-tier-proxy-v1"
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
            importance_weights = None
            diagnostics = None
            if snapshot_recipes is not None:
                if importance_sampler is not None:
                    draws = importance_sampler.draw(
                        count=episodes_per_round, seed=seed + i * 1000003,
                    )
                    selected = tuple(draw.recipe for draw in draws)
                    importance_weights = tuple(draw.weight for draw in draws)
                    diagnostics = importance_diagnostics(draws)
                else:
                    selected = sample_snapshot_curriculum(
                        snapshot_recipes, count=episodes_per_round,
                        seed=seed + i * 1000003, mode=snapshot_sampling,
                    )
                cohort = collect_snapshot_cohort(
                    backend, policy, selected,
                    round_index=i, actor_seed=seed, temperature=temperature,
                    max_decisions=max_decisions, pool=pool, workers=workers,
                )
            elif pool is not None:
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
                episode_weights=importance_weights,
            )
            result = ppo_update_pooled(
                head, samples, optimizer,
                temperature=temperature, epochs=epochs, batch_size=batch_size,
                sample_limit=sample_limit, clip_epsilon=clip_epsilon,
                value_weight=value_weight, entropy_weight=entropy_weight,
                seed=seed + i * 1000003,
                diagnostics=diagnose_ppo,
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
                ppo_samples_used=result.samples,
                mean_policy_loss=result.mean_policy_loss,
                mean_value_loss=result.mean_value_loss,
                mean_entropy=result.mean_entropy,
                rollout_source=(
                    "combat_snapshots" if snapshot_recipes is not None else "full_run"
                ),
                importance_ess=(
                    diagnostics["effective_sample_size"]
                    if diagnostics is not None else None
                ),
                importance_weight_max=(
                    diagnostics["weight_max"] if diagnostics is not None else None
                ),
                mean_preclip_grad_norm=result.mean_preclip_grad_norm,
                max_preclip_grad_norm=result.max_preclip_grad_norm,
                gradient_clip_fraction=result.gradient_clip_fraction,
                raw_advantage_std=result.raw_advantage_std,
                decision_weight_ess_fraction=result.decision_weight_ess_fraction,
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
                f"ppo_samples={row.ppo_samples_used} "
                f"updates={row.optimizer_steps} loss={row.mean_loss!r} "
                f"policy={row.mean_policy_loss!r} value={row.mean_value_loss!r} "
                f"entropy={row.mean_entropy!r} "
                f"importance_ess={row.importance_ess!r} "
                f"importance_weight_max={row.importance_weight_max!r} "
                f"grad_norm={row.mean_preclip_grad_norm!r} "
                f"grad_clip_fraction={row.gradient_clip_fraction!r} "
                f"advantage_std={row.raw_advantage_std!r} "
                f"rollout_s={row.rollout_seconds:.2f} "
                f"optimizer_s={row.optimizer_seconds:.2f}",
                flush=True,
            )
            if progress is not None:
                progress(row)
            if on_round is not None:
                milestone = None
                if snapshot_every and (i + 1) % snapshot_every == 0:
                    # Export only after the corresponding optimizer checkpoint
                    # has been durably committed. Re-exporting is read-only
                    # and does not affect PPO's RNG or optimizer trajectory.
                    milestone = PhaseSplitNeuralModel(
                        base.strategy, head.export(base.combat),
                        f"v15-{architecture}-{fingerprint[:12]}-{i + 1}",
                        combat_value_objective="hp_preservation",
                    )
                on_round(tuple(rows), milestone)
    final = PhaseSplitNeuralModel(
        base.strategy, head.export(base.combat),
        f"v15-{architecture}-{fingerprint[:12]}-{rounds}",
        combat_value_objective="hp_preservation",
    )
    return final, tuple(rows)
