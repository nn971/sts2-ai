"""Cached, padded, batched PPO for v7 and instance-aware v8 combat.

All features are computed once from PUBLIC observations for an on-policy
cohort. The model and objective match phase_split_ppo.py, which remains the
reference implementation. Variable legal menus/enemy sets use explicit masks;
target IDs are resolved to within-frame enemy slots before tensor operations.
No private emulator state is used here.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from sts2_ai.models.enemy_instances import (
    action_target_instance,
    enemy_instance_vectors,
    instance_action_features,
    instance_global_features,
)
from sts2_ai.models.hashed_linear import (
    neural_action_features,
    state_dict,
    state_features,
    tactical_action_features,
    public_resources_tactical_action_features,
)
from sts2_ai.models.tactical_state import (
    tactical_state_features,
    relational_tactical_state_features,
    damage_tactical_state_features,
    public_resources_tactical_state_features,
)
from sts2_ai.training.neural import _dense
from sts2_ai.training.selfplay import Episode, PublicDecision
from sts2_ai.training.phase_split_ppo import PpoDiagnostics, gae_terminal


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    phase: str
    state: Any                 # [dimension], CPU float32
    actions: Any               # [legal_actions, dimension], CPU float32
    chosen_index: int
    enemies: Any | None = None # [enemies, dimension] for v8
    targets: Any | None = None # [legal_actions], enemy slot or -1


@dataclass(frozen=True, slots=True)
class PaddedBatch:
    states: Any
    actions: Any
    mask: Any
    chosen: Any
    enemies: Any | None
    enemy_mask: Any | None
    targets: Any | None


@dataclass(frozen=True, slots=True)
class PreparedSample:
    encoded: EncodedDecision
    old_logprob: float
    advantage: float
    target: float


def encode_decision(
    decision: PublicDecision, dimension: int, encoding: str, torch: Any,
) -> EncodedDecision:
    state = state_dict(decision.observation.payload_json)
    if not decision.legal_actions:
        raise ValueError("PPO decision lacks legal actions")
    if not 0 <= decision.chosen_index < len(decision.legal_actions):
        raise ValueError("Chosen PPO action lies outside legal menu")

    if decision.phase == "strategy":
        state_fn = state_features
        action_fn = neural_action_features
    elif decision.phase == "combat":
        state_fn = {
            "legacy": state_features,
            "structured": tactical_state_features,
            "relational": relational_tactical_state_features,
            "relational_damage": damage_tactical_state_features,
            "public_resources": public_resources_tactical_state_features,
            "enemy_instances": instance_global_features,
        }.get(encoding)
        if state_fn is None:
            raise ValueError("Unknown tactical feature schema")
        action_fn = (
            instance_action_features if encoding == "enemy_instances"
            else public_resources_tactical_action_features
            if encoding == "public_resources" else tactical_action_features
        )
    else:
        raise ValueError("Unknown PPO phase")

    state_tensor = torch.tensor(
        _dense(state_fn(state, dimension), dimension), dtype=torch.float32
    )
    action_tensor = torch.tensor([
        _dense(action_fn(state, action.kind, action.payload_json, dimension), dimension)
        for action in decision.legal_actions
    ], dtype=torch.float32)
    if decision.phase != "combat" or encoding != "enemy_instances":
        return EncodedDecision(
            decision.phase, state_tensor, action_tensor, decision.chosen_index,
        )

    nodes = enemy_instance_vectors(state, dimension)
    ids = list(nodes)  # order is arbitrary but IDs only select the correct slot
    slot = {instance_id: i for i, instance_id in enumerate(ids)}
    enemy_tensor = (
        torch.tensor([_dense(nodes[key], dimension) for key in ids], dtype=torch.float32)
        if ids else torch.empty((0, dimension), dtype=torch.float32)
    )
    target_tensor = torch.tensor([
        slot[target] if (target := action_target_instance(
            action.payload_json, nodes
        )) is not None else -1
        for action in decision.legal_actions
    ], dtype=torch.long)
    return EncodedDecision(
        decision.phase, state_tensor, action_tensor, decision.chosen_index,
        enemy_tensor, target_tensor,
    )


def collate(
    records: list[EncodedDecision], torch: Any,
) -> PaddedBatch:
    if not records:
        raise ValueError("Cannot batch an empty PPO minibatch")
    if len({item.phase for item in records}) != 1:
        raise ValueError("Mixing strategic and combat samples in one minibatch")
    states = torch.stack([item.state for item in records])
    counts = torch.tensor([item.actions.shape[0] for item in records], dtype=torch.long)
    actions = torch.nn.utils.rnn.pad_sequence(
        [item.actions for item in records], batch_first=True,
    )
    mask = torch.arange(actions.shape[1])[None, :] < counts[:, None]
    chosen = torch.tensor([item.chosen_index for item in records], dtype=torch.long)
    if records[0].enemies is None:
        if any(item.enemies is not None or item.targets is not None for item in records):
            raise ValueError("Inconsistent instance encoder batch")
        return PaddedBatch(states, actions, mask, chosen, None, None, None)

    if any(item.enemies is None or item.targets is None for item in records):
        raise ValueError("Missing instance enemy/target tensors")
    # A minimum of one padded enemy slot permits gather() even for an
    # empty enemy set. Padding must never contribute to the pooled context.
    max_enemies = max(1, *(int(item.enemies.shape[0]) for item in records))
    dimension = states.shape[1]
    padded_enemies = torch.zeros(
        (len(records), max_enemies, dimension), dtype=torch.float32,
    )
    enemy_mask = torch.zeros(
        (len(records), max_enemies), dtype=torch.bool,
    )
    targets = torch.full(
        (len(records), actions.shape[1]), -1, dtype=torch.long,
    )
    for index, item in enumerate(records):
        size = item.enemies.shape[0]
        if size:
            padded_enemies[index, :size] = item.enemies
            enemy_mask[index, :size] = True
        targets[index, :item.actions.shape[0]] = item.targets
    return PaddedBatch(
        states, actions, mask, chosen, padded_enemies, enemy_mask, targets,
    )


def forward_batch(
    batch: PaddedBatch, params: dict[str, Any], torch: Any,
) -> tuple[Any, Any]:
    """Same policy/value algebra as _forward and instance_forward, batched."""
    linear = torch.nn.functional.linear
    relu = torch.nn.functional.relu
    hidden_input = linear(
        batch.states, params["state_weight"], params["state_bias"],
    )
    target_hidden = None
    if batch.enemies is not None:
        assert batch.enemy_mask is not None and batch.targets is not None
        enemy_hidden = relu(linear(
            batch.enemies, params["enemy_weight"], params["enemy_bias"],
        ))
        mask = batch.enemy_mask.unsqueeze(-1)
        pooled = (enemy_hidden * mask).sum(dim=1) / (
            batch.enemy_mask.sum(dim=1, keepdim=True).clamp(min=1)
        )
        hidden_input = hidden_input + pooled * params["enemy_context_weight"]
        hidden_size = enemy_hidden.shape[-1]
        target_hidden = enemy_hidden.gather(
            1, batch.targets.clamp(min=0).unsqueeze(-1).expand(
                -1, -1, hidden_size,
            ),
        )
        target_hidden = target_hidden * (batch.targets >= 0).unsqueeze(-1)
    hidden = relu(hidden_input)
    value = torch.tanh(
        torch.mv(hidden, params["value_weight"]) + params["value_bias"],
    )
    action_hidden_input = (
        hidden.unsqueeze(1)
        + linear(batch.actions, params["action_weight"], params["action_bias"])
    )
    if target_hidden is not None:
        action_hidden_input = (
            action_hidden_input
            + target_hidden * params["enemy_target_weight"]
        )
    action_hidden = relu(action_hidden_input)
    logits = torch.matmul(action_hidden, params["policy_weight"]) + params["policy_bias"]
    # Invalid slots have zero probability and never enter PPO's entropy term.
    logits = logits.masked_fill(~batch.mask, -1.0e9)
    return value, logits


def _prepare(
    *, torch: Any, run_targets: list[tuple[Episode, float]],
    local_targets: list[list[tuple[Any, float]]],
    params: dict[str, dict[str, Any]], dimension: int, encoding: str,
    temperature: float, gamma: float, gae_lambda: float,
    inference_batch_size: int = 128,
) -> dict[str, list[PreparedSample]]:
    """Encode the whole cohort ONCE, including old-policy logits and values."""
    encoded_episodes = [
        [encode_decision(decision, dimension, encoding, torch) for decision in ep.decisions]
        for ep, _ in run_targets
    ]
    # Flatten records by phase, retaining their episode and decision indices.
    grouped: dict[str, list[tuple[int, int, EncodedDecision]]] = {
        "strategy": [], "combat": [],
    }
    for episode_index, entries in enumerate(encoded_episodes):
        for decision_index, item in enumerate(entries):
            grouped[item.phase].append((episode_index, decision_index, item))
    old_values: list[list[float]] = [
        [0.0] * len(entries) for entries in encoded_episodes
    ]
    old_logprobs: list[list[float]] = [
        [0.0] * len(entries) for entries in encoded_episodes
    ]
    with torch.no_grad():
        for phase, rows in grouped.items():
            for start in range(0, len(rows), inference_batch_size):
                subset = rows[start:start + inference_batch_size]
                batch = collate([item for _, _, item in subset], torch)
                values, logits = forward_batch(batch, params[phase], torch)
                logprob = torch.nn.functional.log_softmax(logits / temperature, dim=1)
                selected_logprob = logprob.gather(
                    1, batch.chosen.unsqueeze(1),
                ).squeeze(1)
                for index, (ep_idx, decision_idx, _) in enumerate(subset):
                    old_values[ep_idx][decision_idx] = float(values[index])
                    old_logprobs[ep_idx][decision_idx] = float(selected_logprob[index])

    result: dict[str, list[PreparedSample]] = {"strategy": [], "combat": []}
    for ep_index, ((episode, run_target), segments) in enumerate(
        zip(run_targets, local_targets, strict=True)
    ):
        decisions = encoded_episodes[ep_index]
        values = old_values[ep_index]
        probabilities = old_logprobs[ep_index]
        combat_indices: set[int] = set()
        for segment, target in segments:
            indices = list(range(segment.start_decision, segment.end_decision))
            if not indices:
                continue
            if any(decisions[i].phase != "combat" for i in indices):
                raise ValueError("A combat GAE segment contains strategic decisions")
            if any(i in combat_indices for i in indices):
                raise ValueError("Overlapping combat GAE segments")
            combat_indices.update(indices)
            advantages, returns = gae_terminal(
                [values[i] for i in indices], target,
                gamma=gamma, lam=gae_lambda,
            )
            for index, advantage, value_target in zip(
                indices, advantages, returns, strict=True,
            ):
                result["combat"].append(PreparedSample(
                    decisions[index], probabilities[index], advantage, value_target,
                ))
        for index, decision in enumerate(decisions):
            if decision.phase == "combat":
                if index not in combat_indices:
                    raise ValueError("A combat decision has no terminal combat record")
            elif decision.phase == "strategy":
                result["strategy"].append(PreparedSample(
                    decision, probabilities[index],
                    run_target - values[index], run_target,
                ))
            else:
                raise ValueError("Unknown decision phase")
    return result


def ppo_update_batched(
    *, torch: Any, run_targets: list[tuple[Episode, float]],
    local_targets: list[list[tuple[Any, float]]],
    params: dict[str, dict[str, Any]],
    optimizers: dict[str, Any], dimension: int, encoding: str,
    temperature: float, gamma: float, gae_lambda: float, epochs: int,
    batch_size: int, sample_limit: int, clip_epsilon: float,
    value_weight: float, entropy_weight: float, seed: int,
) -> PpoDiagnostics:
    if min(epochs, batch_size, sample_limit) <= 0:
        raise ValueError("PPO epoch/batch/sample sizes must be positive")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("PPO clip epsilon must lie in (0,1)")
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError("PPO behavior temperature must be positive and finite")
    prepared = _prepare(
        torch=torch, run_targets=run_targets, local_targets=local_targets,
        params=params, dimension=dimension, encoding=encoding,
        temperature=temperature, gamma=gamma, gae_lambda=gae_lambda,
    )
    rng = random.Random(seed)
    all_losses: list[float] = []
    summary: dict[str, dict[str, float]] = {}
    steps = 0
    counts = {phase: len(samples) for phase, samples in prepared.items()}
    for phase, all_samples in prepared.items():
        if not all_samples:
            continue
        if len(all_samples) > sample_limit:
            indices = sorted(rng.sample(range(len(all_samples)), sample_limit))
            samples = [all_samples[i] for i in indices]
        else:
            samples = all_samples

        mean = sum(s.advantage for s in samples) / len(samples)
        variance = sum((s.advantage - mean) ** 2 for s in samples) / len(samples)
        std = math.sqrt(variance + 1e-8)
        normalized = [
            max(-5.0, min(5.0, (s.advantage - mean) / std))
            for s in samples
        ]
        phase_loss = phase_policy = phase_value = phase_entropy = 0.0
        phase_kl = phase_clip = 0.0
        phase_seen = phase_steps = 0
        for _ in range(epochs):
            order = list(range(len(samples)))
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                chosen = order[start:start + batch_size]
                if not chosen:
                    continue
                batch = collate([samples[i].encoded for i in chosen], torch)
                values, logits = forward_batch(batch, params[phase], torch)
                logprobs = torch.nn.functional.log_softmax(logits / temperature, dim=1)
                selected = logprobs.gather(1, batch.chosen.unsqueeze(1)).squeeze(1)
                old = torch.tensor(
                    [samples[i].old_logprob for i in chosen], dtype=torch.float32,
                )
                adv = torch.tensor([normalized[i] for i in chosen], dtype=torch.float32)
                targets = torch.tensor(
                    [samples[i].target for i in chosen], dtype=torch.float32,
                )
                log_ratio = selected - old
                ratio = torch.exp(log_ratio.clamp(-20, 20))
                surrogate = torch.minimum(
                    ratio * adv,
                    torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv,
                )
                policy_loss = -surrogate
                critic_loss = (values - targets).square()
                # Padding contributes exactly zero entropy probability.
                entropy = -(logprobs.exp() * logprobs).sum(dim=1)
                loss = (
                    policy_loss + value_weight * critic_loss
                    - entropy_weight * entropy
                ).mean()
                optimizers[phase].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(params[phase].values()), 1.0,
                )
                optimizers[phase].step()
                steps += 1
                phase_steps += 1
                size = len(chosen)
                phase_seen += size
                phase_loss += float(loss.detach()) * size
                phase_policy += float(policy_loss.mean().detach()) * size
                phase_value += float(critic_loss.mean().detach()) * size
                phase_entropy += float(entropy.mean().detach()) * size
                phase_kl += float((-log_ratio.detach()).sum())
                phase_clip += int(
                    ((ratio.detach() - 1.0).abs() > clip_epsilon).sum()
                )
        if phase_seen:
            summary[phase] = {
                "policy": phase_policy / phase_seen,
                "value_mse": phase_value / phase_seen,
                "entropy": phase_entropy / phase_seen,
                "approx_kl": phase_kl / phase_seen,
                "clip_fraction": phase_clip / phase_seen,
                "minibatch_updates": float(phase_steps),
                "used_samples": float(len(samples)),
            }
            all_losses.append(phase_loss / phase_seen)
    return PpoDiagnostics(
        steps=steps,
        loss=sum(all_losses) if all_losses else None,
        counts=counts,
        phase=summary,
    )
