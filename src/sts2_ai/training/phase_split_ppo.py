"""On-policy clipped PPO with combat-boundary GAE and run-level strategy returns.

No teacher, oracle, future RNG, or private emulator state is used. The
sampling distribution includes rollout temperature: the saved old log
probability and optimized policy both use softmax(logits / temperature).
Censored episodes are excluded, never assigned a false defeat.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from sts2_ai.training.selfplay import Episode, PublicDecision


@dataclass(frozen=True, slots=True)
class PpoSample:
    decision: PublicDecision
    old_logprob: float
    advantage: float
    target: float


@dataclass(frozen=True, slots=True)
class PpoDiagnostics:
    steps: int
    loss: float | None
    counts: dict[str, int]
    phase: dict[str, dict[str, float]]


def gae_terminal(
    values: list[float], reward: float, *, gamma: float, lam: float,
) -> tuple[list[float], list[float]]:
    """GAE for a combat ending after the last decision, reward at that boundary.

    Earlier decisions have zero immediate reward. No bootstrap beyond an
    actually completed combat: V(s_terminal)=0. This does not infer hidden
    future outcomes or label capped episodes as losses.
    """
    if not 0.0 < gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("GAE gamma and lambda must be in [0,1] with gamma > 0")
    if not math.isfinite(reward):
        raise ValueError("Nonfinite combat boundary reward")
    advantages = [0.0] * len(values)
    returns = [0.0] * len(values)
    gae = 0.0
    for index in range(len(values) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        immediate = reward if index == len(values) - 1 else 0.0
        delta = immediate + gamma * next_value - values[index]
        gae = delta + gamma * lam * gae
        advantages[index] = gae
        returns[index] = gae + values[index]
    return advantages, returns


def _prepare_samples(
    *,
    torch: Any,
    run_targets: list[tuple[Episode, float]],
    local_targets: list[list[tuple[Any, float]]],
    params: dict[str, dict[str, Any]],
    dimension: int,
    encoding: str,
    temperature: float,
    gamma: float,
    gae_lambda: float,
) -> dict[str, list[PpoSample]]:
    # Import lazily to avoid a circular dependency with the stable v3 trainer.
    from sts2_ai.training.phase_split_selfplay import _forward_decision

    result: dict[str, list[PpoSample]] = {"strategy": [], "combat": []}
    for (ep, run_target), segments in zip(run_targets, local_targets, strict=True):
        old_values: list[float] = []
        old_logprobs: list[float] = []
        with torch.no_grad():
            for dec in ep.decisions:
                value, logits = _forward_decision(
                    dec, params[dec.phase], dimension, torch,
                    tactical_state_encoding=encoding,
                )
                old_values.append(float(value))
                lp = torch.nn.functional.log_softmax(logits / temperature, dim=0)
                old_logprobs.append(float(lp[dec.chosen_index]))
        combat_indices: set[int] = set()
        for segment, target in segments:
            indices = list(range(segment.start_decision, segment.end_decision))
            if not indices:
                continue
            if any(ep.decisions[i].phase != "combat" for i in indices):
                raise ValueError("A combat GAE segment contains strategic decisions")
            if any(i in combat_indices for i in indices):
                raise ValueError("Overlapping combat GAE segments")
            combat_indices.update(indices)
            adv, ret = gae_terminal(
                [old_values[i] for i in indices], target,
                gamma=gamma, lam=gae_lambda,
            )
            for i, advantage, value_target in zip(indices, adv, ret, strict=True):
                result["combat"].append(
                    PpoSample(ep.decisions[i], old_logprobs[i], advantage, value_target)
                )
        for i, dec in enumerate(ep.decisions):
            if dec.phase == "combat":
                if i not in combat_indices:
                    raise ValueError("A combat decision has no terminal combat record")
            elif dec.phase == "strategy":
                # Strategy keeps the certified run-level terminal target.
                # Only the combat learner uses local sparse-boundary GAE.
                result["strategy"].append(
                    PpoSample(
                        dec, old_logprobs[i],
                        run_target - old_values[i], run_target,
                    )
                )
            else:
                raise ValueError("Unknown decision phase")
    return result


def ppo_update(
    *,
    torch: Any,
    run_targets: list[tuple[Episode, float]],
    local_targets: list[list[tuple[Any, float]]],
    params: dict[str, dict[str, Any]],
    optimizers: dict[str, Any],
    dimension: int,
    encoding: str,
    temperature: float,
    gamma: float,
    gae_lambda: float,
    epochs: int,
    batch_size: int,
    sample_limit: int,
    clip_epsilon: float,
    value_weight: float,
    entropy_weight: float,
    seed: int,
) -> PpoDiagnostics:
    """Train only on frozen-cohort actions, with clipped on-policy ratios."""
    if min(epochs, batch_size, sample_limit) <= 0:
        raise ValueError("PPO epoch/batch/sample sizes must be positive")
    if not (0.0 < clip_epsilon < 1.0):
        raise ValueError("PPO clip epsilon must lie in (0,1)")
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError("PPO behavior temperature must be positive and finite")
    from sts2_ai.training.phase_split_selfplay import _forward_decision

    phase_data = _prepare_samples(
        torch=torch, run_targets=run_targets, local_targets=local_targets,
        params=params, dimension=dimension, encoding=encoding,
        temperature=temperature, gamma=gamma, gae_lambda=gae_lambda,
    )
    rng = random.Random(seed)
    all_losses: list[float] = []
    summary: dict[str, dict[str, float]] = {}
    steps = 0
    counts = {phase: len(samples) for phase, samples in phase_data.items()}
    for phase, all_samples in phase_data.items():
        if not all_samples:
            continue
        # A bounded on-policy working set prevents a very long combat from
        # monopolizing one round's gradient. Deterministic per-round sampling.
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
        # If all advantages are identical, the (correct) policy gradient is
        # zero; the critic can still learn. Never assign a fabricated sign.
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
                terms: list[Any] = []
                p_loss: list[Any] = []
                v_loss: list[Any] = []
                e_loss: list[Any] = []
                observed_kl: list[float] = []
                clipped = 0
                for idx in chosen:
                    sample = samples[idx]
                    value, logits = _forward_decision(
                        sample.decision, params[phase], dimension, torch,
                        tactical_state_encoding=encoding,
                    )
                    logps = torch.nn.functional.log_softmax(
                        logits / temperature, dim=0
                    )
                    logp = logps[sample.decision.chosen_index]
                    log_ratio = logp - sample.old_logprob
                    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                    adv = normalized[idx]
                    surrogate = torch.minimum(
                        ratio * adv,
                        torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv,
                    )
                    policy_loss = -surrogate
                    critic_loss = (value - sample.target).square()
                    entropy = -(logps.exp() * logps).sum()
                    p_loss.append(policy_loss)
                    v_loss.append(critic_loss)
                    e_loss.append(entropy)
                    terms.append(
                        policy_loss + value_weight * critic_loss
                        - entropy_weight * entropy
                    )
                    observed_kl.append(float(-log_ratio.detach()))
                    clipped += int(abs(float(ratio.detach()) - 1.0) > clip_epsilon)
                optimizers[phase].zero_grad()
                loss = torch.stack(terms).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(params[phase].values()), 1.0)
                optimizers[phase].step()
                steps += 1
                phase_steps += 1
                size = len(chosen)
                phase_seen += size
                phase_loss += float(loss.detach()) * size
                phase_policy += float(torch.stack(p_loss).mean().detach()) * size
                phase_value += float(torch.stack(v_loss).mean().detach()) * size
                phase_entropy += float(torch.stack(e_loss).mean().detach()) * size
                phase_kl += sum(observed_kl)
                phase_clip += clipped
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
