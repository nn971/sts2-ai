"""H26 search-free, on-policy combat actor–critic with macro continuation.

The combat model has independent weights. The macro model provides the
post-combat Act-clear estimate only after an observed combat victory. No PUCT
or auxiliary future-HP regression is used.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from sts2_ai.training.h16_episode_metrics import combat_results
from sts2_ai.training.h16_joint_act_pilot import (
    PilotDecision,
    PilotEpisode,
    public_state,
)
from sts2_ai.training.h26_tactical import combat_dense_rewards


@dataclass(frozen=True)
class CombatReturnSample:
    decision: PilotDecision
    target: float
    fight_won: bool


def _macro_act_value(
    model: Any, episode: PilotEpisode, boundary_index: int,
) -> float:
    if boundary_index == len(episode.decisions):
        return float(episode.outcome == "act_clear")
    decision = episode.decisions[boundary_index]
    state = public_state(decision.observation)
    if isinstance(state.get("combat"), dict):
        # An Act-clear boundary can still retain a combat observation.
        if episode.outcome == "act_clear":
            return 1.0
        raise ValueError("Macro continuation requires post-combat observation")
    frame = model.encode_decision(state, decision.legal_actions)
    batch = model.collate([frame]).to(next(model.parameters()).device)
    with torch.inference_mode():
        _, act, _ = model(batch)
    return float(act[0].clamp(0, 1))


def combat_return_samples(
    episode: PilotEpisode, macro_model: Any, *,
    gamma: float = 0.99,
    combat_credit_weight: float = 0.25,
    defeat_progress_cap: float = 0.2,
) -> list[CombatReturnSample]:
    """Monte Carlo returns for all completed fights of one full-Act episode."""
    if not episode.complete:
        return []
    if (not math.isfinite(gamma) or not 0 < gamma <= 1
            or not 0 <= combat_credit_weight <= 1
            or not 0 <= defeat_progress_cap < 1):
        raise ValueError("Invalid H26 combat target configuration")
    samples: list[CombatReturnSample] = []
    for fight in combat_results(episode.frames, episode.outcome):
        a, b = fight.start, fight.end
        if b <= a:
            continue
        segment = list(episode.frames[a:b + 1])
        # At a terminal boundary, potential is zero so shaping telescopes.
        segment[-1] = {**segment[-1], "combat": None}
        shaped = combat_dense_rewards(segment, gamma=gamma)
        if fight.won:
            macro_value = _macro_act_value(macro_model, episode, b)
            terminal = (
                combat_credit_weight + (1-combat_credit_weight)*macro_value
            )
        else:
            terminal = (
                combat_credit_weight * defeat_progress_cap * fight.enemy_progress
            )
        running = terminal
        for index in reversed(range(a, b)):
            running = shaped[index-a] + gamma * running
            samples.append(CombatReturnSample(
                episode.decisions[index], running, fight.won,
            ))
    return samples


def fit_search_free_combat(
    combat_model: Any, macro_model: Any, optimizer: torch.optim.Optimizer,
    episodes: Sequence[PilotEpisode], *,
    gamma: float = 0.99,
    minibatch: int = 16,
    epochs: int = 1,
    temperature: float = 0.85,
    entropy_weight: float = 0.001,
    value_weight: float = 0.5,
    max_samples: int = 4096,
    seed: int = 26001,
) -> dict[str, float | int]:
    """Search-free combat policy gradient with a bounded H24 monotone critic."""
    if (minibatch <= 0 or epochs <= 0 or max_samples <= 0
            or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(entropy_weight) or entropy_weight < 0
            or not math.isfinite(value_weight) or value_weight < 0):
        raise ValueError("Invalid H26 combat update configuration")
    macro_model.eval()
    all_samples = [
        sample for episode in episodes
        for sample in combat_return_samples(episode, macro_model, gamma=gamma)
    ]
    if not all_samples:
        return {"examples": 0, "steps": 0, "loss": 0.0}
    rng = random.Random(seed)
    if len(all_samples) > max_samples:
        all_samples = rng.sample(all_samples, max_samples)
    device = next(combat_model.parameters()).device
    combat_model.train()
    losses = actor_losses = value_losses = 0.0
    steps = 0
    for _ in range(epochs):
        rng.shuffle(all_samples)
        for start in range(0, len(all_samples), minibatch):
            chunk = all_samples[start:start + minibatch]
            encoded = [
                combat_model.encode_decision(
                    public_state(sample.decision.observation),
                    sample.decision.legal_actions,
                    intent_histories=sample.decision.h26_public_intent_history,
                ) for sample in chunk
            ]
            batch = combat_model.collate(encoded).to(device)
            logits, critic_logit, _ = combat_model(batch, value_logits=True)
            logits = (logits / temperature).masked_fill(
                ~batch.action_mask, -1e4,
            )
            logp = F.log_softmax(logits, dim=-1)
            target = torch.tensor(
                [x.target for x in chunk], dtype=critic_logit.dtype,
                device=device,
            )
            baseline = torch.sigmoid(critic_logit)
            chosen = torch.tensor(
                [x.decision.chosen_index for x in chunk],
                device=device, dtype=torch.long,
            )
            actor_loss = -(
                (target-baseline.detach())
                * logp.gather(-1, chosen[:, None]).squeeze(-1)
            ).mean()
            value_loss = F.mse_loss(baseline, target.clamp(0, 1))
            entropy = -(logp.exp() * logp).masked_fill(
                ~batch.action_mask, 0,
            ).sum(-1).mean()
            loss = actor_loss + value_weight * value_loss - entropy_weight * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite H26 combat actor-critic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                combat_model.parameters(), 1.0, error_if_nonfinite=True,
            )
            optimizer.step()
            losses += float(loss.detach())
            actor_losses += float(actor_loss.detach())
            value_losses += float(value_loss.detach())
            steps += 1
    combat_model.eval()
    return {
        "examples": len(all_samples), "steps": steps, "loss": losses / steps,
        "actor_loss": actor_losses / steps,
        "value_loss": value_losses / steps,
        "combat_win_samples": sum(x.fight_won for x in all_samples),
    }
