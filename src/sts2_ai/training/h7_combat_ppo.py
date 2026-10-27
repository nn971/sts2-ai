"""H7 PPO on source-fair, reference-law Silent combat trajectories.

Every update uses fresh actions sampled by exactly one frozen policy revision.
No distillation, Nature updates, offline importance weights or hidden state.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
from sts2_ai.models.shared_game_objects import EncodedDecision, encode_public_decision
from sts2_ai.models.shared_transformer import SharedObjectTransformer, collate_h7
from sts2_ai.nature.protocol import canonical_sha256

H7_PPO_SCHEMA = "sts2-h7-onpolicy-reference-combat-ppo-v1"


@dataclass(frozen=True)
class PreparedH7:
    encoded: EncodedDecision
    chosen: int
    old_logp: float
    old_value: float
    advantage: float
    target: float


class H7CombatPPO:
    """Small-cohort optimizer; actor evaluation is independent in CPU workers."""

    def __init__(
        self, model: SharedObjectTransformer, *, device: str = "cpu",
        lr: float = 0.0001, temperature: float = 0.85,
        epochs: int = 3, minibatch: int = 64, sample_limit: int = 4096,
        gamma: float = 1.0,
        gae_lambda: float = 0.98, clip: float = 0.2,
        entropy_coef: float = 0.001, value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ) -> None:
        if (lr <= 0 or temperature <= 0 or epochs < 1 or minibatch < 1
                or sample_limit < 2 or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1
                or not 0 < clip < 1 or entropy_coef < 0 or value_coef < 0
                or max_grad_norm <= 0):
            raise ValueError("Invalid H7 PPO optimizer configuration")
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs = epochs
        self.minibatch = minibatch
        self.sample_limit = sample_limit
        if model.config.dropout != 0.0:
            raise ValueError("H7 on-policy ratio verification requires dropout=0")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    @staticmethod
    def _prepare_episode(
        episode: Any, captured: tuple[dict[str, Any], ...],
        *, behavior_model_id: str, temperature: float,
        gamma: float, gae_lambda: float,
        allow_sampled_terminal_replay: bool = False,
    ) -> list[PreparedH7]:
        if (episode.outcome not in ("victory", "defeat")
                or type(episode.terminal_replay_verified) is not bool
                or (not allow_sampled_terminal_replay
                    and not episode.terminal_replay_verified)):
            raise ValueError(
                "Only verified terminal reference-law combats enter H7 PPO "
                "unless sampled audit is explicitly authorized"
            )
        # Unchecked episodes remain explicitly marked False in the episode
        # transcript. Sampled mode is permitted only by the source-audited H7
        # training collector, never by the strict default PPO interface.
        if episode.draw_mode != "reference":
            raise ValueError("H7 PPO must sample from the reference Chance Law")
        score = float(episode.score)
        if not math.isfinite(score) or not 0 <= score <= 2:
            raise ValueError("H4 combat score must be in [0,2]")
        # H4's victory=(1,2], defeat=[0,1); scale for tanh value in (-1,1).
        terminal_reward = score / 2.0
        steps = [x for x in episode.decisions if x.phase in ("player", "player_choice")]
        if not steps or len(steps) != len(captured):
            raise ValueError("H7 missing or reordered public action samples")
        entries: list[tuple[EncodedDecision, int, float, float]] = []
        for step, item in zip(steps, captured, strict=True):
            if (item.get("selected_id") != step.selected_id
                    or item.get("behavior_model_id") != behavior_model_id
                    or item.get("action_scope") != "all-legal-v1"
                    or item.get("temperature") != temperature
                    or canonical_sha256([
                        item.get("source_observation"),
                        item.get("source_legal_actions"),
                    ]) != item.get("source_support_identity")):
                raise ValueError("H7 captured behavior/source provenance mismatch")
            public, actions = FrozenP0Player._public_types(
                item["source_observation"], item["source_legal_actions"],
            )
            serialized = item["observation"]
            if (public.policy_id != serialized.get("policy_id")
                    or public.payload_json != serialized.get("payload_json")
                    or public.observation_hash != serialized.get("observation_hash")):
                raise ValueError("H7 policy frame differs from source-public projection")
            reported = item["legal_actions"]
            if len(actions) != len(reported) or any(
                a.action_id != b.get("action_id")
                or a.kind != b.get("kind")
                or a.payload_json != b.get("payload_json")
                for a, b in zip(actions, reported, strict=True)
            ):
                raise ValueError("H7 action support differs from source-public menu")
            idx = item["chosen_index"]
            if (type(idx) is not int or not 0 <= idx < len(actions)
                    or actions[idx].action_id != step.selected_id):
                raise ValueError("H7 selected action is outside the public menu")
            logp, value = item["behavior_logprob"], item["behavior_value"]
            if not isinstance(logp, (float, int)) or not math.isfinite(logp):
                raise ValueError("Invalid H7 behavior log probability")
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Invalid H7 behavior critic estimate")
            encoded = encode_public_decision(
                __import__("json").loads(public.payload_json), actions,
            )
            if encoded.phase != "combat":
                raise ValueError("H7 combat trainer received a macro decision")
            entries.append((encoded, idx, float(logp), float(value)))

        gae = 0.0
        prepared: list[PreparedH7] = []
        for i in range(len(entries) - 1, -1, -1):
            encoded, chosen, old_logp, old_value = entries[i]
            next_value = entries[i + 1][3] if i + 1 < len(entries) else 0.0
            reward = terminal_reward if i + 1 == len(entries) else 0.0
            delta = reward + gamma * next_value - old_value
            gae = delta + gamma * gae_lambda * gae
            prepared.append(PreparedH7(
                encoded, chosen, old_logp, old_value, gae, gae + old_value,
            ))
        return prepared[::-1]

    def update(
        self, episodes: tuple[Any, ...],
        captures: tuple[tuple[dict[str, Any], ...], ...],
        *, behavior_model_id: str, seed: int,
        allow_sampled_terminal_replay: bool = False,
    ) -> dict[str, float | int]:
        if len(episodes) != len(captures) or not episodes:
            raise ValueError("H7 update requires complete aligned on-policy episodes")
        prepared: list[PreparedH7] = []
        for ep, cap in zip(episodes, captures, strict=True):
            prepared.extend(self._prepare_episode(
                ep, cap, behavior_model_id=behavior_model_id,
                temperature=self.temperature, gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                allow_sampled_terminal_replay=allow_sampled_terminal_replay,
            ))
        if len(prepared) < 2:
            raise ValueError("H7 PPO requires at least two source-public decisions")
        total_valid_decisions = len(prepared)
        if len(prepared) > self.sample_limit:
            prepared = random.Random(seed).sample(prepared, self.sample_limit)

        # Recompute once before ANY mutation, proving that the captured frozen
        # actor corresponds to the exact trainable source policy revision.
        self.model.eval()
        with torch.no_grad():
            for begin in range(0, len(prepared), self.minibatch):
                chunk = prepared[begin:begin + self.minibatch]
                batch = collate_h7([p.encoded for p in chunk]).to(self.device)
                logits, values = self.model(batch)
                selected = torch.tensor(
                    [p.chosen for p in chunk], device=self.device,
                ).unsqueeze(1)
                logps = torch.log_softmax(logits.float() / self.temperature, dim=1)
                current = logps.gather(1, selected).squeeze(1).cpu().tolist()
                predicted = values.float().cpu().tolist()
                for p, observed, value in zip(chunk, current, predicted, strict=True):
                    if (abs(p.old_logp - observed) > 5e-4
                            or abs(p.old_value - value) > 5e-4):
                        raise ValueError("H7 behavior policy mismatch before PPO update")

        advantages = torch.tensor([p.advantage for p in prepared], dtype=torch.float32)
        normalized = (advantages - advantages.mean()) / advantages.std(
            unbiased=False,
        ).clamp(min=1e-6)
        rng = random.Random(seed)
        diagnostics: list[tuple[float, float, float, float, float]] = []
        for _ in range(self.epochs):
            order = list(range(len(prepared)))
            rng.shuffle(order)
            for begin in range(0, len(order), self.minibatch):
                indices = order[begin:begin + self.minibatch]
                samples = [prepared[i] for i in indices]
                batch = collate_h7([p.encoded for p in samples]).to(self.device)
                old_logp = torch.tensor(
                    [p.old_logp for p in samples], device=self.device,
                )
                target = torch.tensor(
                    [p.target for p in samples], device=self.device,
                )
                adv = normalized[indices].to(self.device)
                chosen = torch.tensor(
                    [p.chosen for p in samples], device=self.device,
                ).unsqueeze(1)
                self.model.train()
                logits, values = self.model(batch)
                logps = torch.log_softmax(logits.float() / self.temperature, dim=1)
                selected_logp = logps.gather(1, chosen).squeeze(1)
                ratio = torch.exp(selected_logp - old_logp)
                surrogate = torch.minimum(
                    ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv,
                )
                valid_probs = torch.where(
                    batch.action_mask, torch.exp(logps), torch.zeros_like(logps),
                )
                entropy = -(valid_probs * torch.where(
                    batch.action_mask, logps, torch.zeros_like(logps),
                )).sum(dim=1).mean()
                value_loss = (values.float() - target).square().mean()
                loss = (
                    -surrogate.mean() + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite H7 PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                with torch.no_grad():
                    diagnostics.append((
                        float((old_logp - selected_logp).mean()),
                        float(((ratio - 1).abs() > self.clip).float().mean()),
                        float(entropy), float(value_loss), float(grad),
                    ))
        self.model.eval()
        return {
            "decisions": len(prepared),
            "source_decisions": total_valid_decisions,
            "updates": len(diagnostics),
            "approx_kl": sum(x[0] for x in diagnostics) / len(diagnostics),
            "clip_fraction": sum(x[1] for x in diagnostics) / len(diagnostics),
            "entropy": sum(x[2] for x in diagnostics) / len(diagnostics),
            "value_mse": sum(x[3] for x in diagnostics) / len(diagnostics),
            "grad_norm": sum(x[4] for x in diagnostics) / len(diagnostics),
        }
