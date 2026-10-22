"""On-policy Nature PPO with a learned critic, sharing the player's PPO recipe.

Portable dependency-free rollout via ContextualDrawNature; PyTorch is used
only in the optimizer process. Samples are complete C# combats, with exact
Nature behavior distributions stored at each unresolved physical card draw.
"""
from __future__ import annotations

import importlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from sts2_ai.nature.contextual_draw import (
    CONTEXTUAL_SCHEMA,
    FEATURE_DIM,
    HIDDEN,
    ContextualDrawNature,
)
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode
from sts2_ai.training.phase_split_ppo import gae_terminal

PPO_DRAW_SCHEMA = "sts2-ai-nature-contextual-draw-ppo-v1"


@dataclass(slots=True)
class PpoDrawNature(ContextualDrawNature):
    """Clipped PPO and value regression on the *same* fair observation.

    The player remains frozen. Nature's objective is -player_score. Actor
    logits use the already replay-tested state/candidate network; the critic
    is a scalar head on the same state embedding. Action support is mixed
    with the declared reference draw distribution as in the existing model.
    """

    learning_rate: float = 0.001
    ppo_epochs: int = 4
    ppo_batch_size: int = 128
    ppo_sample_limit: int = 2048
    clip_epsilon: float = 0.2
    value_weight_factor: float = 0.5
    entropy_weight: float = 0.001
    gamma: float = 1.0
    gae_lambda: float = 0.98
    value_weight: list[float] = field(default_factory=list)
    value_bias: float = 0.0
    optimizer_moments: dict[str, dict[str, Any]] = field(default_factory=dict)
    optimizer_steps: int = 0
    policy_id: str = PPO_DRAW_SCHEMA

    def __post_init__(self) -> None:
        ContextualDrawNature.__post_init__(self)
        if not self.value_weight:
            self.value_weight = [0.0] * HIDDEN
        if (len(self.value_weight) != HIDDEN or
                any(not math.isfinite(v) for v in self.value_weight) or
                not math.isfinite(self.value_bias) or
                min(self.ppo_epochs, self.ppo_batch_size,
                    self.ppo_sample_limit) < 1 or
                not 0 < self.clip_epsilon < 1 or
                not 0 < self.gamma <= 1 or
                not 0 <= self.gae_lambda <= 1 or
                self.value_weight_factor < 0 or self.entropy_weight < 0 or
                self.optimizer_steps < 0):
            raise ValueError("Invalid Nature PPO actor-critic configuration")

    def _value(self, state_features: tuple[tuple[int, float], ...]) -> float:
        hidden = [
            math.tanh(sum(row[index] * val for index, val in state_features))
            for row in self.state_weight
        ]
        return -0.5 + 0.5 * math.tanh(
            math.fsum(v * h for v, h in zip(self.value_weight, hidden, strict=True))
            + self.value_bias
        )

    def propose(self, *args: Any, **kwargs: Any) -> tuple[Any, PilotChoice]:
        proposal, choice = ContextualDrawNature.propose(self, *args, **kwargs)
        return proposal, replace(choice, behavior_value=self._value(choice.state_features))

    def _parameters(self, torch: Any) -> dict[str, Any]:
        def param(value: Any) -> Any:
            return torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))

        result = {
            "state": param(self.state_weight),
            "value": param(self.value_weight),
            "value_bias": param(self.value_bias),
        }
        for key in sorted(self.candidate_vectors):
            result[f"candidate:{key}"] = param(self.candidate_vectors[key])
            result[f"bias:{key}"] = param(self.candidate_bias.get(key, 0.0))
        return result

    @staticmethod
    def _forward_torch(
        torch: Any, params: dict[str, Any],
        features: tuple[tuple[int, float], ...],
        cards: tuple[str, ...], baseline: tuple[float, ...],
        mix: float,
    ) -> tuple[Any, Any]:
        state = torch.zeros(FEATURE_DIM, dtype=torch.float64)
        for index, value in features:
            state[index] = value
        hidden = torch.tanh(torch.mv(params["state"], state))
        value = -0.5 + 0.5 * torch.tanh(
            torch.dot(params["value"], hidden) + params["value_bias"]
        )
        logits = torch.stack([
            torch.dot(params[f"candidate:{card}"], hidden)
            + params[f"bias:{card}"] for card in cards
        ])
        probs = torch.tensor(baseline, dtype=torch.float64)
        if not (len(cards) == len(baseline) and all(p > 0 for p in baseline)):
            raise ValueError("Nature PPO requires supported positive draw-law candidates")
        q = (1 - mix) * probs + mix * torch.softmax(logits, dim=0)
        return value, q

    def _load_optimizer(self, torch: Any, optimizer: Any, params: dict[str, Any]) -> None:
        for name, param in params.items():
            saved = self.optimizer_moments.get(name)
            if saved is None:
                continue
            for part in ("exp_avg", "exp_avg_sq"):
                vector = torch.tensor(saved[part], dtype=torch.float64)
                if vector.shape != param.shape or not torch.isfinite(vector).all():
                    raise ValueError("Stale/malformed Nature PPO Adam moments")
                optimizer.state[param][part] = vector
            count = saved["step"]
            if type(count) not in (int, float) or not math.isfinite(count) or count < 0:
                raise ValueError("Invalid Nature PPO Adam step")
            optimizer.state[param]["step"] = torch.tensor(float(count))

    def _save_optimizer(self, optimizer: Any, params: dict[str, Any]) -> None:
        result = {}
        for name, param in params.items():
            state = optimizer.state.get(param)
            if state:
                result[name] = {
                    "exp_avg": state["exp_avg"].detach().tolist(),
                    "exp_avg_sq": state["exp_avg_sq"].detach().tolist(),
                    "step": float(state["step"].item()),
                }
        self.optimizer_moments = result

    def update(self, episodes: tuple[PilotEpisode, ...]) -> dict[str, float]:
        """Multiple PPO minibatch updates, GAE, learned critic, AdamW.

        Old q and critic predictions come from the *behavior rollout*.
        The optimizer never simulates outcomes or accesses emulator hidden
        state; all actions in the update were chosen under a single revision.
        """
        if len(episodes) < 2:
            raise ValueError("Nature PPO requires multiple complete combats")
        samples = []
        total_reward = 0.0
        for ep in episodes:
            total_reward -= ep.player_return
            if any(choice.policy_revision != self.revision for choice in ep.choices):
                raise ValueError("Stale Nature PPO policy revision")
            if not ep.choices:
                continue
            old_values = []
            for choice in ep.choices:
                if (choice.operation != "draw_card" or not choice.state_features
                        or choice.behavior_value is None
                        or not math.isfinite(choice.behavior_value)):
                    raise ValueError("Incomplete on-policy Nature PPO decision")
                q, _, _ = self._forward(
                    choice.state_features, choice.features, choice.baseline
                )
                if any(abs(x - y) > 1e-9 for x, y
                       in zip(q, choice.probabilities, strict=True)):
                    raise ValueError("Nature PPO behavior q disagrees with current model")
                if abs(self._value(choice.state_features) - choice.behavior_value) > 1e-9:
                    raise ValueError("Stale Nature PPO critic provenance")
                old_values.append(choice.behavior_value)
            advantages, targets = gae_terminal(
                old_values, -ep.player_return,
                gamma=self.gamma, lam=self.gae_lambda,
            )
            for choice, advantage, target in zip(
                ep.choices, advantages, targets, strict=True
            ):
                samples.append((choice, advantage, target))
        if not samples:
            raise ValueError("No Nature decisions in completed combat cohort")

        rng = random.Random((self.seed + 1) * 1000003 + self.revision)
        if len(samples) > self.ppo_sample_limit:
            samples = [samples[i] for i in
                       sorted(rng.sample(range(len(samples)), self.ppo_sample_limit))]
        mu = math.fsum(row[1] for row in samples) / len(samples)
        std = math.sqrt(
            math.fsum((row[1] - mu) ** 2 for row in samples) / len(samples) + 1e-8
        )
        normalized = [max(-5.0, min(5.0, (row[1] - mu) / std))
                      for row in samples]
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("Nature PPO optimizer needs pip install -e '.[neural]'") from exc
        torch.set_num_threads(1)
        params = self._parameters(torch)
        optimizer = torch.optim.AdamW(
            list(params.values()), lr=self.learning_rate, weight_decay=1e-5
        )
        self._load_optimizer(torch, optimizer, params)
        seen = 0
        sum_policy = sum_value = sum_entropy = sum_kl = sum_clip = 0.0
        sum_gradient = 0.0
        optimizer_updates = 0
        for _ in range(self.ppo_epochs):
            order = list(range(len(samples)))
            rng.shuffle(order)
            for start in range(0, len(order), self.ppo_batch_size):
                indices = order[start:start + self.ppo_batch_size]
                if not indices:
                    continue
                losses = []
                for index in indices:
                    choice, _, target = samples[index]
                    critic, q = self._forward_torch(
                        torch, params, choice.state_features, choice.features,
                        choice.baseline, self.mix,
                    )
                    logprob = torch.log(q[choice.selected_index])
                    old_logprob = math.log(
                        choice.probabilities[choice.selected_index]
                    )
                    logratio = logprob - old_logprob
                    ratio = torch.exp(torch.clamp(logratio, -20.0, 20.0))
                    adv = normalized[index]
                    surrogate = torch.minimum(
                        ratio * adv,
                        torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * adv,
                    )
                    policy_loss = -surrogate
                    value_loss = (critic - target).square()
                    entropy = -(q * q.log()).sum()
                    losses.append(
                        policy_loss + self.value_weight_factor * value_loss
                        - self.entropy_weight * entropy
                    )
                    with torch.no_grad():
                        sum_policy += float(policy_loss)
                        sum_value += float(value_loss)
                        sum_entropy += float(entropy)
                        sum_kl += float(-logratio)
                        sum_clip += float(
                            abs(float(ratio) - 1.0) > self.clip_epsilon
                        )
                optimizer.zero_grad()
                loss = torch.stack(losses).mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    list(params.values()), 1.0
                )
                sum_gradient += float(norm)
                optimizer.step()
                optimizer_updates += 1
                seen += len(indices)
        self.state_weight = params["state"].detach().tolist()
        self.value_weight = params["value"].detach().tolist()
        self.value_bias = float(params["value_bias"].detach())
        for key in self.candidate_vectors:
            self.candidate_vectors[key] = params[f"candidate:{key}"].detach().tolist()
            self.candidate_bias[key] = float(params[f"bias:{key}"].detach())
        self._save_optimizer(optimizer, params)
        self.optimizer_steps += optimizer_updates
        self.revision += 1
        return {
            "nature_revision": float(self.revision),
            "mean_player_return": -total_reward / len(episodes),
            "nature_gradient_norm": sum_gradient / max(1, optimizer_updates),
            "nature_optimizer_steps": float(optimizer_updates),
            "nature_optimizer_steps_total": float(self.optimizer_steps),
            "nature_used_decisions": float(len(samples)),
            "nature_ppo_policy_loss": sum_policy / seen,
            "nature_ppo_value_mse": sum_value / seen,
            "nature_ppo_entropy": sum_entropy / seen,
            "nature_ppo_approx_kl": sum_kl / seen,
            "nature_ppo_clip_fraction": sum_clip / seen,
            "nature_choice_count": float(sum(len(ep.choices) for ep in episodes)),
        }

    def dump(self) -> dict[str, object]:
        saved = ContextualDrawNature.dump(self)
        saved["schema"] = PPO_DRAW_SCHEMA
        saved["policy_id"] = PPO_DRAW_SCHEMA
        saved.update({
            "ppo_epochs": self.ppo_epochs,
            "ppo_batch_size": self.ppo_batch_size,
            "ppo_sample_limit": self.ppo_sample_limit,
            "clip_epsilon": self.clip_epsilon,
            "value_weight_factor": self.value_weight_factor,
            "entropy_weight": self.entropy_weight,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "value_weight": list(self.value_weight),
            "value_bias": self.value_bias,
            "optimizer_moments": self.optimizer_moments,
            "optimizer_steps": self.optimizer_steps,
        })
        return saved

    @classmethod
    def restore(cls, data: dict[str, Any]) -> PpoDrawNature:
        if data.get("schema") != PPO_DRAW_SCHEMA:
            raise ValueError("Invalid contextual Nature PPO checkpoint")
        # The parent validates all shared actor tensors and candidate sizes.
        actor = ContextualDrawNature.restore({
            **data, "schema": CONTEXTUAL_SCHEMA
        })
        return cls(
            seed=actor.seed, mix=actor.mix,
            learning_rate=actor.learning_rate,
            score_clip=actor.score_clip, revision=actor.revision,
            state_weight=actor.state_weight,
            candidate_vectors=actor.candidate_vectors,
            candidate_bias=actor.candidate_bias,
            ppo_epochs=data["ppo_epochs"],
            ppo_batch_size=data["ppo_batch_size"],
            ppo_sample_limit=data["ppo_sample_limit"],
            clip_epsilon=data["clip_epsilon"],
            value_weight_factor=data["value_weight_factor"],
            entropy_weight=data["entropy_weight"],
            gamma=data["gamma"], gae_lambda=data["gae_lambda"],
            value_weight=list(data["value_weight"]),
            value_bias=data["value_bias"],
            optimizer_moments=data["optimizer_moments"],
            optimizer_steps=data["optimizer_steps"],
        )
