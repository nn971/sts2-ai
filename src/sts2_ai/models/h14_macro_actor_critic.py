"""H14 macro policy/continuation-value network.

The actor and critic share one public-state Transformer. The frozen H9 combat
policy lives in a separate model/agent and executes the combat option. No
combat outcome predictor, H12 reward target, or heuristic mode is imported.

PyTorch is the explicit optional `neural` dependency.
"""
# mypy: disable-error-code="import-not-found,misc,no-any-return"
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .shared_game_objects import SHARED_OBJECT_SCHEMA, EncodedDecision, encode_public_decision
from .shared_transformer import H7Config, SharedObjectTransformer, collate_h7

H14_MACRO_MODEL_FORMAT = "sts2-h14-macro-policy-value-v1"


@dataclass(frozen=True)
class H14MacroScores:
    """One public macro decision, aligned to the supplied legal action order."""

    action_logits: tuple[float, ...]
    continuation_value: float


class H14MacroActorCritic(nn.Module):
    """Independent macro network; the paired combat model remains frozen.

    Reuses the proven H7 object attention and policy/value heads, selecting
    exclusively the macro heads (index 1). It trains its own representation,
    independent of an incumbent H9 tactical checkpoint.
    """

    def __init__(self, config: H7Config | None = None) -> None:
        super().__init__()
        self.config = config or H7Config()
        self.transformer = SharedObjectTransformer(self.config)

    def forward(self, decisions: list[EncodedDecision]) -> tuple[Tensor, Tensor]:
        if not decisions or any(decision.phase != "macro" for decision in decisions):
            raise ValueError("H14 macro actor/critic requires only macro decisions")
        batch = collate_h7(decisions).to(next(self.parameters()).device)
        return self.transformer(batch, known_phase=1)

    def evaluate(self, public_state: dict[str, Any],
                 legal_actions: Sequence[Any]) -> H14MacroScores:
        """Evaluate a fair public macro state; no exact state or RNG input."""
        encoded = encode_public_decision(public_state, legal_actions)
        if encoded.phase != "macro":
            raise ValueError("H14 macro policy called in combat")
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits, values = self.forward([encoded])
            return H14MacroScores(
                tuple(float(x) for x in logits[0].cpu().tolist()),
                float(values[0].cpu().item()),
            )
        finally:
            self.train(was_training)

    def policy_value_loss(
        self, decisions: list[EncodedDecision], *,
        returns: Tensor,
        selected_actions: Tensor | None = None,
        search_probabilities: Tensor | None = None,
        value_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Train on *measured* run returns and optional legal search policies.

        Exactly one actor supervision source is required. Search distributions
        must be normalized over each legal menu; stochastic outcomes are
        averaged by search, never optimized as desired policy actions.
        """
        if (selected_actions is None) == (search_probabilities is None):
            raise ValueError("Specify either executed action or search policy target")
        if value_weight < 0:
            raise ValueError("Invalid value weight")
        logits, values = self.forward(decisions)
        returns = returns.to(device=values.device, dtype=values.dtype)
        if returns.shape != values.shape or not torch.isfinite(returns).all():
            raise ValueError("Measured returns must match batch shape and be finite")
        if (returns.abs() > 1.0).any():
            raise ValueError("H14 return targets must be scaled into [-1, 1]")
        masks = collate_h7(decisions).action_mask.to(logits.device)
        logp = F.log_softmax(logits, dim=-1)
        if selected_actions is not None:
            chosen = selected_actions.to(device=logits.device, dtype=torch.long)
            if (chosen.shape != values.shape
                    or (chosen < 0).any()
                    or (chosen >= masks.shape[1]).any()
                    or not masks.gather(1, chosen[:, None]).all()):
                raise ValueError("Selected macro action is outside public legal menu")
            policy_loss = -logp.gather(1, chosen[:, None]).mean()
        else:
            assert search_probabilities is not None
            probs = search_probabilities.to(device=logits.device, dtype=values.dtype)
            if (probs.shape != logp.shape or not torch.isfinite(probs).all()
                    or (probs < 0).any() or (probs[~masks] != 0).any()
                    or not torch.allclose(
                        probs.sum(dim=1), torch.ones_like(values), atol=1e-5, rtol=0)):
                raise ValueError("Search policy must be normalized over legal moves")
            policy_loss = -(probs * logp).sum(dim=-1).mean()
        value_loss = F.mse_loss(values, returns)
        total = policy_loss + value_weight * value_loss
        return total, {
            "policy": float(policy_loss.detach().cpu()),
            "value": float(value_loss.detach().cpu()),
        }

    def ppo_loss(
        self, decisions: list[EncodedDecision], *,
        selected_actions: Tensor, behavior_log_probabilities: Tensor,
        behavior_values: Tensor, returns: Tensor, clip_epsilon: float = 0.2,
        value_weight: float = 0.5, entropy_weight: float = 0.01,
        behavior_temperature: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """On-policy macro PPO using certified run outcomes as value targets.

        Behavior logits/values come from a frozen round checkpoint. A run
        contributes only after its Act-1 clear/defeat label is established.
        """
        if (not 0 < clip_epsilon < 1 or value_weight < 0
                or entropy_weight < 0 or behavior_temperature <= 0):
            raise ValueError("Invalid H14 PPO coefficients")
        logits, values = self.forward(decisions)
        device, dtype = values.device, values.dtype
        chosen = selected_actions.to(device=device, dtype=torch.long)
        old_logp = behavior_log_probabilities.to(device=device, dtype=dtype)
        old_v = behavior_values.to(device=device, dtype=dtype)
        targets = returns.to(device=device, dtype=dtype)
        if any(x.shape != values.shape for x in (chosen, old_logp, old_v, targets)):
            raise ValueError("H14 PPO batch shapes disagree")
        if (not torch.isfinite(old_logp).all() or not torch.isfinite(old_v).all()
                or not torch.isfinite(targets).all() or (targets.abs() > 1).any()):
            raise ValueError("H14 PPO requires finite certified returns")
        masks = collate_h7(decisions).action_mask.to(device)
        if ((chosen < 0).any() or (chosen >= masks.shape[1]).any()
                or not masks.gather(1, chosen[:, None]).all()):
            raise ValueError("Invalid selected macro action")
        logp = F.log_softmax(logits / behavior_temperature, dim=-1)
        selected_logp = logp.gather(1, chosen[:, None]).squeeze(1)
        advantage = (targets - old_v).detach()
        ratio = torch.exp(selected_logp - old_logp)
        policy_loss = -torch.minimum(
            ratio * advantage,
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage,
        ).mean()
        value_loss = F.mse_loss(values, targets)
        probabilities = logp.exp()
        entropy = -(probabilities * torch.where(masks, logp, 0.0)).sum(dim=-1).mean()
        loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
        return loss, {
            "policy": float(policy_loss.detach().cpu()),
            "value": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    def save_checkpoint(self, path: Path, *, provenance: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": H14_MACRO_MODEL_FORMAT,
            "object_schema": SHARED_OBJECT_SCHEMA,
            "config": asdict(self.config),
            "provenance": provenance,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path, *, map_location: Any = "cpu",
    ) -> H14MacroActorCritic:
        data = torch.load(path, map_location=map_location, weights_only=True)
        if (data.get("format") != H14_MACRO_MODEL_FORMAT
                or data.get("object_schema") != SHARED_OBJECT_SCHEMA):
            raise ValueError("Unsupported H14 macro model/codec checkpoint")
        model = cls(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model
