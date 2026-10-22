"""Warm-started v8 player PPO on exactly the actions collected against Nature.

The frozen exported v8 policy provides weights and a public observation/action
encoder. The optimizer trains the combat head only; the strategy head remains
byte-identical. No hidden game state enters training samples.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
from sts2_ai.models.neural import (
    TACTICAL_ATTENTION_FORMAT,
    TACTICAL_INSTANCES_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.phase_split_ppo_batch import ppo_update_batched
from sts2_ai.training.selfplay import Episode, PublicDecision


class JointPlayerPPO:
    """A source-v8 batched combat PPO learner with persistent Adam state."""

    def __init__(
        self, model: PhaseSplitNeuralModel, *, torch: Any,
        learning_rate: float = 0.0003, temperature: float = 0.85,
        ppo_epochs: int = 3, batch_size: int = 128,
        sample_limit: int = 2048, clip_epsilon: float = 0.2,
        gamma: float = 1.0, gae_lambda: float = 0.98,
        value_weight: float = 0.5, entropy_weight: float = 0.001,
    ) -> None:
        if model.combat.format_id not in (
            TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT,
        ):
            raise ValueError("Joint PPO requires a v8/v13 instance-aware player")
        if not (0 < temperature and 0 < learning_rate < 1):
            raise ValueError("Invalid player temperature or learning rate")
        self.model = model
        self.torch = torch
        self.lr = learning_rate
        self.temperature = temperature
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.encoding = (
            "enemy_attention" if model.combat.format_id ==
            TACTICAL_ATTENTION_FORMAT else "enemy_instances"
        )
        actor = model.combat
        names = (
            "state_weight", "state_bias", "action_weight", "action_bias",
            "policy_weight", "policy_bias", "value_weight", "value_bias",
            "enemy_weight", "enemy_bias", "enemy_context_weight",
            "enemy_target_weight",
        ) + ((
            "attn_query", "attn_key", "attn_value", "attn_output", "attn_gate"
        ) if self.encoding == "enemy_attention" else ())
        self.params = {
            name: torch.tensor(
                getattr(actor, name), dtype=torch.float32,
                requires_grad=True,
            )
            for name in names
        }
        self.optimizer = torch.optim.AdamW(list(self.params.values()), lr=learning_rate)
        self.revision = 0

    def dump(self) -> dict[str, Any]:
        return {
            "params": {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.params.items()
            },
            "optimizer": self.optimizer.state_dict(),
            "revision": self.revision,
            "model": self.model.to_dict(),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        with self.torch.no_grad():
            if set(payload["params"]) != set(self.params):
                raise ValueError("Player joint-PPO parameter set changed")
            for key, tensor in self.params.items():
                source = payload["params"][key]
                if source.shape != tensor.shape or not self.torch.isfinite(source).all():
                    raise ValueError("Invalid joint player checkpoint tensor")
                tensor.copy_(source)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.revision = payload["revision"]
        self.model = PhaseSplitNeuralModel.from_dict(payload["model"])
        if self.revision < 0 or self.encoding != (
            "enemy_attention" if self.model.combat.format_id ==
            TACTICAL_ATTENTION_FORMAT else "enemy_instances"
        ):
            raise ValueError("Joint player checkpoint architecture mismatch")

    def export(self) -> PhaseSplitNeuralModel:
        raw = self.model.combat.to_dict()
        for name, tensor in self.params.items():
            raw[name] = tensor.detach().cpu().tolist()
        raw["model_id"] = f"h3-player-combat-r{self.revision}"
        raw["value_head_trained"] = True
        self.model = PhaseSplitNeuralModel(
            strategy=self.model.strategy,
            combat=NeuralPolicyValueModel.from_dict(raw),
            model_id=f"h3-player-joint-r{self.revision}",
            combat_value_objective=self.model.combat_value_objective,
        )
        return self.model

    @staticmethod
    def validate_capture(
        episode: Any, records: tuple[dict[str, Any], ...],
        model: PhaseSplitNeuralModel,
    ) -> tuple[PublicDecision, ...]:
        """Fail closed on lost/reshuffled menus and old-policy provenance."""
        steps = tuple(
            step for step in episode.decisions
            if step.phase in ("player", "player_choice")
        )
        if len(records) != len(steps) or not records:
            raise ValueError("Missing or reordered player on-policy decisions")
        decisions = []
        for step, row in zip(steps, records, strict=True):
            if row.get("selected_id") != step.selected_id:
                raise ValueError("Player action and source replay disagree")
            if canonical_sha256([
                row["source_observation"], row["source_legal_actions"]
            ]) != row["source_support_identity"]:
                raise ValueError("Player source public support fingerprint differs")
            obs = row["observation"]
            observation = Observation(
                obs["policy_id"], obs["payload_json"], obs["observation_hash"]
            )
            legal = tuple(
                LegalAction(a["action_id"], a["kind"], a["payload_json"])
                for a in row["legal_actions"]
            )
            from_source, all_source_actions = FrozenP0Player._public_types(
                row["source_observation"], row["source_legal_actions"],
                need_legacy_instance_slot=True,
            )
            scoped_source = FrozenP0Player.scoped_actions(
                row["source_observation"], all_source_actions, "all-legal-v1",
            )
            if from_source != observation or scoped_source != legal:
                raise ValueError("Player capture differs from fair source menu")
            idx = row["chosen_index"]
            if (not 0 <= idx < len(legal) or
                    legal[idx].action_id != step.selected_id or
                    row["behavior_model_id"] != model.model_id or
                    not math.isfinite(row["behavior_logprob"]) or
                    not math.isfinite(row["behavior_value"])):
                raise ValueError("Invalid on-policy player behavior record")
            if canonical_sha256(__import__("json").loads(
                observation.payload_json
            )) != observation.observation_hash:
                raise ValueError("Tampered public observation hash")
            estimate = model.evaluate(observation, legal)
            logits = estimate.action_logits
            temperature = row["temperature"]
            scaled = [float(logit) / temperature for logit in logits]
            peak = max(scaled)
            expected_logp = scaled[idx] - peak - math.log(
                math.fsum(math.exp(v - peak) for v in scaled)
            )
            if (abs(expected_logp - row["behavior_logprob"]) > 1e-7
                    or abs(estimate.value - row["behavior_value"]) > 1e-7):
                raise ValueError("Player behavior differs from frozen PPO policy")
            decisions.append(
                PublicDecision(observation, legal, idx, phase="combat")
            )
        return tuple(decisions)

    def update(
        self, samples: tuple[tuple[Any, tuple[dict[str, Any], ...]], ...],
        *, seed: int,
    ) -> dict[str, float]:
        # Validate every trace BEFORE the first gradient update. Unlike
        # offline replay, all behavior records are from the same policy version.
        targets = []
        segments = []
        for episode, captured in samples:
            if any(abs(row["temperature"] - self.temperature) > 1e-12
                   or row["action_scope"] != "all-legal-v1"
                   for row in captured):
                raise ValueError("Player PPO behavior temperature/scope mismatch")
            choices = self.validate_capture(episode, captured, self.model)
            targets.append((
                Episode(
                    seed=episode.source_seed, decisions=choices,
                    outcome=episode.outcome, act=1, floor=2,
                    hp_ratio=max(0., episode.exit_hp / 70),
                    completed=True,
                ),
                episode.score,
            ))
            segments.append([(
                SimpleNamespace(start_decision=0, end_decision=len(choices)),
                episode.score,
            )])
        if len(targets) < 2:
            raise ValueError("Joint player PPO requires two completed combats")
        result = ppo_update_batched(
            torch=self.torch, run_targets=targets,
            local_targets=segments,
            params={"combat": self.params}, optimizers={"combat": self.optimizer},
            dimension=self.model.combat.dimension, encoding=self.encoding,
            temperature=self.temperature, gamma=self.gamma,
            gae_lambda=self.gae_lambda, epochs=self.ppo_epochs,
            batch_size=self.batch_size, sample_limit=self.sample_limit,
            clip_epsilon=self.clip_epsilon,
            value_weight=self.value_weight,
            entropy_weight=self.entropy_weight, seed=seed,
        )
        self.revision += 1
        self.export()
        summary = result.phase.get("combat", {})
        return {
            "player_ppo_steps": float(result.steps),
            "player_ppo_samples": float(result.counts["combat"]),
            "player_ppo_kl": summary.get("approx_kl", 0.),
            "player_ppo_clip": summary.get("clip_fraction", 0.),
            "player_value_mse": summary.get("value_mse", 0.),
        }
