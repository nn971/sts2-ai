"""Train the strategy head while preserving a portable combat backbone.

The whole-run PPO cohort includes real tactical and macro actions, but the
learner projects each completed episode onto its *strategy decisions only*.
This keeps PPO's old-policy ratios valid while never encoding or differentiating
through combat actions. Combat's serialized weights stay literally unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.neural import _export
from sts2_ai.training.phase_split_ppo_batch import ppo_update_batched
from sts2_ai.training.selfplay import Episode

H6_SCHEMA = "sts2-h6-frozen-combat-macro-ppo-v1"
STRATEGY_PARAMETERS = (
    "state_weight",
    "state_bias",
    "action_weight",
    "action_bias",
    "policy_weight",
    "policy_bias",
    "value_weight",
    "value_bias",
)


def combat_digest(model: PhaseSplitNeuralModel) -> str:
    return hashlib.sha256(
        json.dumps(model.combat.to_dict(), sort_keys=True).encode()
    ).hexdigest()


def macro_only_episode(episode: Episode) -> Episode:
    """Remove tactical actions for optimization, not from the actual run."""
    return replace(
        episode,
        decisions=tuple(
            decision for decision in episode.decisions
            if decision.phase == "strategy"
        ),
        combat_outcomes=(),
    )


class FrozenCombatMacroPPO:
    def __init__(
        self, initial: PhaseSplitNeuralModel, *, torch: Any,
        learning_rate: float = 0.0003, temperature: float = 0.85,
        epochs: int = 3, batch_size: int = 128,
        sample_limit: int = 4096,
    ) -> None:
        if (initial.strategy.format_id !=
                "sts2-neural-policy-value-v2-semantic-action"):
            raise ValueError("H6 requires semantic strategy feature format")
        if (learning_rate <= 0 or learning_rate >= 1
                or temperature <= 0
                or min(epochs, batch_size, sample_limit) < 1):
            raise ValueError("Invalid H6 PPO hyperparameters")
        self.initial = initial
        self.combat = initial.combat
        self.frozen_sha256 = combat_digest(initial)
        self.dimension = initial.strategy.dimension
        self.hidden = initial.strategy.hidden
        self.torch = torch
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        self.params = {
            name: torch.tensor(
                getattr(initial.strategy, name),
                dtype=torch.float32,
                requires_grad=True,
            )
            for name in STRATEGY_PARAMETERS
        }
        self.optimizer = torch.optim.AdamW(
            list(self.params.values()), lr=learning_rate,
        )
        self.revision = 0

    def model(self) -> PhaseSplitNeuralModel:
        strategy: NeuralPolicyValueModel = _export(
            self.params, self.dimension, self.hidden,
            model_id=f"h6-strategy-r{self.revision:04d}",
            value_head_trained=self.revision > 0
            or self.initial.strategy.value_head_trained,
        )
        result = PhaseSplitNeuralModel(
            strategy=strategy,
            combat=self.combat,
            model_id=f"h6-macro-frozen-combat-r{self.revision:04d}",
            combat_value_objective=self.initial.combat_value_objective,
        )
        self.assert_frozen(result)
        return result

    def assert_frozen(self, candidate: PhaseSplitNeuralModel) -> None:
        if combat_digest(candidate) != self.frozen_sha256:
            raise RuntimeError("H6 combat backbone was modified")

    def dump(self) -> dict[str, Any]:
        return {
            "schema": H6_SCHEMA,
            "revision": self.revision,
            "frozen_combat_sha256": self.frozen_sha256,
            "strategy_params": {
                name: value.detach().cpu().clone()
                for name, value in self.params.items()
            },
            "strategy_optimizer": self.optimizer.state_dict(),
        }

    def restore(self, saved: dict[str, Any]) -> None:
        if (saved["schema"] != H6_SCHEMA
                or saved["frozen_combat_sha256"] != self.frozen_sha256
                or set(saved["strategy_params"]) != set(self.params)):
            raise ValueError("H6 checkpoint architecture/backbone mismatch")
        with self.torch.no_grad():
            for name, target in self.params.items():
                old = saved["strategy_params"][name]
                if old.shape != target.shape or not self.torch.isfinite(old).all():
                    raise ValueError("Non-finite or incompatible strategy checkpoint")
                target.copy_(old)
        self.optimizer.load_state_dict(saved["strategy_optimizer"])
        self.revision = int(saved["revision"])
        if self.revision < 0:
            raise ValueError("Invalid H6 revision")
        self.assert_frozen(self.model())

    def update(
        self, completed: list[tuple[Episode, float]], *, seed: int,
    ) -> dict[str, Any]:
        """One PPO revision; truncated runs have no fabricated final reward."""
        if any(not episode.completed for episode, _ in completed):
            raise ValueError("H6 macro optimizer received censored episode")
        projected = [
            (macro_only_episode(ep), ret)
            for ep, ret in completed
            if any(step.phase == "strategy" for step in ep.decisions)
        ]
        if not projected:
            # Still account for the revision: no gradients were computed.
            self.revision += 1
            self.assert_frozen(self.model())
            return {"ppo_steps": 0, "strategy_decisions": 0,
                    "combat_optimizer_steps": 0, "ppo_phase": {}}
        result = ppo_update_batched(
            torch=self.torch,
            run_targets=projected,
            local_targets=[[] for _ in projected],
            params={"strategy": self.params},
            optimizers={"strategy": self.optimizer},
            dimension=self.dimension,
            encoding="legacy",
            temperature=self.temperature,
            gamma=1.0, gae_lambda=1.0,
            epochs=self.epochs, batch_size=self.batch_size,
            sample_limit=self.sample_limit, clip_epsilon=0.2,
            value_weight=0.5, entropy_weight=0.001,
            seed=seed,
        )
        if result.counts["combat"] != 0:
            raise RuntimeError("H6 accidentally trained combat decisions")
        self.revision += 1
        self.assert_frozen(self.model())
        return {
            "ppo_steps": result.steps,
            "strategy_decisions": result.counts["strategy"],
            "combat_optimizer_steps": result.counts["combat"],
            "ppo_phase": result.phase.get("strategy", {}),
        }
