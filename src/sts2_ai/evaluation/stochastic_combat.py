"""Per-combat seeded stochastic tactical agent, using public logits only.

Each snapshot/replicate gets an independent agent and the same uniform-random
stream for both compared policies. Emulator hidden RNG is restored from the
same exact entry separately for each model. This is common-random-numbers
pairing, not counterfactual reuse of actions or RNG after the paths diverge.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe
from sts2_ai.models.phase_split import PhaseSplitNeuralModel


def snapshot_actor_seed(recipe: CombatSnapshotRecipe, replicate: int) -> int:
    if replicate < 0:
        raise ValueError("Replicate must be nonnegative")
    value = json.dumps(
        [recipe.seed, recipe.ascension, recipe.source_policy,
         recipe.exact_hash, replicate],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


class StochasticCombatAgent:
    def __init__(
        self, model: PhaseSplitNeuralModel, *,
        seed: int, temperature: float,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Stochastic combat temperature must be positive")
        self.model = model
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.policy_id = "phase-split-stochastic-v15p-" + model.model_id

    def choose(
        self, observation: Observation, legal_actions: Sequence[LegalAction],
    ) -> Decision:
        if not legal_actions:
            raise ValueError("No legal actions")
        logits = self.model.evaluate(observation, legal_actions).action_logits
        if len(logits) != len(legal_actions) or any(
            not math.isfinite(x) for x in logits
        ):
            raise ValueError("Stochastic policy emitted invalid logits")
        if len(legal_actions) == 1:
            return Decision(legal_actions[0], policy_name=self.policy_id)
        center = max(logits)
        masses = [math.exp((v - center) / self.temperature) for v in logits]
        threshold = self.rng.random() * math.fsum(masses)
        for action, mass in zip(legal_actions, masses, strict=True):
            threshold -= mass
            if threshold < 0:
                return Decision(action, policy_name=self.policy_id)
        return Decision(legal_actions[-1], policy_name=self.policy_id)
