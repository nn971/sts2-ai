"""Experimental Nature opponent for just two semantic combat chance sites.

This is an *adversarial stress-training* proposal, not original-game
importance sampling. It controls only source-authored `enemy_intent` and
`draw_card` events. Unverified native probability laws remain unverified.
The emulator alone owns intent legality, card instances, deck constraints,
hook ordering, and terminal transitions.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .semantic import SemanticEvent, SemanticProposal

PILOT_SCHEMA = "sts2-ai-nature-two-site-opponent-v1"
PILOT_OPERATIONS = frozenset({"enemy_intent", "draw_card"})


def _rng(seed: int, *values: object) -> random.Random:
    material = json.dumps([seed, *values], sort_keys=True).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


@dataclass(frozen=True, slots=True)
class PilotChoice:
    """One original rollout's exact q and chosen feature identities."""

    operation: str
    context_key: str
    features: tuple[str, ...]
    baseline: tuple[float, ...]
    probabilities: tuple[float, ...]
    selected_index: int
    policy_revision: int
    state_features: tuple[tuple[int, float], ...] = ()
    behavior_value: float | None = None

    def __post_init__(self) -> None:
        if (self.operation not in PILOT_OPERATIONS or not self.context_key
                or not self.features or len(self.features) != len(self.baseline)
                or len(self.features) != len(self.probabilities)
                or not 0 <= self.selected_index < len(self.features)):
            raise ValueError("Malformed pilot chance decision")
        if (self.probabilities[self.selected_index] <= 0
                or self.baseline[self.selected_index] <= 0):
            raise ValueError("Nature selected an ineligible candidate")


@dataclass(frozen=True, slots=True)
class PilotEpisode:
    """Fully resolved episode: positive return is good for the player."""

    player_return: float
    choices: tuple[PilotChoice, ...]
    terminal: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.player_return) or not self.terminal:
            raise ValueError("Only finite terminal pilot episodes can train Nature")


@dataclass(slots=True)
class TwoSiteNature:
    """Small candidate-feature softmax with a strictly positive exploration floor.

    Nature minimizes player return. `mix < 1` maintains base proposal
    support on every legal source option. For intents the base distribution
    follows source-emitted weights as a **proposal**, not a calibrated p;
    deferred draws use uniform legal physical instances as a proposal.
    """

    seed: int
    mix: float = 0.75
    learning_rate: float = 0.08
    score_clip: float = 5.0
    weights: dict[str, float] = field(default_factory=dict)
    revision: int = 0
    policy_id: str = PILOT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.seed) is not int or type(self.revision) is not int or self.revision < 0:
            raise ValueError("Invalid Nature seed/revision")
        if not (0 <= self.mix < 1 and math.isfinite(self.mix)
                and math.isfinite(self.learning_rate) and self.learning_rate > 0
                and math.isfinite(self.score_clip) and self.score_clip > 0):
            raise ValueError("Invalid Nature optimizer/exploration parameters")

    @staticmethod
    def legal_baseline(event: SemanticEvent) -> tuple[float, ...]:
        if event.operation not in PILOT_OPERATIONS:
            raise NotImplementedError(f"Outside intent/draw Nature pilot: {event.operation}")
        if event.operation == "enemy_intent":
            return event.normalized_intent_weights
        return (1 / event.support_count,) * event.support_count

    @staticmethod
    def _keys(event: SemanticEvent, context: str, features: tuple[str, ...]) -> tuple[str, ...]:
        if not context or len(features) != event.support_count or any(
            not isinstance(x, str) or not x for x in features
        ):
            raise ValueError("Must supply a feature label for every physical card/intent")
        return tuple(f"{event.operation}|{context}|{feature}" for feature in features)

    def _distribution(
        self, operation: str, context: str, features: tuple[str, ...],
        baseline: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if operation not in PILOT_OPERATIONS or not context:
            raise ValueError("Invalid pilot Nature event type")
        if len(features) != len(baseline) or not baseline:
            raise ValueError("Feature support mismatch")
        active = [i for i, value in enumerate(baseline) if value > 0]
        if not active or any(not math.isfinite(x) or x < 0 for x in baseline):
            raise ValueError("Invalid legal Nature proposal support")
        if not math.isclose(math.fsum(baseline), 1, abs_tol=1e-10):
            raise ValueError("Source proposal baseline does not normalize")
        keys = [f"{operation}|{context}|{feature}" for feature in features]
        largest = max(self.weights.get(keys[i], 0.0) for i in active)
        exp = [math.exp(max(-80.0, self.weights.get(keys[i], 0.0) - largest))
               if i in active else 0.0 for i in range(len(features))]
        denom = math.fsum(exp)
        soft = tuple(x / denom for x in exp)
        q = tuple((1 - self.mix) * base + self.mix * learned
                  for base, learned in zip(baseline, soft, strict=True))
        return q, soft

    def propose(
        self, event: SemanticEvent, *,
        context_key: str, candidate_features: tuple[str, ...],
        episode_key: str,
        public_observation: dict[str, Any] | None = None,
    ) -> tuple[SemanticProposal, PilotChoice]:
        self._keys(event, context_key, candidate_features)
        base = self.legal_baseline(event)
        q, _ = self._distribution(event.operation, context_key, candidate_features, base)
        rng = _rng(self.seed, self.revision, episode_key, event.replay_identity)
        draw = rng.random()
        cumul = 0.0
        selected = None
        for index, probability in enumerate(q):
            if probability <= 0:
                continue
            cumul += probability
            if draw < cumul:
                selected = index
                break
        if selected is None:
            selected = next(i for i in reversed(range(len(q))) if q[i] > 0)
        proposal = SemanticProposal(
            selected_index=selected, selected_id=event.candidate_ids[selected],
            proposal_id=f"{self.policy_id}:{self.revision}",
            log_q=math.log(q[selected]),
        )
        proposal.validate(event)
        return proposal, PilotChoice(
            operation=event.operation, context_key=context_key,
            features=candidate_features, baseline=base,
            probabilities=q, selected_index=selected, policy_revision=self.revision,
        )

    def update(self, episodes: tuple[PilotEpisode, ...]) -> dict[str, float]:
        """Unclipped REINFORCE for opponent, with leave-one-episode-out baseline.

        Assumes the player policy stays frozen during this Nature update.
        Complete episode scores are differentiated through *all* selected
        Nature decisions. A per-batch gradient clip is a numerical guard,
        not the same objective as importance-corrected native PPO.
        """
        if len(episodes) < 2:
            raise ValueError("Nature update needs at least two terminal episodes")
        if any(choice.policy_revision != self.revision
               for episode in episodes for choice in episode.choices):
            raise ValueError("Stale Nature behavior-policy revision")
        total = math.fsum(ep.player_return for ep in episodes)
        grads: dict[str, float] = defaultdict(float)
        for episode in episodes:
            # Leave-one-out baseline is independent of this episode's choices.
            advantage = episode.player_return - (
                total - episode.player_return
            ) / (len(episodes) - 1)
            for choice in episode.choices:
                q, soft = self._distribution(
                    choice.operation, choice.context_key,
                    choice.features, choice.baseline,
                )
                if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
                       for a, b in zip(q, choice.probabilities, strict=True)):
                    raise ValueError("Nature behavior distribution no longer matches stored q")
                i = choice.selected_index
                factor = advantage * self.mix * soft[i] / q[i]
                for j, feature in enumerate(choice.features):
                    if choice.baseline[j] <= 0:
                        continue
                    key = f"{choice.operation}|{choice.context_key}|{feature}"
                    grads[key] += factor * ((1 if i == j else 0) - soft[j])
        scale = 1 / len(episodes)
        norm = math.sqrt(math.fsum((x * scale) ** 2 for x in grads.values()))
        clip = min(1.0, self.score_clip / norm) if norm > 0 else 1.0
        for key, gradient in grads.items():
            self.weights[key] = self.weights.get(key, 0.0) - (
                self.learning_rate * clip * scale * gradient
            )
        self.revision += 1
        return {
            "nature_revision": float(self.revision),
            "mean_player_return": total / len(episodes),
            "nature_gradient_norm": norm,
            "nature_clip_factor": clip,
            "nature_choice_count": float(sum(len(ep.choices) for ep in episodes)),
        }

    def dump(self) -> dict[str, object]:
        return {
            "schema": PILOT_SCHEMA,
            "seed": self.seed, "mix": self.mix,
            "learning_rate": self.learning_rate,
            "score_clip": self.score_clip, "revision": self.revision,
            "weights": dict(sorted(self.weights.items())),
            "training_objective": "minimize_player_return_stress_only",
            "calibrated_native_reference_law": False,
        }

    @classmethod
    def restore(cls, data: dict[str, object]) -> TwoSiteNature:
        if (data.get("schema") != PILOT_SCHEMA or
                data.get("calibrated_native_reference_law") is not False):
            raise ValueError("Nature pilot checkpoint schema/provenance mismatch")
        weights = data["weights"]
        if not isinstance(weights, dict) or any(
            not isinstance(k, str) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for k, v in weights.items()
        ):
            raise ValueError("Malformed Nature weights")
        def integer(key: str) -> int:
            value = data.get(key)
            if type(value) is not int:
                raise ValueError(f"Malformed Nature integer parameter {key}")
            return value

        def number(key: str) -> float:
            value = data.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ValueError(f"Malformed Nature numeric parameter {key}")
            return float(value)

        return cls(
            seed=integer("seed"), mix=number("mix"),
            learning_rate=number("learning_rate"),
            score_clip=number("score_clip"),
            revision=integer("revision"), weights=dict(weights),
        )
