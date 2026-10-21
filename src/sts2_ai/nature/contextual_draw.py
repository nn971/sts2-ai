"""Observation-conditioned draw Nature: small dependency-free bilinear network.

Uses the *existing fair player observation*, then scores each legal physical
card draw candidate. The emulator still owns legality, hidden decks, and
all consequences. A source-law mixture preserves the full reference support.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .semantic import SemanticEvent, SemanticProposal
from .two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature, _rng

CONTEXTUAL_SCHEMA = "sts2-ai-nature-contextual-draw-v1"
FEATURE_DIM = 256
HIDDEN = 24
_PRIVATE = frozenset({
    "pending", "chance_law", "chancelaw", "semantic_deck", "semanticdeck",
    "rng_state", "rngstate", "draw_pile_order", "drawpileorder",
    "exact_state", "canonical_state_hash", "source_seed", "sourceopeningfingerprint",
})
# Identifiers of entities are ephemeral. Species, card, power and relic IDs
# remain observable and are deliberately encoded.
_EPHEMERAL = frozenset({"instance_id", "instanceid", "cardinstanceid", "decision_index",
                        "decisionindex", "node_id", "nodeid", "handle"})


@lru_cache(maxsize=32768)
def _bucket(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % FEATURE_DIM


def encode_public_observation(observation: dict[str, Any]) -> tuple[tuple[int, float], ...]:
    """Hash *all* player-visible scalar/collection fields, without list order.

    This is the same structured public observation given to the player, not an
    engine state or a Nature-private Chance Law. Entity lists are pooled, so
    permutation of unordered enemies/piles does not affect logits.
    """
    if not isinstance(observation, dict) or not isinstance(observation.get("combat"), dict):
        raise ValueError("Contextual Nature requires a complete fair combat observation")
    if type(observation.get("hp")) is not int or type(observation.get("max_hp")) is not int:
        raise ValueError("Contextual Nature missing public player HP")
    values: dict[int, float] = defaultdict(float)

    def visit(value: Any, prefix: str, depth: int) -> None:
        if depth > 12:
            raise ValueError("Excessively nested fair observation")
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                normalized = key.lower().replace("-", "_")
                if normalized in _PRIVATE:
                    raise ValueError("Privileged data passed to contextual Nature")
                if normalized in _EPHEMERAL:
                    continue
                visit(item, f"{prefix}.{key}", depth + 1)
        elif isinstance(value, list):
            values[_bucket(f"{prefix}.length")] += math.log1p(len(value)) / 5
            for item in value:
                visit(item, prefix + "[]", depth + 1)
        elif value is None:
            values[_bucket(f"{prefix}=none")] += 1.0
        elif isinstance(value, bool):
            values[_bucket(f"{prefix}={value}")] += 1.0
        elif isinstance(value, str):
            values[_bucket(f"{prefix}={value}")] += 1.0
        elif type(value) in (int, float) and math.isfinite(value):
            # Continuous information and coarse buckets both matter.
            signed = math.copysign(math.log1p(abs(value)), value) / 7
            values[_bucket(f"{prefix}.number")] += signed
            bucket = math.copysign(int(math.log2(1 + abs(value))), value)
            values[_bucket(f"{prefix}.bucket={bucket}")] += 0.5
        else:
            raise ValueError(f"Unsupported public observation value: {prefix}")

    visit(observation, "visible", 0)
    return tuple(sorted((key, math.tanh(val / 4)) for key, val in values.items()
                        if val != 0))


@dataclass(slots=True)
class ContextualDrawNature:
    """Tanh(state encoder) dot candidate embedding, followed by softmax.

    REINFORCE gradients are computed analytically; no extra Torch dependency
    is imposed on the 15 independent emulator workers.
    """

    seed: int
    mix: float = 0.75
    learning_rate: float = 0.03
    score_clip: float = 5.0
    revision: int = 0
    state_weight: list[list[float]] = field(default_factory=list)
    candidate_vectors: dict[str, list[float]] = field(default_factory=dict)
    candidate_bias: dict[str, float] = field(default_factory=dict)
    policy_id: str = CONTEXTUAL_SCHEMA

    def __post_init__(self) -> None:
        if (type(self.seed) is not int or type(self.revision) is not int
                or self.revision < 0 or not 0 <= self.mix < 1
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.score_clip) or self.score_clip <= 0):
            raise ValueError("Invalid contextual Nature configuration")
        if not self.state_weight:
            rng = _rng(self.seed, "public-observation-encoder-v1")
            self.state_weight = [
                [rng.gauss(0.0, 0.065) for _ in range(FEATURE_DIM)]
                for _ in range(HIDDEN)
            ]
        if (len(self.state_weight) != HIDDEN or any(len(row) != FEATURE_DIM
                or any(not math.isfinite(x) for x in row)
                for row in self.state_weight)):
            raise ValueError("Invalid contextual Nature state network")

    def _candidate(self, key: str) -> list[float]:
        if key not in self.candidate_vectors:
            rng = _rng(self.seed, "candidate-v1", key)
            self.candidate_vectors[key] = [rng.gauss(0.0, 0.065) for _ in range(HIDDEN)]
        return self.candidate_vectors[key]

    def _forward(
        self, features: tuple[tuple[int, float], ...], candidates: tuple[str, ...],
        baseline: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        if len(candidates) != len(baseline) or not candidates:
            raise ValueError("Invalid candidate and Chance Law support")
        if any(not 0 <= index < FEATURE_DIM or not math.isfinite(value)
               for index, value in features):
            raise ValueError("Malformed contextual state features")
        h = tuple(math.tanh(sum(row[index] * val for index, val in features))
                  for row in self.state_weight)
        logits = [
            sum(a * b for a, b in zip(h, self._candidate(card), strict=True))
            + self.candidate_bias.get(card, 0.0)
            for card in candidates
        ]
        eligible = [i for i, p in enumerate(baseline) if p > 0]
        if not eligible or any(not math.isfinite(p) or p < 0 for p in baseline):
            raise ValueError("Invalid reference draw support")
        if not math.isclose(math.fsum(baseline), 1.0, abs_tol=1e-10):
            raise ValueError("Unnormalized reference draw law")
        maximum = max(logits[i] for i in eligible)
        masses = [math.exp(max(-80.0, v - maximum)) if i in eligible else 0.0
                  for i, v in enumerate(logits)]
        total = math.fsum(masses)
        soft = tuple(v / total for v in masses)
        q = tuple((1 - self.mix) * p + self.mix * z
                  for p, z in zip(baseline, soft, strict=True))
        return q, soft, h

    def propose(
        self, event: SemanticEvent, *, context_key: str,
        candidate_features: tuple[str, ...], episode_key: str,
        public_observation: dict[str, Any] | None = None,
    ) -> tuple[SemanticProposal, PilotChoice]:
        if event.operation != "draw_card":
            raise ValueError("Contextual Nature currently controls only physical draws")
        TwoSiteNature._keys(event, context_key, candidate_features)
        if public_observation is None:
            raise ValueError("Missing fair state at contextual Nature draw")
        state_features = encode_public_observation(public_observation)
        baseline = TwoSiteNature.legal_baseline(event)
        q, _, _ = self._forward(state_features, candidate_features, baseline)
        rng = _rng(self.seed, self.revision, episode_key, event.replay_identity)
        threshold = rng.random()
        selected = len(q) - 1
        for index, p in enumerate(q):
            threshold -= p
            if threshold < 0:
                selected = index
                break
        proposal = SemanticProposal(
            selected_index=selected, selected_id=event.candidate_ids[selected],
            proposal_id=f"{self.policy_id}:{self.revision}",
            log_q=math.log(q[selected]),
        )
        proposal.validate(event)
        return proposal, PilotChoice(
            operation=event.operation, context_key=context_key,
            features=candidate_features, baseline=baseline,
            probabilities=q, selected_index=selected,
            policy_revision=self.revision, state_features=state_features,
        )

    def update(self, episodes: tuple[PilotEpisode, ...]) -> dict[str, float]:
        if len(episodes) < 2:
            raise ValueError("Contextual Nature needs multiple complete combats")
        if any(choice.policy_revision != self.revision
               for episode in episodes for choice in episode.choices):
            raise ValueError("Stale contextual Nature choice")
        total_return = math.fsum(ep.player_return for ep in episodes)
        grad_state: dict[tuple[int, int], float] = defaultdict(float)
        grad_candidates: dict[tuple[str, int], float] = defaultdict(float)
        grad_bias: dict[str, float] = defaultdict(float)
        for ep in episodes:
            advantage = ep.player_return - (total_return - ep.player_return) / (len(episodes) - 1)
            for choice in ep.choices:
                if choice.operation != "draw_card" or not choice.state_features:
                    raise ValueError("Noncontextual training choice in contextual Nature")
                q, soft, h = self._forward(
                    choice.state_features, choice.features, choice.baseline
                )
                if any(not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)
                       for x, y in zip(q, choice.probabilities, strict=True)):
                    raise ValueError("Contextual behavior q changed before on-policy update")
                selected = choice.selected_index
                factor = advantage * self.mix * soft[selected] / q[selected]
                grad_h = [0.0] * HIDDEN
                for j, key in enumerate(choice.features):
                    if choice.baseline[j] <= 0:
                        continue
                    coefficient = factor * ((1.0 if j == selected else 0.0) - soft[j])
                    grad_bias[key] += coefficient
                    vector = self._candidate(key)
                    for k in range(HIDDEN):
                        grad_candidates[(key, k)] += coefficient * h[k]
                        grad_h[k] += coefficient * vector[k]
                for k in range(HIDDEN):
                    dh = grad_h[k] * (1.0 - h[k] ** 2)
                    for i, val in choice.state_features:
                        grad_state[(k, i)] += dh * val

        count = len(episodes)
        norm = math.sqrt(math.fsum(
            (g / count) ** 2
            for gradient in (grad_state, grad_candidates, grad_bias)
            for g in gradient.values()
        ))
        clip = min(1.0, self.score_clip / norm) if norm > 0 else 1.0
        step = self.learning_rate * clip / count
        for (k, i), grad in grad_state.items():
            self.state_weight[k][i] -= step * grad
        for (key, k), grad in grad_candidates.items():
            self.candidate_vectors[key][k] -= step * grad
        for key, grad in grad_bias.items():
            self.candidate_bias[key] = self.candidate_bias.get(key, 0.0) - step * grad
        self.revision += 1
        return {
            "nature_revision": float(self.revision),
            "mean_player_return": total_return / count,
            "nature_gradient_norm": norm, "nature_clip_factor": clip,
            "nature_choice_count": float(sum(len(ep.choices) for ep in episodes)),
        }

    def dump(self) -> dict[str, object]:
        return {
            "schema": CONTEXTUAL_SCHEMA, "seed": self.seed,
            "mix": self.mix, "learning_rate": self.learning_rate,
            "score_clip": self.score_clip, "revision": self.revision,
            "state_weight": [list(row) for row in self.state_weight],
            "candidate_vectors": {key: list(vector)
                                  for key, vector in self.candidate_vectors.items()},
            "candidate_bias": dict(self.candidate_bias),
            "training_objective": "minimize_frozen_player_hp_preservation",
            "reference_law_fidelity_gate": False,
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> ContextualDrawNature:
        if data.get("schema") != CONTEXTUAL_SCHEMA:
            raise ValueError("Unexpected contextual Nature checkpoint schema")
        vectors = data["candidate_vectors"]
        biases = data["candidate_bias"]
        if not isinstance(vectors, dict) or not isinstance(biases, dict):
            raise ValueError("Malformed contextual candidate parameters")
        if any(not isinstance(key, str) or not isinstance(value, list)
               or len(value) != HIDDEN or any(not math.isfinite(x) for x in value)
               for key, value in vectors.items()):
            raise ValueError("Invalid candidate embeddings")
        if any(not isinstance(key, str) or not isinstance(v, (int, float))
               or not math.isfinite(v) for key, v in biases.items()):
            raise ValueError("Invalid contextual candidate biases")
        return cls(
            seed=data["seed"], mix=data["mix"],
            learning_rate=data["learning_rate"],
            score_clip=data["score_clip"], revision=data["revision"],
            state_weight=data["state_weight"],
            candidate_vectors={k: list(v) for k, v in vectors.items()},
            candidate_bias=dict(biases),
        )


def restore_nature(data: dict[str, Any]) -> TwoSiteNature | ContextualDrawNature:
    """Versioned worker/optimizer checkpoint dispatch, preserving H1 baselines."""
    if data.get("schema") == CONTEXTUAL_SCHEMA:
        return ContextualDrawNature.restore(data)
    return TwoSiteNature.restore(data)
