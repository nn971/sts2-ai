"""Offline diagnostics for search targets and real MCTS cutoff observations.

Cutoff samples are *unlabeled*: a handcrafted/learned cutoff estimate is not a
ground-truth return. Never merge these records into policy/value targets.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from sts2_ai.emulator import Observation

from .targets import TrainingExample

SCHEMA = "sts2-ai-cutoff-observation-v1"


@dataclass(slots=True)
class CutoffSample:
    observation_hash: str
    information_policy: str
    observation_json: str
    source_run_seed: str
    sampled_occurrences: int = 1


class CutoffSampler:
    """Bounded, deterministic, passive sampler of *nonterminal* rollout cutoffs.

    Every Nth callback is retained, deduplicated by (run seed, policy, hash).
    Limits apply to unique observations; discarded callbacks never alter RNG or
    the UCT search. Counts refer to sampled callbacks, not all rollouts.
    """

    def __init__(self, *, every: int = 32, max_unique: int = 10000) -> None:
        if every < 1 or max_unique < 1:
            raise ValueError("every and max_unique must be positive")
        self.every = every
        self.max_unique = max_unique
        self.seen = 0
        self.sampled = 0
        self._records: dict[tuple[str, str, str], CutoffSample] = {}

    def record(self, observation: Observation, *, run_seed: str) -> None:
        self.seen += 1
        if self.seen % self.every:
            return
        self.sampled += 1
        key = (run_seed, observation.policy_id, observation.observation_hash)
        existing = self._records.get(key)
        if existing is not None:
            if existing.observation_json != observation.payload_json:
                raise ValueError("Conflicting payloads for one cutoff observation hash")
            existing.sampled_occurrences += 1
        elif len(self._records) < self.max_unique:
            self._records[key] = CutoffSample(
                observation_hash=observation.observation_hash,
                information_policy=observation.policy_id,
                observation_json=observation.payload_json,
                source_run_seed=run_seed,
            )

    def write_jsonl(self, path: Path, *, provenance: dict[str, Any]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for key in sorted(self._records):
                data = {
                    "schema": SCHEMA,
                    **asdict(self._records[key]),
                    "capture_every": self.every,
                    "provenance": provenance,
                }
                stream.write(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
        return len(self._records)


def load_cutoff_samples(path: Path) -> tuple[CutoffSample, ...]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
                raise ValueError(f"Invalid cutoff sample schema on line {index}")
            record = CutoffSample(
                observation_hash=str(raw["observation_hash"]),
                information_policy=str(raw["information_policy"]),
                observation_json=str(raw["observation_json"]),
                source_run_seed=str(raw["source_run_seed"]),
                sampled_occurrences=int(raw["sampled_occurrences"]),
            )
            if record.sampled_occurrences < 1:
                raise ValueError(f"Invalid sampled_occurrences on line {index}")
            result.append(record)
    return tuple(result)


def teacher_policy_report(examples: tuple[TrainingExample, ...]) -> dict[str, Any]:
    """Quantify the supervision signal, including ambiguity from flat visits."""
    if not examples:
        raise ValueError("Training dataset is empty")
    entropy, relative_entropy, effective, max_prob = [], [], [], []
    confident = 0
    uniform_like = 0
    for example in examples:
        probabilities = [max(0.0, p.probability) for p in example.policy_targets]
        if not probabilities:
            raise ValueError("Example has no legal policy targets")
        total = sum(probabilities)
        probabilities = (
            [p / total for p in probabilities]
            if total > 0
            else [1.0 / len(probabilities)] * len(probabilities)
        )
        h = -sum(p * math.log(p) for p in probabilities if p > 0)
        normalized = h / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        entropy.append(h)
        relative_entropy.append(normalized)
        effective.append(math.exp(h))
        max_prob.append(max(probabilities))
        confident += max(probabilities) >= 0.7 and len(probabilities) > 1
        uniform_like += normalized >= 0.95 and len(probabilities) > 1

    return {
        "examples": len(examples),
        "mean_legal_actions": fmean(len(e.policy_targets) for e in examples),
        "mean_entropy_nats": fmean(entropy),
        "mean_normalized_entropy": fmean(relative_entropy),
        "mean_effective_actions": fmean(effective),
        "mean_top_action_mass": fmean(max_prob),
        "fraction_multi_action_confident_ge_0_7": confident / len(examples),
        "fraction_multi_action_near_uniform_ge_0_95": uniform_like / len(examples),
        "mean_search_budget": fmean(e.search_budget for e in examples),
    }


def observation_shift_report(
    examples: tuple[TrainingExample, ...],
    samples: tuple[CutoffSample, ...],
) -> dict[str, Any]:
    """Compare root vs sampled cutoff marginals; diagnostic, not causal inference."""
    if not examples or not samples:
        raise ValueError("Both searched roots and cutoff samples are required")

    root_policies = {e.information_policy for e in examples}
    cutoff_policies = {s.information_policy for s in samples}
    if root_policies != cutoff_policies:
        raise ValueError("Root and cutoff information policies differ")

    def describe(observations: list[dict[str, Any]]) -> dict[str, Any]:
        phases = Counter(str(state.get("phase", "?")) for state in observations)
        hp_fractions = []
        for state in observations:
            hp, maximum = state.get("hp"), state.get("max_hp")
            if isinstance(hp, (int, float)) and isinstance(maximum, (int, float)) and maximum > 0:
                hp_fractions.append(max(0.0, min(1.0, hp / maximum)))
        return {
            "count": len(observations),
            "phase_fraction": {key: phases[key] / len(observations) for key in sorted(phases)},
            "combat_fraction": sum(
                isinstance(state.get("combat"), dict) for state in observations
            ) / len(observations),
            "mean_hp_fraction": fmean(hp_fractions) if hp_fractions else None,
            "mean_act": _mean_numeric(observations, "act"),
            "mean_floor": _mean_numeric(observations, "floor"),
        }

    root_obs = [_parse_object(e.observation_json) for e in examples]
    # Sample unique observations, so duplication of certain game states does not
    # masquerade as independent evidence. Counts also remain available in JSONL.
    cutoff_obs = [_parse_object(s.observation_json) for s in samples]
    root_desc = describe(root_obs)
    cutoff_desc = describe(cutoff_obs)
    root_phases = root_desc["phase_fraction"]
    cutoff_phases = cutoff_desc["phase_fraction"]
    phase_tv = 0.5 * sum(
        abs(root_phases.get(k, 0.0) - cutoff_phases.get(k, 0.0))
        for k in set(root_phases) | set(cutoff_phases)
    )
    root_hashes = {e.observation_hash for e in examples}
    cutoff_hashes = {s.observation_hash for s in samples}
    return {
        "roots": root_desc,
        "cutoffs": cutoff_desc,
        "phase_total_variation_distance": phase_tv,
        "cutoff_unique_hash_overlap_with_roots": (
            len(cutoff_hashes & root_hashes) / len(cutoff_hashes)
        ),
        "sampled_cutoff_occurrences": sum(s.sampled_occurrences for s in samples),
        "warning": (
            "Cutoffs are unlabeled, deduplicated and periodically sampled. "
            "Distributions describe captured search states; they are not a fair-policy "
            "performance metric or rollout-return targets."
        ),
    }


def _parse_object(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Observation payload must be a JSON object")
    return value


def _mean_numeric(states: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(s[key]) for s in states
        if isinstance(s.get(key), (int, float)) and not isinstance(s[key], bool)
    ]
    return fmean(values) if values else None
