"""Frozen v15 public-policy bridge to the real P0 semantic-combat action menu.

The emulator remains the *only* source of legal player actions and fair
observations. This is a portable checkpoint adapter: it loads exported
phase-split v15 JSON, never updates its weights, and does not expose Nature
pending support, deck constraints or exact state to the player model.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.protocol import canonical_sha256

PUBLIC_POLICY_ID = "prototype-fair-v0"
PLAYER_SPEC_SCHEMA = "sts2-ai-nature-p0-frozen-player-v1"
PLAYER_ACTION_SCOPES = frozenset({"silent-basic-v1", "all-legal-v1"})
SUPPORTED_BASIC_CARDS = frozenset({"proto.silent.strike", "proto.silent.defend"})


class FrozenPolicyInferenceError(RuntimeError):
    """Bad checkpoint, malformed public menu, or non-finite model inference."""


class PublicPolicy(Protocol):
    model_id: str

    def evaluate(
        self, observation: Observation, legal_actions: tuple[LegalAction, ...],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class FrozenPlayerSpec:
    model_path: str
    model_sha256: str
    actor_seed: int
    temperature: float = 1.0
    greedy: bool = True
    action_scope: str = "silent-basic-v1"
    schema: str = PLAYER_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLAYER_SPEC_SCHEMA or not self.model_path:
            raise ValueError("Unsupported frozen v15 player specification")
        if (len(self.model_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.model_sha256)):
            raise ValueError("Frozen policy requires SHA256 pinned model bytes")
        if type(self.actor_seed) is not int:
            raise ValueError("Frozen player seed must be an integer")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Frozen player temperature must be positive")
        if self.action_scope not in PLAYER_ACTION_SCOPES:
            raise ValueError("Unsupported frozen player action scope")

    @classmethod
    def from_file(
        cls, path: Path, *, seed: int, temperature: float = 1.0,
        greedy: bool = True, action_scope: str = "silent-basic-v1",
    ) -> FrozenPlayerSpec:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen portable v15 JSON missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(str(path.resolve()), digest, seed, temperature, greedy, action_scope)

    def load_pinned(self) -> PhaseSplitNeuralModel:
        path = Path(self.model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.model_sha256:
            raise FrozenPolicyInferenceError("Frozen checkpoint digest mismatch")
        model = PhaseSplitNeuralModel.load(path)
        if model.model_id == "":
            raise FrozenPolicyInferenceError("Missing checkpoint model ID")
        return model

    def episode_seed(self, *, fixture_seed: str, fixture_id: str) -> int:
        return int(
            canonical_sha256([
                self.actor_seed, fixture_seed, fixture_id,
                self.model_sha256, self.temperature, self.greedy, self.action_scope,
            ])[:16], 16,
        )

    @property
    def policy_id(self) -> str:
        variant = "greedy" if self.greedy else "sampled"
        return f"frozen-v15-{variant}-{self.action_scope}-{self.model_sha256[:16]}"


class FrozenP0Player:
    """One frozen player's seeded legal-action sampler for one combat."""

    def __init__(
        self, model: PublicPolicy, *, actor_seed: int,
        temperature: float = 1.0, greedy: bool = True,
        action_scope: str = "all-legal-v1",
        capture_training: bool = False,
    ):
        if type(actor_seed) is not int:
            raise ValueError("Actor seed must be an integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid actor temperature")
        if action_scope not in PLAYER_ACTION_SCOPES:
            raise ValueError("Unsupported frozen player action scope")
        self.model = model
        self.action_scope = action_scope
        self.temperature = temperature
        self.greedy = greedy
        self.rng = random.Random(actor_seed)
        self.policy_id = f"frozen-v15-{action_scope}-{model.model_id}"
        self.decisions = 0
        self.capture_training = capture_training
        self.training_actions: list[dict[str, Any]] = []

    @staticmethod
    def _public_types(
        observation: dict[str, Any], raw_actions: list[dict[str, Any]],
    ) -> tuple[Observation, tuple[LegalAction, ...]]:
        if not isinstance(observation, dict) or not isinstance(raw_actions, list):
            raise FrozenPolicyInferenceError("Malformed fair player boundary")
        if "semanticDeck" in observation or "pending" in observation:
            raise FrozenPolicyInferenceError("Hidden semantic state leaked to player")
        if not raw_actions:
            raise FrozenPolicyInferenceError("Empty legal player action menu")
        actions: list[LegalAction] = []
        seen: set[str] = set()
        for entry in raw_actions:
            if not isinstance(entry, dict):
                raise FrozenPolicyInferenceError("Malformed fair legal action")
            ident, kind, payload = (
                entry.get("actionId"), entry.get("kind"), entry.get("payloadJson")
            )
            if (not isinstance(ident, str) or not ident or ident in seen
                    or not isinstance(kind, str) or not kind
                    or not isinstance(payload, str)):
                raise FrozenPolicyInferenceError("Duplicate/malformed legal player action")
            try:
                if not isinstance(json.loads(payload), dict):
                    raise ValueError("Action payload must be a JSON object")
            except (ValueError, TypeError) as exc:
                raise FrozenPolicyInferenceError("Invalid public action payload") from exc
            actions.append(LegalAction(ident, kind, payload))
            seen.add(ident)
        # JSON source semantics are preserved (all fields are from a fair
        # observation). The value is the public content's own stable hash,
        # not a hidden/full-state fingerprint.
        text = json.dumps(observation, separators=(",", ":"), sort_keys=True)
        public = Observation(
            PUBLIC_POLICY_ID, text, canonical_sha256(observation),
        )
        return public, tuple(actions)

    def choose(
        self, observation: dict[str, Any],
        raw_actions: list[dict[str, Any]],
    ) -> str:
        public, actions = self._public_types(observation, raw_actions)
        if self.action_scope == "silent-basic-v1":
            hand = {
                str(card["instance_id"]): card["card_id"]
                for card in observation.get("combat", {}).get("hand", [])
            }
            filtered: list[LegalAction] = []
            for action in actions:
                if action.kind == "end_turn":
                    filtered.append(action)
                elif action.kind == "play_card":
                    card_id = json.loads(action.payload_json).get("CardInstanceId")
                    if hand.get(str(card_id)) in SUPPORTED_BASIC_CARDS:
                        filtered.append(action)
            actions = tuple(filtered)
            if not actions:
                raise FrozenPolicyInferenceError(
                    "No supported basic Silent action in the fair legal menu"
                )
        estimate = self.model.evaluate(public, actions)
        logits = estimate.action_logits
        if len(logits) != len(actions) or any(
            not isinstance(x, (float, int)) or not math.isfinite(x) for x in logits
        ):
            raise FrozenPolicyInferenceError("Frozen player returned invalid logits")
        if self.greedy:
            selected = max(range(len(actions)), key=lambda i: logits[i])
        else:
            peak = max(logits)
            weights = [
                math.exp((value - peak) / self.temperature) for value in logits
            ]
            draw = self.rng.random() * math.fsum(weights)
            selected = len(actions) - 1
            for index, mass in enumerate(weights):
                draw -= mass
                if draw < 0:
                    selected = index
                    break
        self.decisions += 1
        if self.capture_training:
            if self.greedy:
                raise FrozenPolicyInferenceError(
                    "PPO behavior collection requires stochastic policy sampling"
                )
            if not isinstance(estimate.value, (float, int)) or not math.isfinite(
                estimate.value
            ):
                raise FrozenPolicyInferenceError("Invalid frozen PPO behavior value")
            maximum = max(logits)
            normalized = math.fsum(
                math.exp((x - maximum) / self.temperature) for x in logits
            )
            logp = (
                (logits[selected] - maximum) / self.temperature
                - math.log(normalized)
            )
            self.training_actions.append({
                "observation": {
                    "policy_id": public.policy_id,
                    "payload_json": public.payload_json,
                    "observation_hash": public.observation_hash,
                },
                "legal_actions": [
                    {"action_id": a.action_id, "kind": a.kind,
                     "payload_json": a.payload_json} for a in actions
                ],
                "chosen_index": selected,
                "selected_id": actions[selected].action_id,
                "behavior_logprob": logp,
                "behavior_value": float(estimate.value),
                "temperature": self.temperature,
                "action_scope": self.action_scope,
                "behavior_model_id": self.model.model_id,
            })
        return actions[selected].action_id
