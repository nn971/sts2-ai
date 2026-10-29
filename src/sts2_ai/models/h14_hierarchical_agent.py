"""H14 hierarchy: pinned H9 tactical actor + independently trained macro actor.

The two networks intentionally have separate public codecs. The old H9
checkpoint receives its original H7 combat tokens; H14's macro network sees
new persistent-state features. No hidden emulator access or outcome modes.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation

from .h14_macro_actor_critic import H14MacroActorCritic
from .h9_frozen_codec import (
    SHARED_OBJECT_SCHEMA as H9_OBJECT_SCHEMA,
    encode_public_decision as encode_h9_combat,
)
from .shared_game_objects import SHARED_OBJECT_SCHEMA, EncodedDecision, encode_public_decision
from .shared_transformer import H7Config, SharedObjectTransformer, collate_h7

H9_CHECKPOINT_FORMAT = "sts2-h7-shared-object-transformer-v1"


@dataclass(frozen=True)
class H14MacroActionSample:
    decision: EncodedDecision
    selected: int
    behavior_log_probability: float
    behavior_value: float


class FrozenH9CombatActor:
    """Bitwise-frozen H9-format checkpoint with the historical H7 input codec."""

    def __init__(self, model: SharedObjectTransformer, *, model_sha256: str):
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_sha256 = model_sha256

    @classmethod
    def load_pinned(
        cls, path: Path, *, expected_sha256: str,
        map_location: Any = "cpu",
    ) -> FrozenH9CombatActor:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("H9 checkpoint SHA256 mismatch")
        data = torch.load(path, map_location=map_location, weights_only=True)
        if (data.get("format") != H9_CHECKPOINT_FORMAT
                or data.get("object_schema") != H9_OBJECT_SCHEMA):
            raise ValueError("Expected original H7/H9 combat checkpoint")
        model = SharedObjectTransformer(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return cls(model, model_sha256=digest)

    def choose(self, observation: Observation,
               actions: list[LegalAction] | tuple[LegalAction, ...]) -> Decision:
        if observation.policy_id != FAIR_POLICY_ID or not actions:
            raise ValueError("H9 requires nonempty fair public combat menu")
        state = json.loads(observation.payload_json)
        if not isinstance(state, dict) or not isinstance(state.get("combat"), dict):
            raise ValueError("H9 combat actor received noncombat observation")
        encoded = encode_h9_combat(state, actions)
        # Reinterpret *only the schema label* for the same fixed-size H7
        # collator. Actual H9 object/action tokens are unchanged.
        batch = collate_h7([
            replace(encoded, schema=SHARED_OBJECT_SCHEMA),
        ]).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            logits, _ = self.model(batch, known_phase=0)
        scores = logits[0].detach().float().cpu().tolist()
        if len(scores) != len(actions) or any(not math.isfinite(x) for x in scores):
            raise ValueError("Invalid H9 combat policy logits")
        index = max(range(len(actions)),
                    key=lambda i: (scores[i], actions[i].action_id))
        return Decision(actions[index], policy_name="frozen-h9-" + self.model_sha256[:16])


class H14HierarchicalAgent:
    """One pinned H9 combat policy; macro policy and critic share H14 network."""

    def __init__(
        self, combat: FrozenH9CombatActor, macro: H14MacroActorCritic,
        *, episode_seed: str, macro_checkpoint_sha256: str,
        temperature: float = 1.0, greedy_macro: bool = False,
    ) -> None:
        if not episode_seed or not macro_checkpoint_sha256 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid hierarchical actor provenance")
        self.combat = combat
        self.macro = macro.eval()
        self.greedy_macro = greedy_macro
        self.temperature = temperature
        self.samples: list[H14MacroActionSample] = []
        seed_material = json.dumps(
            [episode_seed, combat.model_sha256, macro_checkpoint_sha256, temperature],
            separators=(",", ":"),
        )
        self.rng = random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16))
        self.policy_id = ("h14-macro-" + macro_checkpoint_sha256[:16]
                          + "-h9-" + combat.model_sha256[:16])

    def choose(self, observation: Observation,
               legal_actions: list[LegalAction] | tuple[LegalAction, ...]) -> Decision:
        if observation.policy_id != FAIR_POLICY_ID or not legal_actions:
            raise ValueError("H14 actor requires fair public observation")
        public = json.loads(observation.payload_json)
        if not isinstance(public, dict):
            raise ValueError("Malformed H14 public observation")
        if isinstance(public.get("combat"), dict):
            return self.combat.choose(observation, legal_actions)
        decision = encode_public_decision(public, legal_actions)
        score = self.macro.evaluate(public, legal_actions)
        scaled = [x / self.temperature for x in score.action_logits]
        largest = max(scaled)
        weights = [math.exp(x - largest) for x in scaled]
        total = sum(weights)
        probs = [x / total for x in weights]
        if self.greedy_macro:
            index = max(range(len(legal_actions)),
                        key=lambda i: (probs[i], legal_actions[i].action_id))
        else:
            index = self.rng.choices(range(len(legal_actions)), weights=probs, k=1)[0]
        self.samples.append(H14MacroActionSample(
            decision, index, math.log(probs[index]), score.continuation_value,
        ))
        return Decision(legal_actions[index], policy_name=self.policy_id)
