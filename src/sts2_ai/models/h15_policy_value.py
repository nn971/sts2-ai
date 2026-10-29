"""H9/H15 policy/value inference with version-correct combat observations.

An H9 warm start MUST use the original H7 public codec; appending move tokens
before H15 training changes the model function. Only H15-trained checkpoints
use the new move-aware schema. Critic estimates are clipped to [0,1] and
clip counts are recorded; true combat results always come from the emulator.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation
from sts2_ai.models.h15_combat_codec import H15_COMBAT_SCHEMA, collate_h15, encode_h15_observation
from sts2_ai.models.shared_game_objects import (
    SHARED_OBJECT_SCHEMA,
    encode_public_decision,
)
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer, collate_h7

H7_FORMAT = "sts2-h7-shared-object-transformer-v1"
H15_FORMAT = "sts2-h15-combat-search-policy-value-v1"


@dataclass(frozen=True)
class CombatPrediction:
    logits: tuple[float, ...]
    value: float
    unclipped_value: float


class H15CombatEvaluator:
    def __init__(
        self, model: SharedObjectTransformer, *, checkpoint_sha256: str,
        codec_schema: str = SHARED_OBJECT_SCHEMA,
    ):
        if codec_schema not in (SHARED_OBJECT_SCHEMA, H15_COMBAT_SCHEMA):
            raise ValueError("Unsupported search-teacher combat codec")
        self.codec_schema = codec_schema
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.checkpoint_sha256 = checkpoint_sha256
        self.clipped_negative_values = 0

    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "cpu") -> H15CombatEvaluator:
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fmt, schema = data.get("format"), data.get("object_schema")
        if ((fmt, schema) not in (
                (H7_FORMAT, SHARED_OBJECT_SCHEMA),
                (H15_FORMAT, H15_COMBAT_SCHEMA),
        )):
            raise ValueError("Expected H9 or H15 checkpoint with correct public schema")
        model = SharedObjectTransformer(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return cls(
            model.to(device), checkpoint_sha256=digest, codec_schema=schema,
        )

    def predict(self, observation: Observation,
                legal_actions: tuple[LegalAction, ...]) -> CombatPrediction:
        if observation.policy_id != FAIR_POLICY_ID or not legal_actions:
            raise ValueError("H15 search requires fair nonempty combat menu")
        if self.codec_schema == SHARED_OBJECT_SCHEMA:
            public = json.loads(observation.payload_json)
            if not isinstance(public, dict):
                raise ValueError("Malformed fair public observation")
            encoded = encode_public_decision(public, legal_actions)
            batch = collate_h7([encoded])
        else:
            encoded = encode_h15_observation(observation.payload_json, legal_actions)
            batch = collate_h15([encoded])
        if encoded.phase != "combat":
            raise ValueError("Search teacher cannot evaluate macro decisions")
        batch = batch.to(next(self.model.parameters()).device)
        with torch.inference_mode():
            logits, values = self.model(batch, known_phase=0)
        actions = tuple(float(x) for x in logits[0].cpu().tolist())
        raw = float(values[0].cpu())
        if (len(actions) != len(legal_actions) or
                not all(math.isfinite(x) for x in actions) or not math.isfinite(raw)):
            raise ValueError("Invalid H15 inference result")
        if raw < 0:
            self.clipped_negative_values += 1
        return CombatPrediction(actions, min(1.0, max(0.0, raw)), raw)

    def __call__(self, observation: Observation,
                 legal_actions: tuple[LegalAction, ...]) -> tuple[tuple[float, ...], float]:
        result = self.predict(observation, legal_actions)
        return result.logits, result.value
