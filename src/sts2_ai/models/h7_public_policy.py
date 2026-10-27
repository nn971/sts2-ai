"""Source-fair H7 combat actor adapter for the existing P0 legal-menu interface.

It deliberately implements the small PublicPolicy protocol used by
FrozenP0Player, without changing legacy v8/v13 serialization or workers.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.nature.protocol import canonical_sha256

from .shared_game_objects import encode_public_decision
from .shared_transformer import SharedObjectTransformer, collate_h7

H7_PLAYER_SPEC_SCHEMA = "sts2-h7-p0-player-spec-v1"


class TorchH7PublicPolicy:
    """Inference adapter; returns only a public action distribution and value."""

    def __init__(self, model: SharedObjectTransformer, *, model_id: str) -> None:
        self.model = model.eval()
        self.model_id = model_id
        self.calls = 0
        self.encoding_seconds = 0.0
        self.collate_seconds = 0.0
        self.forward_seconds = 0.0

    def timings_snapshot(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "encoding_seconds": self.encoding_seconds,
            "collate_seconds": self.collate_seconds,
            "forward_seconds": self.forward_seconds,
        }

    def evaluate(
        self, observation: Observation, legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate:
        begun = time.perf_counter()
        frame = json.loads(observation.payload_json)
        if not isinstance(frame, dict):
            raise ValueError("Malformed fair public observation")
        encoded = encode_public_decision(frame, legal_actions)
        self.encoding_seconds += time.perf_counter() - begun
        begun = time.perf_counter()
        batch = collate_h7([encoded]).to(next(self.model.parameters()).device)
        self.collate_seconds += time.perf_counter() - begun
        begun = time.perf_counter()
        with torch.inference_mode():
            logits, values = self.model(batch)
        result = tuple(float(x) for x in logits[0].detach().float().cpu())
        value = float(values[0].detach().float().cpu())
        self.forward_seconds += time.perf_counter() - begun
        self.calls += 1
        if len(result) != len(legal_actions) or not all(map(math.isfinite, (*result, value))):
            raise ValueError("Nonfinite H7 fair-policy inference")
        return PolicyValueEstimate(action_logits=result, value=value)


@dataclass(frozen=True, slots=True)
class H7PlayerSpec:
    """Immutable Torch checkpoint with the same runtime contract as FrozenPlayerSpec."""

    model_path: str
    model_sha256: str
    actor_seed: int
    temperature: float = 1.0
    greedy: bool = False
    action_scope: str = "all-legal-v1"
    schema: str = H7_PLAYER_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != H7_PLAYER_SPEC_SCHEMA or len(self.model_sha256) != 64
                or not self.model_path or not math.isfinite(self.temperature)
                or self.temperature <= 0 or type(self.actor_seed) is not int
                or self.action_scope != "all-legal-v1"):
            raise ValueError("Invalid source-fair H7 player specification")

    @classmethod
    def from_file(
        cls, path: Path, *, seed: int, temperature: float = 0.85,
        greedy: bool = False,
    ) -> H7PlayerSpec:
        return cls(
            str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest(),
            seed, temperature, greedy,
        )

    def load_pinned(self) -> TorchH7PublicPolicy:
        checkpoint = Path(self.model_path)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.model_sha256:
            raise ValueError("H7 actor checkpoint digest mismatch")
        torch.set_num_threads(1)
        return TorchH7PublicPolicy(
            SharedObjectTransformer.load_checkpoint(checkpoint),
            model_id=f"h7-{self.model_sha256[:16]}",
        )

    def episode_seed(self, *, fixture_seed: str, fixture_id: str) -> int:
        return int(canonical_sha256([
            self.actor_seed, fixture_seed, fixture_id, self.model_sha256,
            self.temperature, self.greedy, self.action_scope,
        ])[:16], 16)

    @property
    def policy_id(self) -> str:
        mode = "greedy" if self.greedy else "sampled"
        return f"h7-{mode}-{self.action_scope}-{self.model_sha256[:16]}"
