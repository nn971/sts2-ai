from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from sts2_ai.emulator import Observation
from sts2_ai.models import NeuralPolicyValueModel, PolicyValueModel, load_model

from .mcts import sts2_value


class LearnedCutoffValue:
    """Opt-in learned value for nonterminal MCTS rollout cutoffs.

    The terminal game outcome always overrides the model. Policy logits are not
    used: the heuristic continues to choose rollout actions. A configurable
    learned_weight blends the estimate with the hand-built nonterminal value.
    The content hash
    fingerprints the *actual weight file*, not just a reusable model name.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        *,
        model_sha256: str,
        learned_weight: float = 1.0,
    ) -> None:
        if len(model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_sha256
        ):
            raise ValueError("model_sha256 must be a SHA-256 hexadecimal digest")
        if not math.isfinite(learned_weight) or not 0.0 <= learned_weight <= 1.0:
            raise ValueError("learned_weight must lie in [0, 1]")
        if isinstance(model, NeuralPolicyValueModel) and not model.value_head_trained:
            raise ValueError("Cannot use an untrained policy-only value head as cutoff")
        self._model = model
        self.learned_weight = learned_weight
        self.value_id = (
            f"learned-model-blend-v2-weight-{learned_weight.hex()}"
            f"-sha256-{model_sha256}"
        )

    @classmethod
    def load(
        cls, path: Path, *, learned_weight: float = 1.0
    ) -> LearnedCutoffValue:
        payload = path.read_bytes()
        model = load_model(path)
        return cls(
            model,
            model_sha256=hashlib.sha256(payload).hexdigest(),
            learned_weight=learned_weight,
        )

    def __call__(self, observation: Observation) -> float:
        raw = json.loads(observation.payload_json)
        if not isinstance(raw, dict):
            raise ValueError("Learned cutoff requires an object observation")
        outcome = raw.get("terminal_outcome")
        if outcome == "victory":
            return 1.0
        if outcome == "defeat":
            return -1.0
        if outcome is not None:
            raise ValueError(f"Unsupported terminal outcome: {outcome!r}")

        if self.learned_weight == 0.0:
            return sts2_value(observation)

        prediction = self._model.evaluate(observation, ()).value
        if not math.isfinite(prediction) or not -1.0 <= prediction <= 1.0:
            raise ValueError("Learned cutoff produced an invalid value")
        if self.learned_weight == 1.0:
            return prediction
        baseline = sts2_value(observation)
        return self.learned_weight * prediction + (1.0 - self.learned_weight) * baseline
