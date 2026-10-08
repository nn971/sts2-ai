from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from sts2_ai.emulator import Observation
from sts2_ai.models import HashedLinearPolicyValueModel


class LearnedCutoffValue:
    """Opt-in learned value for nonterminal MCTS rollout cutoffs.

    The terminal game outcome always overrides the model. Policy logits are not
    used: the heuristic continues to choose rollout actions. The content hash
    fingerprints the *actual weight file*, not just a reusable model name.
    """

    def __init__(
        self,
        model: HashedLinearPolicyValueModel,
        *,
        model_sha256: str,
    ) -> None:
        if len(model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_sha256
        ):
            raise ValueError("model_sha256 must be a SHA-256 hexadecimal digest")
        self._model = model
        self.value_id = f"learned-linear-cutoff-v1-sha256-{model_sha256}"

    @classmethod
    def load(cls, path: Path) -> LearnedCutoffValue:
        payload = path.read_bytes()
        model = HashedLinearPolicyValueModel.load(path)
        return cls(model, model_sha256=hashlib.sha256(payload).hexdigest())

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

        prediction = self._model.evaluate(observation, ()).value
        if not math.isfinite(prediction) or not -1.0 <= prediction <= 1.0:
            raise ValueError("Learned cutoff produced an invalid value")
        return prediction
