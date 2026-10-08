from pathlib import Path

from .hashed_linear import HashedLinearPolicyValueModel
from .neural import NEURAL_FORMAT, NeuralPolicyValueModel
from .protocol import PolicyValueEstimate, PolicyValueModel

__all__ = [
    "HashedLinearPolicyValueModel",
    "NeuralPolicyValueModel",
    "NEURAL_FORMAT",
    "load_model",
    "PolicyValueEstimate",
    "PolicyValueModel",
]


def load_model(path: Path) -> HashedLinearPolicyValueModel | NeuralPolicyValueModel:
    """Load a known portable model format without running untrusted code."""
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model weights must be a JSON object")
    if raw.get("format") == NEURAL_FORMAT:
        return NeuralPolicyValueModel.load(path)
    return HashedLinearPolicyValueModel.load(path)
