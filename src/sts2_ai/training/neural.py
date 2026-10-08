"""Opt-in PyTorch trainer for the portable neural policy/value model.

Root search visits teach *policy*. Root value targets teach value only when no
independent cutoff labels were requested. If cutoff labels are supplied, the
value head learns from actual terminal outcomes of the named continuation
policy. Censored continuations NEVER become synthetic losses.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.hashed_linear import neural_action_features, state_dict, state_features
from sts2_ai.models.neural import NEURAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.training.continuations import ContinuationRecord

from .targets import TrainingExample


@dataclass(frozen=True, slots=True)
class NeuralMetrics:
    roots: int
    policy_cross_entropy: float
    policy_top1_accuracy: float
    root_value_rmse: float | None


def _as_observation(example: TrainingExample) -> Observation:
    return Observation(
        example.information_policy, example.observation_json, example.observation_hash
    )


def _as_actions(example: TrainingExample) -> tuple[LegalAction, ...]:
    return tuple(
        LegalAction(target.action_id, target.action_kind, target.action_payload_json)
        for target in example.policy_targets
    )


def _probabilities(example: TrainingExample) -> list[float]:
    probs = [max(0.0, target.probability) for target in example.policy_targets]
    if not probs:
        raise ValueError("Root training example has no legal actions")
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else [1.0 / len(probs)] * len(probs)


def _dense(features: dict[int, float], dimension: int) -> list[float]:
    result = [0.0] * dimension
    for index, value in features.items():
        result[index] = value
    return result


def _state_tensor(
    observation_json: str, *, dimension: int, torch: Any
) -> Any:
    state = state_dict(observation_json)
    return torch.tensor(_dense(state_features(state, dimension), dimension))


def _policy_tensors(
    example: TrainingExample, *, dimension: int, torch: Any
) -> Any:
    state = state_dict(example.observation_json)
    matrix = [
        _dense(neural_action_features(state, item.action_kind, item.action_payload_json, dimension),
               dimension)
        for item in example.policy_targets
    ]
    return torch.tensor(matrix)


def _new_params(dimension: int, hidden: int, torch: Any) -> dict[str, Any]:
    # Xavier-style initialization for the two feature projections.
    scale = math.sqrt(2.0 / (dimension + hidden))
    return {
        "state_weight": (torch.randn(hidden, dimension) * scale).requires_grad_(),
        "state_bias": torch.zeros(hidden, requires_grad=True),
        "action_weight": (torch.randn(hidden, dimension) * scale).requires_grad_(),
        "action_bias": torch.zeros(hidden, requires_grad=True),
        "policy_weight": (torch.randn(hidden) / math.sqrt(hidden)).requires_grad_(),
        "policy_bias": torch.zeros((), requires_grad=True),
        "value_weight": (torch.randn(hidden) / math.sqrt(hidden)).requires_grad_(),
        "value_bias": torch.zeros((), requires_grad=True),
    }


def _forward(
    params: dict[str, Any], state: Any, actions: Any, torch: Any
) -> tuple[Any, Any]:
    functional = torch.nn.functional
    hidden = functional.relu(functional.linear(
        state, params["state_weight"], params["state_bias"]
    ))
    value = torch.tanh(
        torch.dot(hidden, params["value_weight"]) + params["value_bias"]
    )
    if actions is None:
        return value, None
    action_hidden = functional.relu(
        hidden.unsqueeze(0)
        + functional.linear(actions, params["action_weight"], params["action_bias"])
    )
    logits = torch.mv(action_hidden, params["policy_weight"]) + params["policy_bias"]
    return value, logits


def _export(
    params: dict[str, Any], dimension: int, hidden: int, *, model_id: str
) -> NeuralPolicyValueModel:
    weights: dict[str, Any] = {}
    for name, value in params.items():
        weights[name] = value.detach().cpu().tolist()
    return NeuralPolicyValueModel.from_dict({
        "format": NEURAL_FORMAT,
        "dimension": dimension,
        "hidden": hidden,
        "model_id": model_id,
        **weights,
    })


def train_neural(
    roots: tuple[TrainingExample, ...],
    *,
    continuations: tuple[ContinuationRecord, ...] = (),
    dimension: int = 256,
    hidden: int = 32,
    epochs: int = 12,
    learning_rate: float = 0.002,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> NeuralPolicyValueModel:
    """Train with AdamW. PyTorch is imported only for this opt-in command."""
    if not roots:
        raise ValueError("Neural training needs at least one search-root policy example")
    if min(dimension, hidden, epochs) <= 0 or learning_rate <= 0.0:
        raise ValueError("Architecture, epochs and learning rate must be positive")
    if weight_decay < 0 or not math.isfinite(learning_rate):
        raise ValueError("Invalid optimizer settings")
    for record in continuations:
        if record.terminal_value is None:
            raise ValueError("Censored records must be filtered before training")
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Neural training requires PyTorch: pip install -e '.[neural]'"
        ) from exc

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    params = _new_params(dimension, hidden, torch)
    optimizer = torch.optim.AdamW(list(params.values()), lr=learning_rate,
                                  weight_decay=weight_decay)

    root_data = [
        (
            _state_tensor(ex.observation_json, dimension=dimension, torch=torch),
            _policy_tensors(ex, dimension=dimension, torch=torch),
            torch.tensor(_probabilities(ex)),
            torch.tensor(ex.value_target),
        )
        for ex in roots
    ]
    continuation_data = [
        (
            _state_tensor(rec.observation_json, dimension=dimension, torch=torch),
            torch.tensor(rec.terminal_value),
        )
        for rec in continuations
    ]
    steps = [("root", index) for index in range(len(root_data))]
    steps.extend(("cutoff", index) for index in range(len(continuation_data)))
    for _ in range(epochs):
        rng.shuffle(steps)
        for category, index in steps:
            optimizer.zero_grad()
            if category == "root":
                state, actions, probs, root_value = root_data[index]
                value, logits = _forward(params, state, actions, torch)
                policy_loss = -(probs * torch.nn.functional.log_softmax(logits, dim=0)).sum()
                if continuation_data:
                    loss = policy_loss
                else:
                    loss = policy_loss + (value - root_value).square()
            else:
                state, target = continuation_data[index]
                value, _ = _forward(params, state, None, torch)
                loss = (value - target).square()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm=5.0)
            optimizer.step()

    signature = hashlib.sha256(
        (
            f"{seed}|{dimension}|{hidden}|{epochs}|{learning_rate}|"
            f"{len(roots)}|{len(continuation_data)}"
        ).encode()
    ).hexdigest()[:12]
    return _export(
        params, dimension, hidden, model_id=f"neural-v2-d{dimension}-h{hidden}-{signature}"
    )


def evaluate_neural(
    model: NeuralPolicyValueModel,
    roots: tuple[TrainingExample, ...],
    *,
    compare_root_values: bool = True,
) -> NeuralMetrics:
    if not roots:
        raise ValueError("Neural evaluation requires search-root examples")
    policy_losses: list[float] = []
    root_errors: list[float] = []
    correct = 0
    for example in roots:
        prediction = model.evaluate(_as_observation(example), _as_actions(example))
        expected = _probabilities(example)
        if len(prediction.action_logits) != len(expected):
            raise RuntimeError("Model returned wrong number of legal action logits")
        largest = max(prediction.action_logits)
        exps = [math.exp(logit - largest) for logit in prediction.action_logits]
        norm = sum(exps)
        probabilities = [x / norm for x in exps]
        policy_losses.append(
            -sum(p * math.log(max(q, 1e-12)) for p, q in zip(expected, probabilities,
                                                             strict=True))
        )
        correct += (
            max(range(len(expected)), key=lambda i: prediction.action_logits[i])
            == max(range(len(expected)), key=lambda i: expected[i])
        )
        if compare_root_values:
            root_errors.append((prediction.value - example.value_target) ** 2)
    return NeuralMetrics(
        roots=len(roots),
        policy_cross_entropy=fmean(policy_losses),
        policy_top1_accuracy=correct / len(roots),
        root_value_rmse=(math.sqrt(fmean(root_errors)) if root_errors else None),
    )


def split_continuations_by_seed(
    records: tuple[ContinuationRecord, ...],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[tuple[ContinuationRecord, ...], tuple[ContinuationRecord, ...]]:
    """Hold out entire source runs, not individual exact/visible states.

    A source state shared between train and validation seed partitions must
    also be dropped from train to avoid exact/observation leakage.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must lie between zero and one")
    seeds = sorted({record.source_run_seed for record in records})
    if len(seeds) < 2:
        raise ValueError("At least two distinct source-run seeds are required")
    ordered = sorted(
        seeds, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).digest()
    )
    holdout_count = max(1, min(len(seeds) - 1, round(len(seeds) * validation_fraction)))
    heldout = set(ordered[:holdout_count])
    validation = tuple(rec for rec in records if rec.source_run_seed in heldout)
    exact_hashes = {rec.source_exact_hash for rec in validation}
    obs_hashes = {rec.observation_hash for rec in validation}
    training = tuple(
        rec for rec in records
        if rec.source_run_seed not in heldout
        and rec.source_exact_hash not in exact_hashes
        and rec.observation_hash not in obs_hashes
    )
    if not training or not validation:
        raise ValueError("Seed split leaves an empty training or validation partition")
    return training, validation
