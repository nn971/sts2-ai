from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean

from sts2_ai.models.hashed_linear import (
    HashedLinearPolicyValueModel,
    dot,
    policy_features,
    state_dict,
    state_features,
)

from .targets import TrainingExample


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    examples: int
    policy_cross_entropy: float
    policy_top1_accuracy: float
    value_rmse: float


def train_hashed_linear(
    examples: tuple[TrainingExample, ...],
    *,
    dimension: int = 4096,
    epochs: int = 8,
    learning_rate: float = 0.03,
    l2: float = 1e-6,
    seed: int = 0,
) -> tuple[HashedLinearPolicyValueModel, TrainingMetrics]:
    """Train the dependency-free first policy/value baseline with SGD."""

    if not examples:
        raise ValueError("training examples must be non-empty")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    model = HashedLinearPolicyValueModel.zeros(
        dimension,
        model_id=f"hashed-linear-v1-d{dimension}-seed{seed}",
    )
    rng = random.Random(seed)
    order = list(range(len(examples)))

    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            _sgd_step(
                model,
                examples[index],
                learning_rate=learning_rate,
                l2=l2,
            )

    return model, evaluate_hashed_linear(model, examples)


def evaluate_hashed_linear(
    model: HashedLinearPolicyValueModel,
    examples: tuple[TrainingExample, ...],
) -> TrainingMetrics:
    if not examples:
        raise ValueError("evaluation examples must be non-empty")

    policy_losses = []
    value_squared_errors = []
    top1_correct = 0

    for example in examples:
        state = state_dict(example.observation_json)
        action_features = [
            policy_features(
                state,
                target.action_kind,
                target.action_payload_json,
                model.dimension,
            )
            for target in example.policy_targets
        ]
        logits = [
            dot(model.policy_weights, features)
            for features in action_features
        ]
        probabilities = _softmax(logits)
        targets = _normalized_targets(example)

        policy_losses.append(
            -sum(
                target * math.log(max(probability, 1e-12))
                for target, probability in zip(
                    targets,
                    probabilities,
                    strict=True,
                )
            )
        )
        predicted_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )
        target_index = max(
            range(len(targets)),
            key=targets.__getitem__,
        )
        top1_correct += predicted_index == target_index

        value_prediction = math.tanh(
            dot(
                model.value_weights,
                state_features(state, model.dimension),
            )
        )
        value_error = value_prediction - example.value_target
        value_squared_errors.append(value_error * value_error)

    return TrainingMetrics(
        examples=len(examples),
        policy_cross_entropy=fmean(policy_losses),
        policy_top1_accuracy=top1_correct / len(examples),
        value_rmse=math.sqrt(fmean(value_squared_errors)),
    )


def _sgd_step(
    model: HashedLinearPolicyValueModel,
    example: TrainingExample,
    *,
    learning_rate: float,
    l2: float,
) -> None:
    state = state_dict(example.observation_json)
    action_features = [
        policy_features(
            state,
            target.action_kind,
            target.action_payload_json,
            model.dimension,
        )
        for target in example.policy_targets
    ]
    logits = [
        dot(model.policy_weights, features)
        for features in action_features
    ]
    probabilities = _softmax(logits)
    targets = _normalized_targets(example)

    for probability, target, features in zip(
        probabilities,
        targets,
        action_features,
        strict=True,
    ):
        error = probability - target
        for feature_index, feature_value in features.items():
            weight = model.policy_weights[feature_index]
            gradient = (error * feature_value) + (l2 * weight)
            model.policy_weights[feature_index] = (
                weight - (learning_rate * gradient)
            )

    value_features = state_features(state, model.dimension)
    raw_value = dot(model.value_weights, value_features)
    predicted_value = math.tanh(raw_value)
    value_error = predicted_value - example.value_target
    value_gradient = value_error * (1.0 - (predicted_value * predicted_value))
    for feature_index, feature_value in value_features.items():
        weight = model.value_weights[feature_index]
        gradient = (value_gradient * feature_value) + (l2 * weight)
        model.value_weights[feature_index] = (
            weight - (learning_rate * gradient)
        )


def _normalized_targets(example: TrainingExample) -> tuple[float, ...]:
    if not example.policy_targets:
        raise ValueError("training example has no policy targets")
    positive = tuple(
        max(0.0, target.probability)
        for target in example.policy_targets
    )
    total = sum(positive)
    if total <= 0.0:
        uniform = 1.0 / len(positive)
        return tuple(uniform for _ in positive)
    return tuple(value / total for value in positive)


def _softmax(logits: list[float]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("softmax requires at least one logit")
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    total = sum(weights)
    return tuple(weight / total for weight in weights)
