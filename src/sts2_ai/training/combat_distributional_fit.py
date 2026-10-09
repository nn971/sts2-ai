"""Fit and validate joint survival / coarse surviving-HP distributions.

Full trajectories remain the source of truth; this helper is strictly
supervised from public combat-entry observations with known combat outcomes.
Heldout splitting is delegated to the existing run-seed-disjoint splitter.
No training target depends on the hidden RNG seed or exact game state.
"""
from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any

from sts2_ai.models.hashed_linear import state_dict, state_features
from sts2_ai.training.combat_entry_features import (
    FEATURE_SCHEMA, combat_entry_features,
)
from sts2_ai.training.combat_distributional_predictor import (
    HP_BINS,
    DistributionalCombatOutcomePredictor,
    hp_bin,
)
from sts2_ai.training.combat_predictor import OutcomeExample, _dense


def fit_distributional_predictor(
    examples: Sequence[OutcomeExample], *,
    dimension: int = 256,
    hidden: int = 64,
    epochs: int = 30,
    learning_rate: float = 0.003,
    seed: int = 41, feature_schema: str = FEATURE_SCHEMA,
    progress: Callable[[int, float], None] | None = None,
) -> DistributionalCombatOutcomePredictor:
    if not examples or min(dimension, hidden, epochs) <= 0:
        raise ValueError("Need examples and positive model sizes/epochs")
    if not 0 < learning_rate < 1:
        raise ValueError("Invalid learning rate")
    if feature_schema not in ("legacy-hashed-v1", FEATURE_SCHEMA):
        raise ValueError("Unsupported combat feature encoder")
    if feature_schema == FEATURE_SCHEMA and dimension <= 24:
        raise ValueError("Structured encoder requires dimension >24")
    feature_fn = (
        combat_entry_features if feature_schema == FEATURE_SCHEMA
        else state_features
    )
    torch: Any = importlib.import_module("torch")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    xs = torch.tensor([
        _dense(feature_fn(state_dict(example.entry_public_json), dimension),
               dimension) for example in examples
    ], dtype=torch.float32)
    survival = torch.tensor([
        float(example.survived) for example in examples
    ], dtype=torch.float32)
    alive_mask = survival > 0.5
    alive_bins = torch.tensor([
        hp_bin(example.exit_hp_ratio)
        for example in examples if example.survived > 0.5
    ], dtype=torch.long)

    trunk = torch.nn.Linear(dimension, hidden)
    survive_head = torch.nn.Linear(hidden, 1)
    hp_head = torch.nn.Linear(hidden, HP_BINS)
    with torch.no_grad():
        for layer in (trunk, survive_head, hp_head):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.zero_()
    parameters = [
        *trunk.parameters(), *survive_head.parameters(), *hp_head.parameters()
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

    for epoch in range(1, epochs + 1):
        representation = torch.relu(trunk(xs))
        survive_logits = survive_head(representation).squeeze(-1)
        hp_logits = hp_head(representation)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            survive_logits, survival
        )
        if alive_bins.numel():
            # HP histogram is CONDITIONAL on surviving. Defeats are
            # represented by the survival head; no double-counted HP labels.
            loss = loss + torch.nn.functional.cross_entropy(
                hp_logits[alive_mask], alive_bins
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        if progress is not None:
            progress(epoch, float(loss.detach()))
    with torch.no_grad():
        return DistributionalCombatOutcomePredictor(
            dimension=dimension,
            hidden=hidden,
            state_weight=trunk.weight.tolist(),
            state_bias=trunk.bias.tolist(),
            survival_weight=survive_head.weight[0].tolist(),
            survival_bias=float(survive_head.bias[0]),
            hp_bin_weight=hp_head.weight.tolist(),
            hp_bin_bias=hp_head.bias.tolist(),
            model_id=(f"combat-distribution-v2-{feature_schema}-"
                      f"seed{seed}-epochs{epochs}"),
            feature_schema=feature_schema,
        )


def distributional_metrics(
    model: DistributionalCombatOutcomePredictor,
    examples: Sequence[OutcomeExample],
) -> dict[str, float]:
    if not examples:
        raise ValueError("Cannot evaluate empty cohort")
    brier: list[float] = []
    hp_abs: list[float] = []
    hp_sq: list[float] = []
    nll: list[float] = []
    for example in examples:
        pred = model.predict(example.entry_public_json)
        brier.append((pred.survival_probability - example.survived) ** 2)
        hp_abs.append(
            abs(pred.expected_exit_hp_fraction - example.exit_hp_ratio)
        )
        hp_sq.append(
            (pred.expected_exit_hp_fraction - example.exit_hp_ratio) ** 2
        )
        category = (
            0 if example.survived < 0.5
            else hp_bin(example.exit_hp_ratio) + 1
        )
        nll.append(-math.log(max(1e-12, pred.joint_hp_probabilities[category])))
    return {
        "survival_brier": fmean(brier),
        "exit_hp_mae": fmean(hp_abs),
        "exit_hp_rmse": fmean(hp_sq) ** 0.5,
        "joint_hp_negative_log_likelihood": fmean(nll),
    }


def constant_distributional_baseline(
    train: Sequence[OutcomeExample], test: Sequence[OutcomeExample]
) -> dict[str, float]:
    """Laplace-smoothed training-frequency baseline on the same heldout runs."""
    if not train or not test:
        raise ValueError("Need nonempty training and heldout samples")
    won = [item for item in train if item.survived > 0.5]
    survival = (len(won) + 0.5) / (len(train) + 1)
    counts = [0] * HP_BINS
    for item in won:
        counts[hp_bin(item.exit_hp_ratio)] += 1
    hp_cond = [
        (count + 0.5) / (len(won) + HP_BINS * 0.5)
        for count in counts
    ]
    joint = [1 - survival] + [survival * p for p in hp_cond]
    mean = sum(
        joint[i + 1] * ((i + 0.5) / HP_BINS)
        for i in range(HP_BINS)
    )
    brier = [(survival - item.survived) ** 2 for item in test]
    hp_abs = [abs(mean - item.exit_hp_ratio) for item in test]
    hp_sq = [(mean - item.exit_hp_ratio) ** 2 for item in test]
    nll = [
        -math.log(max(1e-12, joint[
            0 if item.survived < 0.5 else hp_bin(item.exit_hp_ratio) + 1
        ])) for item in test
    ]
    return {
        "survival_brier": fmean(brier),
        "exit_hp_mae": fmean(hp_abs),
        "exit_hp_rmse": fmean(hp_sq) ** 0.5,
        "joint_hp_negative_log_likelihood": fmean(nll),
    }
