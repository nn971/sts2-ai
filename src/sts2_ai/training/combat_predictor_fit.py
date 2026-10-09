"""Train short-horizon outcome heads from resolved public combat samples."""
from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any

from sts2_ai.models.hashed_linear import state_dict, state_features
from sts2_ai.training.combat_predictor import (
    CombatOutcomePredictor, OutcomeExample, _dense,
)


def split_by_run_seed(
    samples: Sequence[OutcomeExample], *, test_fraction: float = 0.2
) -> tuple[tuple[OutcomeExample, ...], tuple[OutcomeExample, ...]]:
    """Entire runs, not individual correlated combats, belong to one split."""
    if not 0 < test_fraction < 1:
        raise ValueError("Test fraction must lie strictly between zero and one")
    seeds = sorted({sample.seed for sample in samples}, key=lambda s: (
        hashlib.sha256(("combat-holdout-v1:" + s).encode()).hexdigest(), s
    ))
    if len(seeds) < 2:
        raise ValueError("Need at least two distinct run seeds for heldout evaluation")
    cut = max(1, min(len(seeds) - 1, int(round(len(seeds) * test_fraction))))
    heldout = set(seeds[:cut])
    train = tuple(sample for sample in samples if sample.seed not in heldout)
    test = tuple(sample for sample in samples if sample.seed in heldout)
    return train, test


def evaluation_metrics(
    model: CombatOutcomePredictor, examples: Sequence[OutcomeExample]
) -> dict[str, float]:
    if not examples:
        raise ValueError("Cannot evaluate on an empty cohort")
    survival_sq = []
    hp_abs = []
    hp_sq = []
    for example in examples:
        prediction = model.predict(example.entry_public_json)
        survival_sq.append((prediction.survival_probability - example.survived) ** 2)
        hp_abs.append(abs(prediction.expected_exit_hp_ratio - example.exit_hp_ratio))
        hp_sq.append((prediction.expected_exit_hp_ratio - example.exit_hp_ratio) ** 2)
    return {
        "survival_brier": fmean(survival_sq),
        "exit_hp_mae": fmean(hp_abs),
        "exit_hp_rmse": fmean(hp_sq) ** 0.5,
    }


def fit_outcome_predictor(
    examples: Sequence[OutcomeExample], *,
    dimension: int = 256, hidden: int = 64, epochs: int = 30,
    learning_rate: float = 0.003, seed: int = 41,
    progress: Callable[[int, float], None] | None = None,
) -> CombatOutcomePredictor:
    """Supervised auxiliary predictor; no tactical action policy update.

    Exit HP is modeled as a fraction of max HP, and never counted as a
    substitute for preserving a particular potion. Train on multiple seeds,
    with run-seed-disjoint validation performed by the caller.
    """
    if not examples or min(dimension, hidden, epochs) <= 0:
        raise ValueError("Need positive dimensions, epochs, and nonempty examples")
    if not 0 < learning_rate < 1:
        raise ValueError("Invalid learning rate")
    torch: Any = importlib.import_module("torch")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    xs = torch.tensor([
        _dense(state_features(state_dict(ex.entry_public_json), dimension), dimension)
        for ex in examples
    ], dtype=torch.float32)
    survival = torch.tensor([ex.survived for ex in examples], dtype=torch.float32)
    hp = torch.tensor([ex.exit_hp_ratio for ex in examples], dtype=torch.float32)
    # Independent heads can later be replaced by a joint resource distribution.
    import math

    trunk = torch.nn.Linear(dimension, hidden)
    survival_head = torch.nn.Linear(hidden, 1)
    hp_head = torch.nn.Linear(hidden, 1)
    with torch.no_grad():
        for layer in (trunk, survival_head, hp_head):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.zero_()
    parameters = [
        *trunk.parameters(), *survival_head.parameters(), *hp_head.parameters()
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    for epoch in range(1, epochs + 1):
        representation = torch.relu(trunk(xs))
        win_logit = survival_head(representation).squeeze(-1)
        hp_logit = hp_head(representation).squeeze(-1)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(win_logit, survival)
            + torch.nn.functional.mse_loss(torch.sigmoid(hp_logit), hp)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        if progress is not None:
            progress(epoch, float(loss.detach()))
    with torch.no_grad():
        return CombatOutcomePredictor(
            dimension=dimension,
            hidden=hidden,
            state_weight=trunk.weight.tolist(),
            state_bias=trunk.bias.tolist(),
            survival_weight=survival_head.weight[0].tolist(),
            survival_bias=float(survival_head.bias[0]),
            hp_weight=hp_head.weight[0].tolist(),
            hp_bias=float(hp_head.bias[0]),
            model_id=f"combat-outcome-v1-seed{seed}-epochs{epochs}",
        )


def constant_baseline_metrics(
    train: Sequence[OutcomeExample], test: Sequence[OutcomeExample],
) -> dict[str, float]:
    survival_mean = fmean(e.survived for e in train)
    hp_mean = fmean(e.exit_hp_ratio for e in train)
    return {
        "survival_brier": fmean((survival_mean - e.survived) ** 2 for e in test),
        "exit_hp_mae": fmean(abs(hp_mean - e.exit_hp_ratio) for e in test),
        "exit_hp_rmse": math.sqrt(fmean(
            (hp_mean - e.exit_hp_ratio) ** 2 for e in test
        )),
    }
