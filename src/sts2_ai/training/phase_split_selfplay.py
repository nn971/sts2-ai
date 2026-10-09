"""Experimental teacher-free phase-separated actor–critic.

Strategy and combat have independent policy/value weights and independent
optimizers. A combat's advantage may use the FROZEN strategy critic's
post-combat value, including remaining named potions, rather than a constant
HP-per-potion penalty. The auxiliary outcome model is trained separately.

This is a conservative on-policy REINFORCE experiment, not PPO/GAE and not
yet a complete hierarchical actor–critic. Full-run outcome remains the
dominant target until a boundary critic is calibrated.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import EmulatorBackend, InformationPolicy, JsonlEmulatorBackend, Observation
from sts2_ai.emulator.run_environment import LEGACY, require_environment
from sts2_ai.models.hashed_linear import (
    neural_action_features, state_dict, state_features, tactical_action_features,
)
from sts2_ai.models.neural import TACTICAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.neural import _dense, _export, _forward, _new_params
from sts2_ai.training.parallel_rollouts import (
    collect_parallel, episode_actor_seed, open_worker_pool,
)
from sts2_ai.training.selfplay import (
    Episode, PublicDecision, _bounded_return, _curriculum_coefficient,
    collect_public_episode, temperature_for_round,
)
from contextlib import ExitStack

SPLIT_TRAINING_VERSION = "sts2-phase-split-reinforce-v1"
SPLIT_CHECKPOINT_FORMAT = "sts2-phase-split-training-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class PhaseSplitRound:
    round_index: int
    played: int
    completed: int
    censored: int
    wins: int
    combat_victories: int
    combat_defeats: int
    strategic_decisions: int
    tactical_decisions: int
    optimization_steps: int
    mean_run_return: float | None
    mean_combat_boundary_target: float | None
    mean_loss: float | None
    rollout_seconds: float
    optimizer_seconds: float


def _forward_decision(
    decision: PublicDecision, params: dict[str, Any],
    dimension: int, torch: Any,
) -> tuple[Any, Any]:
    state = state_dict(decision.observation.payload_json)
    state_x = torch.tensor(
        _dense(state_features(state, dimension), dimension),
        dtype=torch.float32,
    )
    feature_fn = (
        tactical_action_features if decision.phase == "combat"
        else neural_action_features
    )
    actions_x = torch.tensor([
        _dense(feature_fn(state, a.kind, a.payload_json, dimension), dimension)
        for a in decision.legal_actions
    ], dtype=torch.float32)
    return _forward(params, state_x, actions_x, torch)


def _export_split(
    strategy_params: dict[str, Any], combat_params: dict[str, Any],
    dimension: int, hidden: int, *, model_id: str, trained: bool,
) -> PhaseSplitNeuralModel:
    return PhaseSplitNeuralModel(
        strategy=_export(
            strategy_params, dimension, hidden,
            model_id=model_id + "-strategy", value_head_trained=trained,
        ),
        combat=_export(
            combat_params, dimension, hidden,
            model_id=model_id + "-combat", value_head_trained=trained,
            format_id=TACTICAL_FORMAT,
        ),
        model_id=model_id,
    )


def _combat_boundary_target(
    episode: Episode, segment_index: int, model: PhaseSplitNeuralModel,
    run_return: float, *, boundary_weight: float,
) -> float:
    segment = episode.combat_outcomes[segment_index]
    if segment.result == "defeat":
        # An actual terminal defeat has no future continuation value.
        continuation = 0.0
    else:
        if not segment.exit_public_json:
            raise ValueError("Combat lacks a public exit frame for bootstrapping")
        observation = Observation(
            "prototype-fair-v0", segment.exit_public_json, "",
        )
        # This V is a run-continuation estimate conditioned on the actual
        # remaining potions/HP. The tactical learner cannot read hidden state.
        continuation = max(0.0, min(1.0, model.strategy.evaluate(
            observation, (),
        ).value))
    return (1.0 - boundary_weight) * run_return + boundary_weight * continuation


def _checkpoint_save(
    path: Path, *, torch: Any,
    params: dict[str, dict[str, Any]], optimizers: dict[str, Any],
    fingerprint: str, revision: str, rows: list[PhaseSplitRound],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format": SPLIT_CHECKPOINT_FORMAT,
        "fingerprint": fingerprint,
        "emulator_revision": revision,
        "params": {
            phase: {key: tensor.detach().cpu().clone()
                    for key, tensor in values.items()}
            for phase, values in params.items()
        },
        "optimizers": {phase: o.state_dict() for phase, o in optimizers.items()},
        "torch_rng_state": torch.get_rng_state(),
        "rows": [asdict(row) for row in rows],
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, suffix=".pending", delete=False,
        ) as output:
            temporary = Path(output.name)
            torch.save(record, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_phase_split(
    backend: EmulatorBackend, *,
    rounds: int, episodes_per_round: int,
    dimension: int = 128, hidden: int = 32, learning_rate: float = 0.003,
    entropy_weight: float = 0.002, value_weight: float = 0.5,
    auxiliary_weight: float = 0.4, boundary_weight: float = 0.25,
    win_anneal_threshold: int = 16, max_decisions: int = 4096,
    temperature_start: float = 0.05, temperature_end: float = 0.035,
    temperature_decay_rounds: int = 20,
    seed: int = 19, seed_prefix: str = "phase-split-v1",
    workers: int = 1, environment: str = LEGACY,
    warm_start: Path | None = None,
    checkpoint: Path | None = None, resume: bool = False,
    progress: Callable[[PhaseSplitRound], None] | None = None,
) -> tuple[PhaseSplitNeuralModel, tuple[PhaseSplitRound, ...]]:
    if min(rounds, episodes_per_round, dimension, hidden, max_decisions, workers) <= 0:
        raise ValueError("Expected positive sizes and worker count")
    if not 0.0 <= boundary_weight <= 1.0:
        raise ValueError("Boundary bootstrapping weight must lie in [0, 1]")
    if not (0.0 <= entropy_weight <= 1.0 and 0.0 <= value_weight <= 10.0):
        raise ValueError("Invalid loss coefficients")
    if not 0 < learning_rate < 1:
        raise ValueError("Invalid learning rate")
    if resume and checkpoint is None:
        raise ValueError("Resume requires checkpoint")
    if workers > 1 and not isinstance(backend, JsonlEmulatorBackend):
        raise ValueError("Parallel training requires the JSONL emulator")
    require_environment(backend, environment)
    torch: Any = importlib.import_module("torch")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    params = {
        "strategy": _new_params(dimension, hidden, torch),
        "combat": _new_params(dimension, hidden, torch),
    }
    optimizers = {
        phase: torch.optim.AdamW(list(weights.values()), lr=learning_rate)
        for phase, weights in params.items()
    }
    warm_sha: str | None = None
    if warm_start is not None:
        raw_bytes = warm_start.read_bytes()
        warm_sha = hashlib.sha256(raw_bytes).hexdigest()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, dict):
            raise ValueError("Warm-start model must be an object")
        if raw.get("format") == "sts2-phase-split-policy-value-v1":
            warm = PhaseSplitNeuralModel.from_dict(raw)
            sources = {"strategy": warm.strategy, "combat": warm.combat}
        else:
            # The old semantic-action format can warm-start strategy. Its
            # action encoder differs from combat-v3; don't transplant its
            # incompatible combat action weights.
            base = NeuralPolicyValueModel.from_dict(raw)
            sources = {"strategy": base}
        for phase, source in sources.items():
            if (source.dimension, source.hidden) != (dimension, hidden):
                raise ValueError("Warm-start dimensions must match")
            with torch.no_grad():
                for name, weight in params[phase].items():
                    weight.copy_(torch.tensor(
                        getattr(source, name), dtype=weight.dtype
                    ))
    revision = getattr(backend, "emulator_revision", "test-backend")
    config = {
        "version": SPLIT_TRAINING_VERSION,
        "episodes_per_round": episodes_per_round,
        "dimension": dimension, "hidden": hidden,
        "learning_rate": learning_rate, "entropy_weight": entropy_weight,
        "value_weight": value_weight, "auxiliary_weight": auxiliary_weight,
        "boundary_weight": boundary_weight,
        "win_anneal_threshold": win_anneal_threshold,
        "max_decisions": max_decisions, "temperature_start": temperature_start,
        "temperature_end": temperature_end,
        "temperature_decay_rounds": temperature_decay_rounds,
        "seed": seed, "seed_prefix": seed_prefix, "environment": environment,
        "warm_sha": warm_sha,
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    rows: list[PhaseSplitRound] = []
    if resume:
        assert checkpoint is not None
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("format") != SPLIT_CHECKPOINT_FORMAT:
            raise ValueError("Wrong phase-split checkpoint schema")
        if saved.get("fingerprint") != fingerprint or saved.get(
            "emulator_revision"
        ) != revision:
            raise ValueError("Checkpoint configuration/emulator mismatch")
        with torch.no_grad():
            for phase, values in params.items():
                old = saved["params"][phase]
                if set(old) != set(values):
                    raise ValueError("Checkpoint parameter mismatch")
                for name, target in values.items():
                    if old[name].shape != target.shape:
                        raise ValueError("Checkpoint tensor shape mismatch")
                    target.copy_(old[name])
        for phase, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][phase])
        torch.set_rng_state(saved["torch_rng_state"])
        rows = [PhaseSplitRound(**row) for row in saved["rows"]]
        if any(row.round_index != i for i, row in enumerate(rows)):
            raise ValueError("Checkpoint rounds are not contiguous")
        if len(rows) > rounds:
            raise ValueError("Checkpoint has more rounds than requested")
    victories = sum(row.wins for row in rows)

    with ExitStack() as scope:
        pool = (
            scope.enter_context(open_worker_pool(backend, workers))
            if isinstance(backend, JsonlEmulatorBackend) and workers > 1
            else None
        )
        for index in range(len(rows), rounds):
            temperature = temperature_for_round(
                index, start=temperature_start, end=temperature_end,
                decay_rounds=temperature_decay_rounds,
            )
            model = _export_split(
                params["strategy"], params["combat"], dimension, hidden,
                model_id=f"phase-split-before-round-{index}", trained=index > 0,
            )
            seeds = [
                f"{seed_prefix}-{index}-{j}" for j in range(episodes_per_round)
            ]
            start = time.perf_counter()
            if pool is None:
                cohort = tuple(collect_public_episode(
                    backend, model, seed=run_seed,
                    actor_rng=random.Random(episode_actor_seed(seed, run_seed)),
                    max_decisions=max_decisions, environment=environment,
                    temperature=temperature,
                ) for run_seed in seeds)
            else:
                cohort = collect_parallel(
                    pool, model, seeds, base_seed=seed,
                    max_decisions=max_decisions, policy_id="prototype-fair-v0",
                    environment=environment, temperature=temperature,
                )
            rollout_seconds = time.perf_counter() - start
            completed = [ep for ep in cohort if ep.completed]
            alpha = _curriculum_coefficient(
                auxiliary_weight, victories, win_anneal_threshold
            )
            run_targets = [(ep, _bounded_return(ep, alpha)) for ep in completed]
            optimizer_started = time.perf_counter()
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            strategic_count = tactical_count = 0
            combat_targets: list[float] = []
            losses: list[float] = []
            for episode, run_target in run_targets:
                # Targets for the entire cohort use the same frozen policy
                # snapshot, even though the optimizer is subsequently updated.
                targets = [run_target] * len(episode.decisions)
                for j, segment in enumerate(episode.combat_outcomes):
                    local_target = _combat_boundary_target(
                        episode, j, model, run_target,
                        boundary_weight=boundary_weight,
                    )
                    combat_targets.append(local_target)
                    for d in range(segment.start_decision, segment.end_decision):
                        if episode.decisions[d].phase != "combat":
                            raise ValueError("Combat segment includes a strategy decision")
                        targets[d] = local_target
                policy_terms = []
                value_terms = []
                entropy_terms = []
                for decision, target in zip(episode.decisions, targets, strict=True):
                    phase = decision.phase
                    if phase not in params:
                        raise ValueError("Unknown decision phase")
                    if phase == "combat":
                        tactical_count += 1
                    else:
                        strategic_count += 1
                    value, logits = _forward_decision(
                        decision, params[phase], dimension, torch,
                    )
                    log_probs = torch.nn.functional.log_softmax(
                        logits / temperature, dim=0,
                    )
                    probs = log_probs.exp()
                    advantage = torch.tensor(target, dtype=torch.float32) - value.detach()
                    policy_terms.append(-log_probs[decision.chosen_index] * advantage)
                    value_terms.append(
                        (value - torch.tensor(target, dtype=torch.float32)).square()
                    )
                    entropy_terms.append(-(probs * log_probs).sum())
                if policy_terms:
                    loss = (
                        torch.stack(policy_terms).sum()
                        + value_weight * torch.stack(value_terms).mean()
                        - entropy_weight * torch.stack(entropy_terms).mean()
                    ) / max(1, len(run_targets))
                    loss.backward()
                    losses.append(float(loss.detach()))
            steps = 0
            if strategic_count or tactical_count:
                for phase, count in (
                    ("strategy", strategic_count), ("combat", tactical_count)
                ):
                    if count:
                        torch.nn.utils.clip_grad_norm_(
                            list(params[phase].values()), max_norm=5.0
                        )
                        optimizers[phase].step()
                        steps += 1
            victories += sum(ep.won for ep in completed)
            row = PhaseSplitRound(
                round_index=index, played=episodes_per_round,
                completed=len(completed),
                censored=episodes_per_round - len(completed),
                wins=sum(ep.won for ep in completed),
                combat_victories=sum(
                    o.result == "victory" for ep in cohort for o in ep.combat_outcomes
                ),
                combat_defeats=sum(
                    o.result == "defeat" for ep in cohort for o in ep.combat_outcomes
                ),
                strategic_decisions=strategic_count,
                tactical_decisions=tactical_count,
                optimization_steps=steps,
                mean_run_return=(
                    sum(value for _, value in run_targets) / len(run_targets)
                    if run_targets else None
                ),
                mean_combat_boundary_target=(
                    sum(combat_targets) / len(combat_targets)
                    if combat_targets else None
                ),
                mean_loss=sum(losses) if losses else None,
                rollout_seconds=rollout_seconds,
                optimizer_seconds=time.perf_counter() - optimizer_started,
            )
            rows.append(row)
            if checkpoint is not None:
                _checkpoint_save(
                    checkpoint, torch=torch, params=params,
                    optimizers=optimizers, fingerprint=fingerprint,
                    revision=revision, rows=rows,
                )
            if progress is not None:
                progress(row)
    return _export_split(
        params["strategy"], params["combat"], dimension, hidden,
        model_id=f"phase-split-v1-{fingerprint[:12]}-{rounds}",
        trained=bool(rows and any(x.optimization_steps for x in rows)),
    ), tuple(rows)
