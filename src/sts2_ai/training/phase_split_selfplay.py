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
import os
import random
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import EmulatorBackend, JsonlEmulatorBackend, Observation
from sts2_ai.emulator.run_environment import LEGACY, require_environment
from sts2_ai.models.hashed_linear import (
    neural_action_features,
    state_dict,
    state_features,
    tactical_action_features,
    public_resources_tactical_action_features,
)
from sts2_ai.models.neural import (
    TACTICAL_DAMAGE_FORMAT,
    TACTICAL_RESOURCES_FORMAT,
    TACTICAL_FORMAT,
    TACTICAL_RELATIONAL_FORMAT,
    TACTICAL_STRUCTURED_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.tactical_state import (
    damage_tactical_state_features,
    public_resources_tactical_state_features,
    relational_tactical_state_features,
    tactical_state_features,
)
from sts2_ai.training.combat_outcomes import CombatOutcome
from sts2_ai.training.neural import _dense, _export, _forward, _new_params
from sts2_ai.training.parallel_rollouts import (
    collect_parallel,
    episode_actor_seed,
    open_worker_pool,
)
from sts2_ai.training.selfplay import (
    Episode,
    PublicDecision,
    _bounded_return,
    _curriculum_coefficient,
    collect_public_episode,
    temperature_for_round,
)

SPLIT_TRAINING_VERSION = "sts2-phase-split-reinforce-v2-hp-monotonicity"
BALANCED_TRAINING_VERSION = "sts2-phase-split-reinforce-v3-phase-mean"
PPO_TRAINING_VERSION = "sts2-phase-split-tactical-gae-strategy-mc-ppo-v1"
SPLIT_CHECKPOINT_FORMAT = "sts2-phase-split-training-checkpoint-v1"
HP_FIRST_TACTICAL_OBJECTIVE = "hp_first"


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
    hp_monotonic_pairs: int = 0
    phase_loss_diagnostics: dict[str, dict[str, float]] | None = None
    mean_victory_exit_hp_fraction: float | None = None
    mean_potions_used_per_combat: float | None = None
    mean_victory_hp_lost_fraction: float | None = None


def _higher_hp_public_pair(
    public_json: str, step: int = 5,
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    """Critic-only weak HP dominance pair; inventory and every other field fixed.

    This is not a feasible simulator transition, not a teacher move, and never
    reaches the policy as an observation. Low-HP relic thresholds may create
    exceptions, so monotonicity is a *soft* auxiliary constraint.
    """
    state = json.loads(public_json)
    if not isinstance(state, dict):
        raise ValueError("Malformed post-combat public observation")
    hp = state.get("hp")
    max_hp = state.get("max_hp")
    if (
        not isinstance(hp, int) or isinstance(hp, bool)
        or not isinstance(max_hp, int) or isinstance(max_hp, bool)
        or max_hp <= 0 or hp <= 0 or hp >= max_hp
    ):
        return None
    higher_hp = min(max_hp, hp + step)
    higher = dict(state)
    higher["hp"] = higher_hp
    return state, higher, (higher_hp - hp) / max_hp


def _forward_decision(
    decision: PublicDecision, params: dict[str, Any],
    dimension: int, torch: Any, *,
    tactical_state_encoding: str = "legacy",
) -> tuple[Any, Any]:
    state = state_dict(decision.observation.payload_json)
    feature_fn_state = (
        public_resources_tactical_state_features
        if decision.phase == "combat" and tactical_state_encoding == "public_resources"
        else damage_tactical_state_features
        if decision.phase == "combat" and tactical_state_encoding == "relational_damage"
        else relational_tactical_state_features
        if decision.phase == "combat" and tactical_state_encoding == "relational"
        else tactical_state_features
        if decision.phase == "combat" and tactical_state_encoding == "structured"
        else state_features
    )
    state_x = torch.tensor(
        _dense(feature_fn_state(state, dimension), dimension),
        dtype=torch.float32,
    )
    feature_fn = (
        public_resources_tactical_action_features
        if decision.phase == "combat" and tactical_state_encoding == "public_resources"
        else tactical_action_features if decision.phase == "combat"
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
    tactical_state_encoding: str = "legacy",
    combat_objective: str = "continuation",
) -> PhaseSplitNeuralModel:
    return PhaseSplitNeuralModel(
        strategy=_export(
            strategy_params, dimension, hidden,
            model_id=model_id + "-strategy", value_head_trained=trained,
        ),
        combat=_export(
            combat_params, dimension, hidden,
            model_id=model_id + "-combat", value_head_trained=trained,
            format_id=(
                TACTICAL_RESOURCES_FORMAT
                if tactical_state_encoding == "public_resources" else
                TACTICAL_DAMAGE_FORMAT
                if tactical_state_encoding == "relational_damage" else
                TACTICAL_RELATIONAL_FORMAT
                if tactical_state_encoding == "relational" else
                TACTICAL_STRUCTURED_FORMAT
                if tactical_state_encoding == "structured" else TACTICAL_FORMAT
            ),
        ),
        model_id=model_id,
        combat_value_objective=combat_objective,
    )


def _hp_first_combat_target(segment: CombatOutcome) -> float:
    """Temporary tactical target: survive and preserve HP; potions cost zero.

    Defeat = 0; victory = 0.5 + 0.5 * (exit HP / exit max HP).
    Survival always outranks any HP-preservation difference on a victory.
    No fixed value, sign, or penalty is assigned to named potions.
    """
    if segment.result == "defeat":
        return 0.0
    hp = segment.exit.hp
    max_hp = segment.exit.max_hp or segment.entry.max_hp
    if hp is None or max_hp is None or max_hp <= 0:
        raise ValueError("HP-first objective requires public victory exit HP/max HP")
    return 0.5 + 0.5 * min(1.0, max(0.0, hp / max_hp))


def _hp_preservation_target(segment: CombatOutcome) -> float:
    """Survival + *within-combat* net HP preservation, no potion penalty.

    Unlike hp_first, the reward does not favor encounters that happen to
    start with more HP. Healing counts as preservation, capped at no net
    loss. Defeat remains strictly worse than any victorious exit.
    """
    if segment.result == "defeat":
        return 0.0
    entry, exit_hp = segment.entry.hp, segment.exit.hp
    max_hp = segment.exit.max_hp or segment.entry.max_hp
    if entry is None or exit_hp is None or max_hp is None or max_hp <= 0:
        raise ValueError("HP preservation requires public entry/exit HP and max HP")
    lost_fraction = min(1.0, max(0.0, (entry - exit_hp) / max_hp))
    return 1.0 - 0.5 * lost_fraction


def _other_run_combat_baselines(
    samples: list[list[tuple[CombatOutcome, float]]],
) -> list[list[float]]:
    """Encounter-matched mean from OTHER runs; 0.5 if no match exists.

    Excluding the entire current run, rather than merely one combat,
    ensures its actions cannot influence the baseline through future
    combats on the same trajectory. A victory earns at least 0.5 and
    a defeat zero, so the data-independent fallback is meaningful.
    """
    from collections import defaultdict

    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for group in samples:
        for combat, reward in group:
            grouped[tuple(sorted(combat.enemy_ids))].append(reward)
    baselines: list[list[float]] = []
    for group in samples:
        own: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for combat, reward in group:
            own[tuple(sorted(combat.enemy_ids))].append(reward)
        values: list[float] = []
        for combat, _ in group:
            key = tuple(sorted(combat.enemy_ids))
            n_other = len(grouped[key]) - len(own[key])
            values.append(
                (sum(grouped[key]) - sum(own[key])) / n_other
                if n_other > 0 else 0.5
            )
        baselines.append(values)
    return baselines


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
    hp_monotonic_weight: float = 0.2,
    tactical_state_encoding: str = "legacy",
    loss_normalization: str = "legacy_episode_sum",
    optimizer_method: str = "reinforce",
    ppo_epochs: int = 3,
    ppo_batch_size: int = 128,
    ppo_sample_limit: int = 4096,
    ppo_clip_epsilon: float = 0.2,
    ppo_gamma: float = 0.995,
    ppo_gae_lambda: float = 0.98,
    combat_objective: str = "continuation",
    combat_advantage_baseline: str = "critic",
    win_anneal_threshold: int = 16, max_decisions: int = 4096,
    temperature_start: float = 0.05, temperature_end: float = 0.035,
    temperature_decay_rounds: int = 20,
    seed: int = 19, seed_prefix: str = "phase-split-v1",
    workers: int = 1, environment: str = LEGACY,
    warm_start: Path | None = None,
    checkpoint: Path | None = None, resume: bool = False,
    combat_samples_dir: Path | None = None,
    progress: Callable[[PhaseSplitRound], None] | None = None,
    excluded_training_seeds: frozenset[str] = frozenset(),
) -> tuple[PhaseSplitNeuralModel, tuple[PhaseSplitRound, ...]]:
    if min(rounds, episodes_per_round, dimension, hidden, max_decisions, workers) <= 0:
        raise ValueError("Expected positive sizes and worker count")
    if combat_objective not in ("continuation", "hp_first", "hp_preservation"):
        raise ValueError("Unknown combat objective")
    if combat_advantage_baseline not in ("critic", "leave_one_run_out"):
        raise ValueError("Unknown combat advantage baseline")
    if combat_advantage_baseline == "leave_one_run_out" and (
        combat_objective == "continuation"
    ):
        raise ValueError("Leave-one-run-out requires an HP-based combat objective")
    if loss_normalization not in ("legacy_episode_sum", "phase_mean"):
        raise ValueError("Loss normalization must be legacy_episode_sum or phase_mean")
    if optimizer_method not in ("reinforce", "ppo"):
        raise ValueError("Unknown phase-split optimizer")
    if optimizer_method == "ppo":
        if combat_advantage_baseline != "critic":
            raise ValueError("PPO uses its own GAE critic baseline, not leave-one-run-out")
        if hp_monotonic_weight:
            raise ValueError("PPO currently requires hp_monotonic_weight=0")
        if min(ppo_epochs, ppo_batch_size, ppo_sample_limit) <= 0:
            raise ValueError("PPO epoch/batch/sample sizes must be positive")
        if not 0 < ppo_clip_epsilon < 1:
            raise ValueError("Invalid PPO clipping")
        if not 0 < ppo_gamma <= 1 or not 0 <= ppo_gae_lambda <= 1:
            raise ValueError("Invalid PPO gamma/lambda")
    if tactical_state_encoding not in (
        "legacy", "structured", "relational", "relational_damage",
        "public_resources",
    ):
        raise ValueError("Unsupported tactical state encoding")
    if tactical_state_encoding == "structured" and dimension <= 32:
        raise ValueError("Structured tactical state requires dimension > 32")
    if tactical_state_encoding == "relational" and dimension <= 36:
        raise ValueError("Relational tactical state requires dimension > 36")
    if tactical_state_encoding == "relational_damage" and dimension <= 40:
        raise ValueError("Damage-aware tactical state requires dimension > 40")
    if tactical_state_encoding == "public_resources" and dimension <= 40:
        raise ValueError("Public-resource tactical state requires dimension > 40")
    if not 0.0 <= boundary_weight <= 1.0:
        raise ValueError("Boundary bootstrapping weight must lie in [0, 1]")
    if not 0.0 <= hp_monotonic_weight <= 1.0:
        raise ValueError("HP monotonicity regularization must lie in [0, 1]")
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
            previous_combat_objective = warm.combat_value_objective
        else:
            # Tactical-v3 extends v2 semantic tokens with target context.
            # Both heads warm-start approximately: normalization changes with
            # extra features, so the action distributions are not identical.
            base = NeuralPolicyValueModel.from_dict(raw)
            sources = {"strategy": base, "combat": base}
            previous_combat_objective = "continuation"
        expected_tactical_format = (
            TACTICAL_RESOURCES_FORMAT
            if tactical_state_encoding == "public_resources" else
            TACTICAL_DAMAGE_FORMAT
            if tactical_state_encoding == "relational_damage" else
            TACTICAL_RELATIONAL_FORMAT
            if tactical_state_encoding == "relational" else
            TACTICAL_STRUCTURED_FORMAT
            if tactical_state_encoding == "structured" else TACTICAL_FORMAT
        )
        for phase, source in sources.items():
            if (source.dimension, source.hidden) != (dimension, hidden):
                raise ValueError("Warm-start dimensions must match")
            # The old state projection weights refer to DIFFERENT input
            # coordinates. Reusing them would silently corrupt the tactical
            # policy. Retain compatible action/policy parameters, but reset
            # state and value projections across a format migration.
            migrating_tactical = (
                phase == "combat" and source.format_id != expected_tactical_format
            )
            migrating_combat_value = (
                phase == "combat" and previous_combat_objective != combat_objective
            )
            with torch.no_grad():
                for name, weight in params[phase].items():
                    if migrating_tactical and name in (
                        "state_weight", "state_bias", "value_weight", "value_bias"
                    ):
                        continue
                    if migrating_combat_value and name in (
                        "value_weight", "value_bias"
                    ):
                        # The prior tactical critic estimates a different
                        # quantity, so its weights must not be silently reused.
                        continue
                    weight.copy_(torch.tensor(
                        getattr(source, name), dtype=weight.dtype
                    ))
    revision = getattr(backend, "emulator_revision", "test-backend")
    config = {
        "version": (
            BALANCED_TRAINING_VERSION if loss_normalization == "phase_mean"
            else SPLIT_TRAINING_VERSION
        ),
        "episodes_per_round": episodes_per_round,
        "dimension": dimension, "hidden": hidden,
        "learning_rate": learning_rate, "entropy_weight": entropy_weight,
        "value_weight": value_weight, "auxiliary_weight": auxiliary_weight,
        "boundary_weight": boundary_weight,
        "hp_monotonic_weight": hp_monotonic_weight,
        "win_anneal_threshold": win_anneal_threshold,
        "max_decisions": max_decisions, "temperature_start": temperature_start,
        "temperature_end": temperature_end,
        "temperature_decay_rounds": temperature_decay_rounds,
        "seed": seed, "seed_prefix": seed_prefix, "environment": environment,
        "warm_sha": warm_sha,
    }
    # Legacy fingerprints remain compatible with earlier checkpoints.
    if optimizer_method == "ppo":
        config.update({
            "optimizer_method": optimizer_method,
            "ppo_epochs": ppo_epochs,
            "ppo_batch_size": ppo_batch_size,
            "ppo_sample_limit": ppo_sample_limit,
            "ppo_clip_epsilon": ppo_clip_epsilon,
            "ppo_gamma": ppo_gamma,
            "ppo_gae_lambda": ppo_gae_lambda,
        })
    if tactical_state_encoding != "legacy":
        config["tactical_state_encoding"] = tactical_state_encoding
    if combat_objective != "continuation":
        config["combat_objective"] = combat_objective
    if combat_advantage_baseline != "critic":
        config["combat_advantage_baseline"] = combat_advantage_baseline
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
                tactical_state_encoding=tactical_state_encoding,
                combat_objective=combat_objective,
            )
            planned_seeds = [
                f"{seed_prefix}-{index}-{j}" for j in range(episodes_per_round)
            ]
            excluded = [s for s in planned_seeds if s in excluded_training_seeds]
            seeds = [s for s in planned_seeds if s not in excluded_training_seeds]
            if not seeds:
                raise ValueError(f"Every training seed excluded in round {index + 1}")
            if excluded:
                print(
                    f"[split-excluded] round {index + 1}: explicitly excluded "
                    f"{len(excluded)} unsupported emulator seed(s): "
                    f"{', '.join(excluded)}; no rollout, loss, or reward recorded",
                    flush=True,
                )
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
            local_targets: list[list[tuple[CombatOutcome, float]]] = []
            for episode, run_target in run_targets:
                local_targets.append([
                    (
                        segment,
                        (
                            _hp_preservation_target(segment)
                            if combat_objective == "hp_preservation"
                            else _hp_first_combat_target(segment)
                            if combat_objective == "hp_first"
                            else _combat_boundary_target(
                                episode, j, model, run_target,
                                boundary_weight=boundary_weight,
                            )
                        ),
                    )
                    for j, segment in enumerate(episode.combat_outcomes)
                ])
            other_run_baselines = (
                _other_run_combat_baselines(local_targets)
                if combat_advantage_baseline == "leave_one_run_out" else None
            )
            combat_targets: list[float]
            monotonic_terms: list[Any]
            losses: list[float]
            optimizer_started = time.perf_counter()
            if optimizer_method == "ppo":
                from sts2_ai.training.phase_split_ppo import ppo_update

                ppo = ppo_update(
                    torch=torch,
                    run_targets=run_targets,
                    local_targets=local_targets,
                    params=params,
                    optimizers=optimizers,
                    dimension=dimension,
                    encoding=tactical_state_encoding,
                    temperature=temperature,
                    gamma=ppo_gamma,
                    gae_lambda=ppo_gae_lambda,
                    epochs=ppo_epochs,
                    batch_size=ppo_batch_size,
                    sample_limit=ppo_sample_limit,
                    clip_epsilon=ppo_clip_epsilon,
                    value_weight=value_weight,
                    entropy_weight=entropy_weight,
                    seed=seed + index * 1000003,
                )
                steps = ppo.steps
                strategic_count = ppo.counts["strategy"]
                tactical_count = ppo.counts["combat"]
                phase_diagnostics = ppo.phase
                losses = [ppo.loss] if ppo.loss is not None else []
                monotonic_terms = []
                combat_targets = [
                    target for group in local_targets for _, target in group
                ]
            else:
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                strategic_count = tactical_count = 0
                combat_targets = []
                losses = []
                phase_counts = {
                    phase: sum(
                        dec.phase == phase
                        for episode, _ in run_targets for dec in episode.decisions
                    )
                    for phase in ("strategy", "combat")
                }
                diagnostic_totals = {
                    phase: {"policy": 0.0, "value_mse": 0.0, "entropy": 0.0}
                    for phase in phase_counts
                }
                for episode_index, (episode, run_target) in enumerate(run_targets):
                    # Every target uses outcomes observed under the frozen cohort policy.
                    targets = [run_target] * len(episode.decisions)
                    tactical_baselines: dict[int, float] = {}
                    for j, (segment, local_target) in enumerate(
                        local_targets[episode_index]
                    ):
                        combat_targets.append(local_target)
                        for d in range(segment.start_decision, segment.end_decision):
                            if episode.decisions[d].phase != "combat":
                                raise ValueError("Combat segment includes a strategy decision")
                            targets[d] = local_target
                            if other_run_baselines is not None:
                                tactical_baselines[d] = other_run_baselines[episode_index][j]
                    phase_terms: dict[str, dict[str, list[Any]]] = {
                        phase: {"policy": [], "value_mse": [], "entropy": []}
                        for phase in phase_counts
                    }
                    for decision_index, (decision, target) in enumerate(
                        zip(episode.decisions, targets, strict=True)
                    ):
                        phase = decision.phase
                        if phase not in params:
                            raise ValueError("Unknown decision phase")
                        if phase == "combat":
                            tactical_count += 1
                        else:
                            strategic_count += 1
                        value, logits = _forward_decision(
                            decision, params[phase], dimension, torch,
                            tactical_state_encoding=tactical_state_encoding,
                        )
                        log_probs = torch.nn.functional.log_softmax(
                            logits / temperature, dim=0,
                        )
                        probs = log_probs.exp()
                        target_tensor = torch.tensor(target, dtype=torch.float32)
                        if phase == "combat" and other_run_baselines is not None:
                            if decision_index not in tactical_baselines:
                                raise ValueError("Combat action lacks a resolved combat outcome")
                            advantage = target_tensor - tactical_baselines[decision_index]
                        else:
                            advantage = target_tensor - value.detach()
                        phase_terms[phase]["policy"].append(
                            -log_probs[decision.chosen_index] * advantage
                        )
                        phase_terms[phase]["value_mse"].append(
                            (value - target_tensor).square()
                        )
                        phase_terms[phase]["entropy"].append(
                            -(probs * log_probs).sum()
                        )
                    if loss_normalization == "legacy_episode_sum":
                        # Preserve old behavior exactly for reproducibility.
                        policy_terms = [
                            x for phase in phase_counts for x in phase_terms[phase]["policy"]
                        ]
                        value_terms = [
                            x for phase in phase_counts for x in phase_terms[phase]["value_mse"]
                        ]
                        entropy_terms = [
                            x for phase in phase_counts for x in phase_terms[phase]["entropy"]
                        ]
                        if policy_terms:
                            loss = (
                                torch.stack(policy_terms).sum()
                                + value_weight * torch.stack(value_terms).mean()
                                - entropy_weight * torch.stack(entropy_terms).mean()
                            ) / max(1, len(run_targets))
                            loss.backward()
                            losses.append(float(loss.detach()))
                    else:
                        # Each phase has the SAME overall weight regardless of how
                        # many card plays or map choices occur in a cohort.
                        # Mean reduction also makes entropy/value coefficients
                        # meaningful relative to the policy gradient.
                        for phase, total in phase_counts.items():
                            if not total or not phase_terms[phase]["policy"]:
                                continue
                            terms = phase_terms[phase]
                            loss = (
                                torch.stack(terms["policy"]).sum()
                                + value_weight * torch.stack(terms["value_mse"]).sum()
                                - entropy_weight * torch.stack(terms["entropy"]).sum()
                            ) / total
                            loss.backward()
                            losses.append(float(loss.detach()))
                    for phase, phase_components in phase_terms.items():
                        for name, component_tensors in phase_components.items():
                            if component_tensors:
                                diagnostic_totals[phase][name] += sum(
                                    float(tensor.detach())
                                    for tensor in component_tensors
                                )
                phase_diagnostics = {
                    phase: {
                        key: total / phase_counts[phase]
                        for key, total in terms.items()
                    }
                    for phase, terms in diagnostic_totals.items()
                    if phase_counts[phase]
                }
                # Direct combat-boundary feedback to the strategic continuation
                # critic: for otherwise identical future resources, slightly more
                # HP is normally preferable. This does NOT fix a potion/HP rate.
                monotonic_terms = []
                for ep, _ in run_targets:
                    for segment in ep.combat_outcomes:
                        if segment.result != "victory" or not segment.exit_public_json:
                            continue
                        pair = _higher_hp_public_pair(segment.exit_public_json)
                        if pair is None:
                            continue
                        lower, higher, delta = pair
                        low_x = torch.tensor(
                            _dense(state_features(lower, dimension), dimension),
                            dtype=torch.float32,
                        )
                        high_x = torch.tensor(
                            _dense(state_features(higher, dimension), dimension),
                            dtype=torch.float32,
                        )
                        low_value, _ = _forward(
                            params["strategy"], low_x, None, torch,
                        )
                        high_value, _ = _forward(
                            params["strategy"], high_x, None, torch,
                        )
                        monotonic_terms.append(torch.relu(
                            low_value - high_value + 0.04 * delta
                        ))
                if monotonic_terms and hp_monotonic_weight:
                    (hp_monotonic_weight * torch.stack(monotonic_terms).mean()).backward()

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
                round_index=index, played=len(cohort),
                completed=len(completed),
                censored=len(cohort) - len(completed),
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
                hp_monotonic_pairs=len(monotonic_terms),
                phase_loss_diagnostics=phase_diagnostics,
                mean_victory_exit_hp_fraction=(
                    sum(
                        min(1.0, max(0.0, (o.exit.hp or 0) /
                            max(1, o.exit.max_hp or o.entry.max_hp or 1)))
                        for ep in cohort for o in ep.combat_outcomes
                        if o.result == "victory" and o.exit.hp is not None
                    ) / sum(
                        o.result == "victory" and o.exit.hp is not None
                        for ep in cohort for o in ep.combat_outcomes
                    ) if any(
                        o.result == "victory" and o.exit.hp is not None
                        for ep in cohort for o in ep.combat_outcomes
                    ) else None
                ),
                mean_victory_hp_lost_fraction=(
                    sum(
                        (o.entry.hp - o.exit.hp) /
                        max(1, o.exit.max_hp or o.entry.max_hp or 1)
                        for ep in cohort for o in ep.combat_outcomes
                        if o.result == "victory"
                        and o.entry.hp is not None and o.exit.hp is not None
                    ) / sum(
                        o.result == "victory"
                        and o.entry.hp is not None and o.exit.hp is not None
                        for ep in cohort for o in ep.combat_outcomes
                    ) if any(
                        o.result == "victory"
                        and o.entry.hp is not None and o.exit.hp is not None
                        for ep in cohort for o in ep.combat_outcomes
                    ) else None
                ),
                mean_potions_used_per_combat=(
                    sum(len(o.potions_used) for ep in cohort for o in ep.combat_outcomes)
                    / sum(len(ep.combat_outcomes) for ep in cohort)
                    if any(ep.combat_outcomes for ep in cohort) else None
                ),
            )
            rows.append(row)
            if checkpoint is not None:
                _checkpoint_save(
                    checkpoint, torch=torch, params=params,
                    optimizers=optimizers, fingerprint=fingerprint,
                    revision=revision, rows=rows,
                )
            if combat_samples_dir is not None:
                combat_samples_dir.mkdir(parents=True, exist_ok=True)
                target = combat_samples_dir / f"round-{index + 1:04d}.jsonl"
                lines = [
                    json.dumps({
                        "schema": "sts2-public-combat-sample-v1",
                        "seed": ep.seed,
                        "round": index + 1,
                        "policy_snapshot": model.model_id,
                        "sampling_temperature": temperature,
                        "emulator_revision": revision,
                        "environment": environment,
                        "outcome": asdict(segment),
                    }, sort_keys=True)
                    for ep in cohort for segment in ep.combat_outcomes
                ]
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", dir=combat_samples_dir,
                        suffix=".pending", delete=False,
                    ) as output:
                        temporary = Path(output.name)
                        output.write("\n".join(lines) + ("\n" if lines else ""))
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            if progress is not None:
                progress(row)
    return _export_split(
        params["strategy"], params["combat"], dimension, hidden,
        model_id=f"phase-split-v1-{fingerprint[:12]}-{rounds}",
        trained=bool(rows and any(x.optimization_steps for x in rows)),
        tactical_state_encoding=tactical_state_encoding,
        combat_objective=combat_objective,
    ), tuple(rows)
