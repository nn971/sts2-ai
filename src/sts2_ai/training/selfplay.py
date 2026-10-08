"""First teacher-free on-policy neural self-play against the *ordinary* emulator.

No posterior sampler, oracle-state search, expert or teacher action labels.
The emulator owns hidden RNG; the actor only sees Observation + legal menu.
This is Monte Carlo REINFORCE with a detached learned value baseline, not PPO.
Truncated episodes are NEVER labelled as failures or terminal rewards.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    JsonlEmulatorBackend,
    LegalAction,
    Observation,
)
from sts2_ai.emulator.run_environment import (
    LEGACY,
    cumulative_floor_progress,
    require_environment,
    reset_training_run,
)
from sts2_ai.models.hashed_linear import neural_action_features, state_dict, state_features
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.neural import _dense, _export, _forward, _new_params
from sts2_ai.training.parallel_rollouts import (
    collect_parallel,
    episode_actor_seed,
    open_worker_pool,
)
from sts2_ai.training.rollout_failure import PublicRolloutFailure, SCHEMA as FAILURE_SCHEMA
from sts2_ai.training.selfplay_checkpoint import load_checkpoint, save_checkpoint

SELFPLAY_VERSION = "public-onpolicy-reinforce-actor-critic-v3-parallel-resumable"


@dataclass(frozen=True, slots=True)
class PublicDecision:
    observation: Observation
    legal_actions: tuple[LegalAction, ...]
    chosen_index: int


@dataclass(frozen=True, slots=True)
class Episode:
    seed: str
    decisions: tuple[PublicDecision, ...]
    outcome: str
    act: int | None
    floor: int | None
    hp_ratio: float
    completed: bool
    environment: str = LEGACY

    @property
    def won(self) -> bool:
        return self.completed and self.outcome == "victory"


@dataclass(frozen=True, slots=True)
class TrainingRound:
    round_index: int
    played: int
    completed: int
    censored: int
    wins: int
    update_steps: int
    mean_loss: float | None
    auxiliary_coefficient: float
    decision_samples: int
    mean_return: float | None
    mean_progress: float | None


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    model: NeuralPolicyValueModel
    initial_model: NeuralPolicyValueModel
    rounds: tuple[TrainingRound, ...]


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits or any(not math.isfinite(logit) for logit in logits):
        raise ValueError("Neural actor requires nonempty finite legal action logits")
    high = max(logits)
    exp = [math.exp(x - high) for x in logits]
    total = sum(exp)
    return tuple(x / total for x in exp)


def sample_public_action(
    model: NeuralPolicyValueModel,
    observation: Observation,
    actions: Sequence[LegalAction],
    *,
    rng: random.Random,
) -> int:
    """On-policy categorical sampling over the *public* legal action list."""
    if not actions:
        raise ValueError("Cannot choose from an empty legal-action menu")
    probabilities = _softmax(model.evaluate(observation, actions).action_logits)
    threshold = rng.random()
    total = 0.0
    for index, probability in enumerate(probabilities):
        total += probability
        if threshold < total:
            return index
    return len(actions) - 1


def _public_metrics(observation: Observation) -> tuple[int | None, int | None, float]:
    obj = json.loads(observation.payload_json)
    if not isinstance(obj, dict):
        raise ValueError("Public observation must be a JSON object")
    act = obj.get("act")
    floor = obj.get("floor")
    hp = obj.get("hp")
    max_hp = obj.get("max_hp")
    ratio = (
        max(0.0, min(1.0, float(hp) / float(max_hp)))
        if type(hp) is int and type(max_hp) is int and max_hp > 0
        else 0.0
    )
    return (
        act if type(act) is int else None,
        floor if type(floor) is int else None,
        ratio,
    )


def collect_public_episode(
    backend: EmulatorBackend,
    model: NeuralPolicyValueModel,
    *,
    seed: str,
    actor_rng: random.Random,
    max_decisions: int = 2048,
    policy_id: str = "prototype-fair-v0",
    environment: str = LEGACY,
) -> Episode:
    """Run the actual emulator; never give hidden handles to the actor.

    Terminal rewards are observed only after real terminal transitions.
    Early caps/censorship return a distinct non-training episode.
    Every emulator handle is released even on partial failures.
    """
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    policy = InformationPolicy(policy_id)
    state = reset_training_run(backend, seed, environment)
    history: list[PublicDecision] = []
    try:
        while True:
            frame = backend.observe(state, policy)
            if backend.is_terminal(state):
                act, floor, hp_ratio = _public_metrics(frame)
                obj = json.loads(frame.payload_json)
                assert isinstance(obj, dict)
                outcome = obj.get("terminal_outcome")
                if outcome not in ("victory", "defeat"):
                    raise ValueError("Terminal episode lacks a certified victory/defeat")
                return Episode(
                    seed, tuple(history), outcome, act, floor, hp_ratio, True, environment
                )
            if len(history) >= max_decisions:
                act, floor, hp_ratio = _public_metrics(frame)
                return Episode(
                    seed, tuple(history), "truncated", act, floor, hp_ratio, False, environment
                )

            actions = tuple(backend.legal_actions(state))
            index = sample_public_action(model, frame, actions, rng=actor_rng)
            # Record *only* public inputs and the sampled legal index.
            history.append(PublicDecision(frame, actions, index))
            try:
                next_state = backend.step(state, actions[index]).child
            except Exception as exc:
                # Fail closed: a broken transition is neither a defeat nor an
                # ordinary decision-cap truncation. Preserve the complete
                # player-visible action path for deterministic emulator replay.
                # The current handle is released by the outer finally block.
                raise PublicRolloutFailure({
                    "schema": FAILURE_SCHEMA,
                    "seed": seed,
                    "environment": environment,
                    "policy_id": policy_id,
                    "actor_model_id": model.model_id,
                    "decision_index": len(history) - 1,
                    "chosen_action_ids": [
                        decision.legal_actions[decision.chosen_index].action_id
                        for decision in history
                    ],
                    "failing_observation_hash": frame.observation_hash,
                    "failing_public_observation": json.loads(frame.payload_json),
                    "failing_action": {
                        "action_id": actions[index].action_id,
                        "kind": actions[index].kind,
                        "payload_json": actions[index].payload_json,
                    },
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }) from exc
            old = state
            state = next_state
            backend.release_many((old,))
    finally:
        backend.release_many((state,))


def _normalized_progress(episode: Episode) -> float:
    """Normalize using the geometry of the episode's actual run mode."""
    current, maximum = cumulative_floor_progress(
        episode.act, episode.floor, episode.environment
    )
    return current / maximum


def _curriculum_coefficient(
    base_weight: float, completed_victories: int, win_anneal_threshold: int
) -> float:
    """Anneal only after observed victories, not after an arbitrary round count."""
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("Auxiliary weight must be in [0, 1]")
    if type(completed_victories) is not int or completed_victories < 0:
        raise ValueError("Completed victories must be nonnegative")
    if type(win_anneal_threshold) is not int or win_anneal_threshold <= 0:
        raise ValueError("win_anneal_threshold must be positive")
    return base_weight * (1.0 - min(1.0, completed_victories / win_anneal_threshold))


def _bounded_return(episode: Episode, aux_coefficient: float) -> float:
    if not episode.completed:
        raise ValueError("Truncated episodes cannot receive a Monte Carlo target")
    if not 0.0 <= aux_coefficient <= 1.0:
        raise ValueError("Auxiliary coefficient must be within [0,1]")
    auxiliary = 0.7 * _normalized_progress(episode) + 0.3 * episode.hp_ratio
    # A genuine victory is always worth 1. Auxiliary signals only distinguish
    # completed defeats until the curriculum has observed enough victories.
    # This prevents an early low-HP victory from being worth less than a defeat.
    return 1.0 if episode.won else aux_coefficient * auxiliary


def _tensors(decision: PublicDecision, dimension: int, torch: Any) -> tuple[Any, Any]:
    state = state_dict(decision.observation.payload_json)
    x = torch.tensor(_dense(state_features(state, dimension), dimension),
                     dtype=torch.float32)
    actions = torch.tensor([
        _dense(
            neural_action_features(
                state, action.kind, action.payload_json, dimension
            ), dimension
        )
        for action in decision.legal_actions
    ], dtype=torch.float32)
    return x, actions


def train_selfplay(
    backend: EmulatorBackend,
    *,
    rounds: int = 3,
    episodes_per_round: int = 4,
    dimension: int = 128,
    hidden: int = 16,
    learning_rate: float = 0.003,
    auxiliary_weight: float = 0.4,
    win_anneal_threshold: int = 16,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    update_epochs: int = 1,
    max_decisions: int = 2048,
    run_seed_prefix: str = "neural-train",
    seed: int = 0,
    workers: int = 1,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    environment: str = LEGACY,
) -> SelfPlayResult:
    """Run genuine full-game episodes and optimize an on-policy neural actor.

    Every round samples from the *current* actor. No offline teacher data.
    Each complete on-policy cohort produces exactly one optimizer update,
    without making earlier action samples off-policy partway through a round.
    The return is victory first; defeat-only floor/HP auxiliaries are
    annealed by observed victories, never merely by elapsed training rounds.
    A leave-one-episode-out constant baseline reduces early gradient noise
    when the learned value function is still uncalibrated.
    """
    if any(type(v) is not int or v <= 0 for v in (
        rounds, episodes_per_round, dimension, hidden, update_epochs, max_decisions
    )):
        raise ValueError("Rounds, episode counts and dimensions must be positive integers")
    if not 0 < learning_rate < 1 or not 0 <= auxiliary_weight <= 1:
        raise ValueError("Invalid learning rate or auxiliary coefficient")
    if not 0 <= value_weight <= 10 or not 0 <= entropy_weight <= 1:
        raise ValueError("Invalid value or entropy loss weight")
    if update_epochs != 1:
        raise ValueError("Strict on-policy v2 uses exactly one update epoch")
    if type(win_anneal_threshold) is not int or win_anneal_threshold <= 0:
        raise ValueError("win_anneal_threshold must be positive")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if resume and checkpoint_path is None:
        raise ValueError("Resume requires a checkpoint path")
    if workers > 1 and not isinstance(backend, JsonlEmulatorBackend):
        raise ValueError("Parallel workers require a pinned JSONL emulator backend")
    # Check capability BEFORE importing Torch or opening workers / writing a
    # checkpoint. A missing bridge operation must never silently train legacy.
    require_environment(backend, environment)
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Install PyTorch for self-play training: pip install -e '.[neural]'"
        ) from exc

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    params = _new_params(dimension, hidden, torch)
    optimizer = torch.optim.AdamW(list(params.values()), lr=learning_rate)
    metrics: list[TrainingRound] = []
    initial_model = _export(
        params, dimension, hidden,
        model_id=f"selfplay-initial-seed-{seed}", value_head_trained=False,
    )
    revision = getattr(backend, "emulator_revision", "test-backend")
    config: dict[str, Any] = {
        "training_version": SELFPLAY_VERSION,
        "seed": seed,
        "episodes_per_round": episodes_per_round,
        "dimension": dimension,
        "hidden": hidden,
        "learning_rate": learning_rate,
        "auxiliary_weight": auxiliary_weight,
        "win_anneal_threshold": win_anneal_threshold,
        "value_weight": value_weight,
        "entropy_weight": entropy_weight,
        "update_epochs": update_epochs,
        "max_decisions": max_decisions,
        "run_seed_prefix": run_seed_prefix,
        "environment": environment,
    }
    if resume:
        assert checkpoint_path is not None
        initial_model, metrics = load_checkpoint(
            checkpoint_path, torch=torch, params=params, optimizer=optimizer,
            config=config, revision=revision,
        )
    if len(metrics) > rounds:
        raise ValueError("Checkpoint has more completed rounds than requested")
    cumulative_victories = sum(row.wins for row in metrics)

    with ExitStack() as resources:
        pool = (
            resources.enter_context(open_worker_pool(backend, workers))
            if isinstance(backend, JsonlEmulatorBackend) and workers > 1
            else None
        )
        for round_index in range(len(metrics), rounds):
            # Only observed completed victories can reduce the shaping coefficient.
            # Zero victories = the early learning signal is not switched off.
            alpha = _curriculum_coefficient(
                auxiliary_weight, cumulative_victories, win_anneal_threshold
            )
            model = _export(
                params, dimension, hidden,
                model_id=f"selfplay-actor-before-round-{round_index}",
                value_head_trained=round_index > 0 and any(
                    row.update_steps > 0 for row in metrics
                ),
            )
            completed_episodes: list[tuple[Episode, float]] = []
            completed = wins = censored = 0
            run_seeds = [
                f"{run_seed_prefix}-{round_index}-{episode_index}"
                for episode_index in range(episodes_per_round)
            ]
            if pool is None:
                cohort = tuple(
                    collect_public_episode(
                        backend, model,
                        seed=run_seed,
                        actor_rng=random.Random(episode_actor_seed(seed, run_seed)),
                        max_decisions=max_decisions,
                        environment=environment,
                    )
                    for run_seed in run_seeds
                )
            else:
                cohort = collect_parallel(
                    pool, model, run_seeds,
                    base_seed=seed, max_decisions=max_decisions,
                    policy_id="prototype-fair-v0",
                    environment=environment,
                )
            for episode in cohort:
                if not episode.completed:
                    censored += 1
                    continue
                completed += 1
                wins += int(episode.won)
                completed_episodes.append((episode, _bounded_return(episode, alpha)))

            decisions_seen = sum(len(ep.decisions) for ep, _ in completed_episodes)
            mean_loss: float | None = None
            updates = 0
            if completed_episodes and decisions_seen:
                # No parameter update until the entire cohort has been scored with
                # the same actor. Accumulate gradients per episode to bound memory
                # by one trajectory, then perform ONE optimizer step.
                optimizer.zero_grad()
                total_return = sum(ret for _, ret in completed_episodes)
                losses: list[float] = []
                for episode, outcome in completed_episodes:
                    target_tensor = torch.tensor(outcome, dtype=torch.float32)
                    # Other episodes are statistically independent of this one's
                    # sampled actions, so leave-one-out is a valid baseline.
                    other_mean = (
                        (total_return - outcome) / (len(completed_episodes) - 1)
                        if len(completed_episodes) > 1 else 0.0
                    )
                    policy_terms = []
                    value_terms = []
                    entropy_terms = []
                    for decision in episode.decisions:
                        obs, actions = _tensors(decision, dimension, torch)
                        estimate, logits = _forward(params, obs, actions, torch)
                        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
                        probabilities = log_probs.exp()
                        # Both baselines are action-independent at this decision.
                        # Keep the learned V detached from the policy gradient.
                        baseline = (estimate.detach() + other_mean) / 2.0 if (
                            len(completed_episodes) > 1
                        ) else estimate.detach()
                        policy_terms.append(
                            -log_probs[decision.chosen_index] * (target_tensor - baseline)
                        )
                        value_terms.append((estimate - target_tensor).square())
                        entropy_terms.append(-(probabilities * log_probs).sum())

                    # The REINFORCE policy term is a SUM of log-prob gradients
                    # along the full episode; dividing by its length would bias
                    # the objective towards short runs. Value/entropy terms are
                    # per-decision averages to keep their scales stable.
                    policy_loss = torch.stack(policy_terms).sum()
                    value_loss = torch.stack(value_terms).mean()
                    entropy = torch.stack(entropy_terms).mean()
                    episode_loss = (
                        policy_loss + value_weight * value_loss
                        - entropy_weight * entropy
                    ) / len(completed_episodes)
                    episode_loss.backward()
                    losses.append(float(episode_loss.detach()))
                torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm=5.0)
                optimizer.step()
                mean_loss = sum(losses)
                updates = 1

            cumulative_victories += wins
            metrics.append(TrainingRound(
                round_index, episodes_per_round, completed, censored, wins,
                updates, mean_loss, alpha,
                decisions_seen,
                (sum(ret for _, ret in completed_episodes) / len(completed_episodes)
                 if completed_episodes else None),
                (sum(_normalized_progress(ep) for ep, _ in completed_episodes)
                 / len(completed_episodes) if completed_episodes else None),
            ))
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path, torch=torch, params=params, optimizer=optimizer,
                    config=config, revision=revision, initial_model=initial_model,
                    metrics=metrics,
                )

    signature = hashlib.sha256(
        (
            f"{SELFPLAY_VERSION}:{seed}:{rounds}:{episodes_per_round}:"
            f"{dimension}:{hidden}:{learning_rate}:{run_seed_prefix}"
        ).encode()
    ).hexdigest()[:12]
    final = _export(
        params, dimension, hidden,
        model_id=f"selfplay-v3-{signature}",
        value_head_trained=any(item.update_steps > 0 for item in metrics),
    )
    return SelfPlayResult(final, initial_model, tuple(metrics))
