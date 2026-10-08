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
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, Observation
from sts2_ai.models.hashed_linear import neural_action_features, state_dict, state_features
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.neural import _dense, _export, _forward, _new_params

SELFPLAY_VERSION = "public-onpolicy-reinforce-actor-critic-v1"


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


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    model: NeuralPolicyValueModel
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
) -> Episode:
    """Run the actual emulator; never give hidden handles to the actor.

    Terminal rewards are observed only after real terminal transitions.
    Early caps/censorship return a distinct non-training episode.
    Every emulator handle is released even on partial failures.
    """
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    policy = InformationPolicy(policy_id)
    state = backend.reset(seed)
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
                    seed, tuple(history), outcome, act, floor, hp_ratio, True
                )
            if len(history) >= max_decisions:
                act, floor, hp_ratio = _public_metrics(frame)
                return Episode(
                    seed, tuple(history), "truncated", act, floor, hp_ratio, False
                )

            actions = tuple(backend.legal_actions(state))
            index = sample_public_action(model, frame, actions, rng=actor_rng)
            # Record *only* public inputs and the sampled legal index.
            history.append(PublicDecision(frame, actions, index))
            next_state = backend.step(state, actions[index]).child
            old = state
            state = next_state
            backend.release_many((old,))
    finally:
        backend.release_many((state,))


def _bounded_return(episode: Episode, aux_coefficient: float) -> float:
    if not episode.completed:
        raise ValueError("Truncated episodes cannot receive a Monte Carlo target")
    if not 0.0 <= aux_coefficient <= 1.0:
        raise ValueError("Auxiliary coefficient must be within [0,1]")
    # Prototype currently has three acts, six floors per act. This
    # normalization is not a native STS2 progress claim.
    progress = min(
        1.0,
        max(0.0, ((max(1, episode.act or 1) - 1) * 6
                   + max(0, episode.floor or 0)) / 18.0),
    )
    auxiliary = 0.7 * progress + 0.3 * episode.hp_ratio
    return (1.0 - aux_coefficient) * float(episode.won) + aux_coefficient * auxiliary


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
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    update_epochs: int = 1,
    max_decisions: int = 2048,
    run_seed_prefix: str = "neural-train",
    seed: int = 0,
) -> SelfPlayResult:
    """Run genuine full-game episodes and optimize an on-policy neural actor.

    Every round samples from the *current* actor. No offline teacher data.
    Each collected batch is optimized with an episodic REINFORCE objective
    and a detached Monte Carlo value baseline. The auxiliary reward is
    annealed to zero across rounds, never replacing the final win target.
    """
    if any(type(v) is not int or v <= 0 for v in (
        rounds, episodes_per_round, dimension, hidden, update_epochs, max_decisions
    )):
        raise ValueError("Rounds, episode counts and dimensions must be positive integers")
    if not 0 < learning_rate < 1 or not 0 <= auxiliary_weight <= 1:
        raise ValueError("Invalid learning rate or auxiliary coefficient")
    if not 0 <= value_weight <= 10 or not 0 <= entropy_weight <= 1:
        raise ValueError("Invalid value or entropy loss weight")
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Install PyTorch for self-play training: pip install -e '.[neural]'"
        ) from exc

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    params = _new_params(dimension, hidden, torch)
    optimizer = torch.optim.AdamW(list(params.values()), lr=learning_rate)
    metrics: list[TrainingRound] = []

    for round_index in range(rounds):
        # The very last round has pure terminal-victory target.
        alpha = auxiliary_weight * (1.0 - round_index / max(1, rounds - 1))
        model = _export(
            params, dimension, hidden,
            model_id=f"selfplay-actor-before-round-{round_index}",
            value_head_trained=round_index > 0,
        )
        batch: list[tuple[PublicDecision, float]] = []
        completed = wins = censored = 0
        for episode_index in range(episodes_per_round):
            episode = collect_public_episode(
                backend, model,
                seed=f"{run_seed_prefix}-{round_index}-{episode_index}",
                actor_rng=rng,
                max_decisions=max_decisions,
            )
            if not episode.completed:
                censored += 1
                continue
            completed += 1
            wins += int(episode.won)
            target = _bounded_return(episode, alpha)
            batch.extend((decision, target) for decision in episode.decisions)

        total_loss = 0.0
        steps = 0
        if batch:
            samples = [
                (*_tensors(decision, dimension, torch), decision.chosen_index, value)
                for decision, value in batch
            ]
            for _ in range(update_epochs):
                # Train from the most recently collected on-policy batch.
                # Repeated epochs introduce policy staleness; keep default one.
                rng.shuffle(samples)
                for obs, actions, index, outcome in samples:
                    optimizer.zero_grad()
                    estimate, logits = _forward(params, obs, actions, torch)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=0)
                    probs = log_probs.exp()
                    target_tensor = torch.tensor(outcome, dtype=torch.float32)
                    advantage = target_tensor - estimate.detach()
                    entropy = -(probs * log_probs).sum()
                    loss = (
                        -log_probs[index] * advantage
                        + value_weight * (estimate - target_tensor).square()
                        - entropy_weight * entropy
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm=5.0)
                    optimizer.step()
                    total_loss += float(loss.detach())
                    steps += 1
        metrics.append(TrainingRound(
            round_index, episodes_per_round, completed, censored, wins,
            steps, total_loss / steps if steps else None, alpha
        ))

    signature = hashlib.sha256(
        (
            f"{SELFPLAY_VERSION}:{seed}:{rounds}:{episodes_per_round}:"
            f"{dimension}:{hidden}:{learning_rate}:{run_seed_prefix}"
        ).encode()
    ).hexdigest()[:12]
    final = _export(
        params, dimension, hidden,
        model_id=f"selfplay-v1-{signature}",
        value_head_trained=any(item.update_steps > 0 for item in metrics),
    )
    return SelfPlayResult(final, tuple(metrics))
