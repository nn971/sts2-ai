from __future__ import annotations

import json
from dataclasses import dataclass

from sts2_ai.agents.base import Agent, Decision
from sts2_ai.emulator import EmulatorBackend, InformationPolicy, Observation


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    decision_index: int
    parent_hash: str
    observation_hash: str
    action_id: str
    action_kind: str
    policy_name: str
    child_hash: str
    terminal: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: str
    information_policy: str
    initial_hash: str
    final_hash: str
    final_observation: Observation
    terminal_outcome: str | None
    steps: tuple[EpisodeStep, ...]


def run_episode(
    backend: EmulatorBackend,
    agent: Agent,
    *,
    seed: str,
    information_policy: InformationPolicy,
    max_decisions: int = 5_000,
) -> EpisodeResult:
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    state = backend.reset(seed)
    initial_hash = backend.exact_hash(state)
    steps: list[EpisodeStep] = []

    for decision_index in range(max_decisions):
        observation = backend.observe(state, information_policy)
        if backend.is_terminal(state):
            return _finish(
                backend,
                state,
                seed,
                information_policy,
                initial_hash,
                observation,
                steps,
            )

        legal_actions = backend.legal_actions(state)
        if not legal_actions:
            raise RuntimeError(
                f"Nonterminal emulator state {backend.exact_hash(state)} has no legal actions"
            )

        decision = agent.choose(observation, legal_actions)
        _ensure_legal_decision(decision, legal_actions)

        parent_hash = backend.exact_hash(state)
        transition = backend.step(state, decision.action)
        child_hash = backend.exact_hash(transition.child)

        steps.append(
            EpisodeStep(
                decision_index=decision_index,
                parent_hash=parent_hash,
                observation_hash=observation.observation_hash,
                action_id=decision.action.action_id,
                action_kind=decision.action.kind,
                policy_name=decision.policy_name,
                child_hash=child_hash,
                terminal=transition.terminal,
            )
        )
        state = transition.child

    raise RuntimeError(f"Episode exceeded max_decisions={max_decisions} for seed {seed!r}")


def _finish(
    backend: EmulatorBackend,
    state: str,
    seed: str,
    information_policy: InformationPolicy,
    initial_hash: str,
    observation: Observation,
    steps: list[EpisodeStep],
) -> EpisodeResult:
    payload = json.loads(observation.payload_json)
    outcome: str | None = None
    if isinstance(payload, dict):
        raw_outcome = payload.get("terminal_outcome")
        if isinstance(raw_outcome, str):
            outcome = raw_outcome

    return EpisodeResult(
        seed=seed,
        information_policy=information_policy.policy_id,
        initial_hash=initial_hash,
        final_hash=backend.exact_hash(state),
        final_observation=observation,
        terminal_outcome=outcome,
        steps=tuple(steps),
    )


def _ensure_legal_decision(
    decision: Decision,
    legal_actions: object,
) -> None:
    legal_ids = {
        action.action_id
        for action in legal_actions  # type: ignore[union-attr]
    }
    if decision.action.action_id not in legal_ids:
        raise ValueError(
            f"Agent selected illegal action {decision.action.action_id!r}"
        )
