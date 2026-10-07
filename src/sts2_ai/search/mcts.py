from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, Observation, StateHandle

from .base import ActionEvaluation, SearchBudget, SearchResult

ValueFunction = Callable[[Observation], float]


@dataclass(slots=True)
class _Edge:
    action: LegalAction
    child_hash: str
    visits: int = 0
    value_sum: float = 0.0


@dataclass(slots=True)
class _Node:
    handle: StateHandle
    exact_hash: str
    observation: Observation
    legal_actions: tuple[LegalAction, ...]
    terminal: bool
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    edges: dict[str, _Edge] = field(default_factory=dict)


class UctMcts:
    """Transposition-aware oracle-exact UCT over emulator states."""

    search_version = "oracle-exact-uct-v1"

    def __init__(
        self,
        backend: EmulatorBackend,
        *,
        policy: InformationPolicy,
        rollout_policy: HeuristicAgent | None = None,
        value_fn: ValueFunction | None = None,
        exploration: float = math.sqrt(2.0),
        rollout_depth: int = 128,
        seed: int = 0,
    ) -> None:
        if rollout_depth < 0:
            raise ValueError("rollout_depth must be non-negative")
        self._backend = backend
        self._policy = policy
        self._rollout_policy = rollout_policy or HeuristicAgent()
        self._value_fn = value_fn or sts2_value
        self._exploration = exploration
        self._rollout_depth = rollout_depth
        self._rng = random.Random(seed)

    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult:
        simulations = budget.max_simulations
        if simulations is None:
            simulations = budget.max_nodes
        if simulations is None:
            simulations = 128
        if simulations <= 0:
            return self._zero_budget_result(state)

        deadline = None
        if budget.max_seconds is not None:
            deadline = time.monotonic() + budget.max_seconds

        transitions = 0
        transposition_hits = 0
        created_handles: set[StateHandle] = set()

        root = self._make_node(state)
        table: dict[str, _Node] = {root.exact_hash: root}

        def expand(node: _Node) -> None:
            nonlocal transitions, transposition_hits
            if node.expanded or node.terminal:
                node.expanded = True
                return

            expansions = tuple(self._backend.expand(node.handle))
            transitions += len(expansions)
            by_action = {item.action.action_id: item for item in expansions}
            if set(by_action) != {action.action_id for action in node.legal_actions}:
                raise RuntimeError("emulator.expand disagrees with legal_actions")

            for action in node.legal_actions:
                transition = by_action[action.action_id]
                child_hash = transition.exact_hash or self._backend.exact_hash(transition.child)
                existing = table.get(child_hash)
                if existing is None:
                    child = self._make_node(
                        transition.child,
                        exact_hash=child_hash,
                        terminal=transition.terminal,
                    )
                    table[child_hash] = child
                    created_handles.add(transition.child)
                else:
                    transposition_hits += 1
                    self._backend.release_many([transition.child])
                node.edges[action.action_id] = _Edge(
                    action=action,
                    child_hash=child_hash,
                )
            node.expanded = True

        completed_simulations = 0
        try:
            for _ in range(simulations):
                if deadline is not None and time.monotonic() >= deadline:
                    break

                node = root
                path_nodes = [root]
                path_edges: list[_Edge] = []

                while True:
                    if node.terminal:
                        value = self._value_fn(node.observation)
                        break

                    if not node.expanded:
                        expand(node)
                        if not node.edges:
                            value = self._value_fn(node.observation)
                            break
                        edge = self._choose_unvisited_or_uct(node)
                        path_edges.append(edge)
                        node = table[edge.child_hash]
                        path_nodes.append(node)
                        value, used = self._rollout(node)
                        transitions += used
                        break

                    if not node.edges:
                        value = self._value_fn(node.observation)
                        break

                    edge = self._choose_unvisited_or_uct(node)
                    path_edges.append(edge)
                    node = table[edge.child_hash]
                    path_nodes.append(node)
                    if edge.visits == 0:
                        value, used = self._rollout(node)
                        transitions += used
                        break

                for visited_node in path_nodes:
                    visited_node.visits += 1
                    visited_node.value_sum += value
                for visited_edge in path_edges:
                    visited_edge.visits += 1
                    visited_edge.value_sum += value
                completed_simulations += 1
        finally:
            if created_handles:
                self._backend.release_many(tuple(created_handles))

        evaluations = tuple(
            self._edge_evaluation(edge)
            for edge in sorted(root.edges.values(), key=lambda item: item.action.action_id)
        )
        if not evaluations and root.legal_actions:
            evaluations = tuple(
                ActionEvaluation(action=action, value=0.0, visits=0)
                for action in root.legal_actions
            )

        return SearchResult(
            root_state_hash=root.exact_hash,
            root_observation_hash=root.observation.observation_hash,
            evaluations=evaluations,
            expanded_nodes=sum(1 for node in table.values() if node.expanded),
            transitions=transitions,
            transposition_hits=transposition_hits,
            search_version=self.search_version,
        )

    def _zero_budget_result(self, state: StateHandle) -> SearchResult:
        node = self._make_node(state)
        return SearchResult(
            root_state_hash=node.exact_hash,
            root_observation_hash=node.observation.observation_hash,
            evaluations=tuple(
                ActionEvaluation(action=action, value=0.0, visits=0)
                for action in node.legal_actions
            ),
            expanded_nodes=0,
            transitions=0,
            transposition_hits=0,
            search_version=self.search_version,
        )

    def _make_node(
        self,
        handle: StateHandle,
        *,
        exact_hash: str | None = None,
        terminal: bool | None = None,
    ) -> _Node:
        actual_terminal = self._backend.is_terminal(handle) if terminal is None else terminal
        observation = self._backend.observe(handle, self._policy)
        legal_actions = (
            tuple(self._backend.legal_actions(handle))
            if not actual_terminal
            else ()
        )
        return _Node(
            handle=handle,
            exact_hash=exact_hash or self._backend.exact_hash(handle),
            observation=observation,
            legal_actions=legal_actions,
            terminal=actual_terminal,
        )

    def _choose_unvisited_or_uct(self, node: _Node) -> _Edge:
        unvisited = [edge for edge in node.edges.values() if edge.visits == 0]
        if unvisited:
            return self._rng.choice(sorted(unvisited, key=lambda edge: edge.action.action_id))

        log_parent = math.log(max(1, node.visits))
        return max(
            node.edges.values(),
            key=lambda edge: (
                (edge.value_sum / edge.visits)
                + self._exploration * math.sqrt(log_parent / edge.visits),
                edge.action.action_id,
            ),
        )

    def _rollout(self, start: _Node) -> tuple[float, int]:
        if start.terminal:
            return self._value_fn(start.observation), 0

        state = start.handle
        observation = start.observation
        legal_actions = start.legal_actions
        temporary: list[StateHandle] = []
        transitions = 0
        try:
            for _ in range(self._rollout_depth):
                if not legal_actions:
                    return self._value_fn(observation), transitions
                decision = self._rollout_policy.choose(observation, legal_actions)
                transition = self._backend.step(state, decision.action)
                transitions += 1
                temporary.append(transition.child)
                state = transition.child
                observation = self._backend.observe(state, self._policy)
                if transition.terminal:
                    return self._value_fn(observation), transitions
                legal_actions = tuple(self._backend.legal_actions(state))
            return self._value_fn(observation), transitions
        finally:
            if temporary:
                self._backend.release_many(temporary)

    @staticmethod
    def _edge_evaluation(edge: _Edge) -> ActionEvaluation:
        if edge.visits == 0:
            return ActionEvaluation(action=edge.action, value=0.0, visits=0)
        mean = edge.value_sum / edge.visits
        # Values are intended to live roughly in [-1, 1]. This is a compact
        # Bernoulli-like standard-error proxy rather than a calibrated interval.
        variance_proxy = max(0.0, 1.0 - (mean * mean))
        uncertainty = math.sqrt(variance_proxy / edge.visits)
        return ActionEvaluation(
            action=edge.action,
            value=mean,
            visits=edge.visits,
            uncertainty=uncertainty,
        )


def sts2_value(observation: Observation) -> float:
    """Simple terminal/cutoff value used before a learned value model exists."""

    raw = json.loads(observation.payload_json)
    state = cast(dict[str, Any], raw if isinstance(raw, dict) else {})
    outcome = state.get("terminal_outcome")
    if outcome == "victory":
        return 1.0
    if outcome == "defeat":
        return -1.0

    hp = _number(state.get("hp"), 0.0)
    max_hp = max(1.0, _number(state.get("max_hp"), 1.0))
    hp_fraction = max(0.0, min(1.0, hp / max_hp))
    act = max(1.0, _number(state.get("act"), 1.0))
    floor = max(0.0, _number(state.get("floor"), 0.0))
    progress = max(0.0, min(1.0, (((act - 1.0) * 20.0) + floor) / 60.0))
    return (2.0 * ((0.65 * progress) + (0.35 * hp_fraction))) - 1.0


def _number(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
