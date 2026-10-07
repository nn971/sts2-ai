from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
)

from .base import ActionEvaluation, SearchBudget, SearchResult

ValueFunction = Callable[[Observation], float]

_SEARCH_ALGORITHM_VERSION = "oracle-exact-light-rollout-uct-v8-configured"
_CUTOFF_VALUE_VERSION = "sts2-value-v2-progress-hp"


class _RolloutDecision(Protocol):
    @property
    def action(self) -> LegalAction: ...


class RolloutPolicy(Protocol):
    def choose(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> _RolloutDecision: ...


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


@dataclass(slots=True)
class _PendingSimulation:
    path_nodes: tuple[_Node, ...]
    path_edges: tuple[_Edge, ...]
    leaf: _Node | None
    immediate_value: float | None


@dataclass(frozen=True, slots=True)
class _Reservation:
    node_values: tuple[float, ...]
    edge_values: tuple[float, ...]


@dataclass(slots=True)
class _RolloutState:
    handle: StateHandle
    observation: Observation
    legal_actions: tuple[LegalAction, ...]
    value: float | None = None


class UctMcts:
    """Transposition-aware oracle-exact UCT with batched heuristic rollouts."""

    def __init__(
        self,
        backend: EmulatorBackend,
        *,
        policy: InformationPolicy,
        rollout_policy: RolloutPolicy,
        value_fn: ValueFunction | None = None,
        exploration: float = math.sqrt(2.0),
        rollout_depth: int = 128,
        rollout_batch_size: int = 8,
        virtual_loss: float | None = None,
        finish_combat_rollouts: bool = False,
        combat_extension_depth: int = 64,
        seed: int = 0,
    ) -> None:
        if rollout_depth < 0:
            raise ValueError("rollout_depth must be non-negative")
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if virtual_loss is not None and not -1.0 <= virtual_loss <= 1.0:
            raise ValueError("virtual_loss must lie in [-1, 1]")
        if combat_extension_depth < 0:
            raise ValueError("combat_extension_depth must be non-negative")
        self._backend = backend
        self._policy = policy
        self._rollout_policy = rollout_policy
        self._value_fn = value_fn or sts2_value
        self._exploration = exploration
        self._rollout_depth = rollout_depth
        self._rollout_batch_size = rollout_batch_size
        self._virtual_loss = virtual_loss
        self._finish_combat_rollouts = finish_combat_rollouts
        self._combat_extension_depth = combat_extension_depth
        self.search_version = _configured_search_version(
            rollout_policy=rollout_policy,
            value_fn=value_fn,
            exploration=exploration,
            rollout_depth=rollout_depth,
            rollout_batch_size=rollout_batch_size,
            virtual_loss=virtual_loss,
            finish_combat_rollouts=finish_combat_rollouts,
            combat_extension_depth=combat_extension_depth,
        )
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

        completed = 0
        try:
            while completed < simulations:
                if deadline is not None and time.monotonic() >= deadline:
                    break

                target = min(self._rollout_batch_size, simulations - completed)
                pending: list[tuple[_PendingSimulation, _Reservation]] = []

                for _ in range(target):
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    simulation = self._select_simulation(root, table, expand)
                    reservation = self._reserve(simulation, self._virtual_loss)
                    pending.append((simulation, reservation))

                if not pending:
                    break

                rollout_entries = [
                    simulation.leaf
                    for simulation, _ in pending
                    if simulation.leaf is not None
                ]
                rollout_values: tuple[float, ...] = ()
                if rollout_entries:
                    rollout_values, used = self._rollout_batch(tuple(rollout_entries))
                    transitions += used

                value_iter = iter(rollout_values)
                for simulation, reservation in pending:
                    value = simulation.immediate_value
                    if value is None:
                        value = next(value_iter)
                    self._backup_reserved(simulation, value, reservation)

                completed += len(pending)
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
            root_observation_json=root.observation.payload_json,
            evaluations=evaluations,
            expanded_nodes=sum(1 for node in table.values() if node.expanded),
            transitions=transitions,
            transposition_hits=transposition_hits,
            search_version=self.search_version,
        )

    def _select_simulation(
        self,
        root: _Node,
        table: dict[str, _Node],
        expand: Callable[[_Node], None],
    ) -> _PendingSimulation:
        node = root
        path_nodes = [root]
        path_edges: list[_Edge] = []

        while True:
            if node.terminal:
                return _PendingSimulation(
                    path_nodes=tuple(path_nodes),
                    path_edges=tuple(path_edges),
                    leaf=None,
                    immediate_value=self._value_fn(node.observation),
                )

            if not node.expanded:
                expand(node)
                if not node.edges:
                    return _PendingSimulation(
                        path_nodes=tuple(path_nodes),
                        path_edges=tuple(path_edges),
                        leaf=None,
                        immediate_value=self._value_fn(node.observation),
                    )
                edge = self._choose_unvisited_or_uct(node)
                path_edges.append(edge)
                node = table[edge.child_hash]
                path_nodes.append(node)
                if node.terminal:
                    return _PendingSimulation(
                        path_nodes=tuple(path_nodes),
                        path_edges=tuple(path_edges),
                        leaf=None,
                        immediate_value=self._value_fn(node.observation),
                    )
                return _PendingSimulation(
                    path_nodes=tuple(path_nodes),
                    path_edges=tuple(path_edges),
                    leaf=node,
                    immediate_value=None,
                )

            if not node.edges:
                return _PendingSimulation(
                    path_nodes=tuple(path_nodes),
                    path_edges=tuple(path_edges),
                    leaf=None,
                    immediate_value=self._value_fn(node.observation),
                )

            edge = self._choose_unvisited_or_uct(node)
            was_unvisited = edge.visits == 0
            path_edges.append(edge)
            node = table[edge.child_hash]
            path_nodes.append(node)

            if node.terminal:
                return _PendingSimulation(
                    path_nodes=tuple(path_nodes),
                    path_edges=tuple(path_edges),
                    leaf=None,
                    immediate_value=self._value_fn(node.observation),
                )
            if was_unvisited:
                return _PendingSimulation(
                    path_nodes=tuple(path_nodes),
                    path_edges=tuple(path_edges),
                    leaf=node,
                    immediate_value=None,
                )

    @staticmethod
    def _reserve(
        simulation: _PendingSimulation,
        virtual_loss: float | None,
    ) -> _Reservation:
        # Mean-preserving reservations are the default: the temporary visit gets
        # the current estimate of that node/edge, so Q is unchanged while the UCT
        # exploration bonus falls. A fixed virtual_loss can still be supplied for
        # experiments that want explicit optimism or pessimism.
        node_values = tuple(
            (
                virtual_loss
                if virtual_loss is not None
                else _mean_or_zero(node.value_sum, node.visits)
            )
            for node in simulation.path_nodes
        )
        edge_values = tuple(
            (
                virtual_loss
                if virtual_loss is not None
                else (
                    _mean_or_zero(edge.value_sum, edge.visits)
                    if edge.visits > 0
                    else node_values[index]
                )
            )
            for index, edge in enumerate(simulation.path_edges)
        )

        for node, temporary_value in zip(
            simulation.path_nodes,
            node_values,
            strict=True,
        ):
            node.visits += 1
            node.value_sum += temporary_value
        for edge, temporary_value in zip(
            simulation.path_edges,
            edge_values,
            strict=True,
        ):
            edge.visits += 1
            edge.value_sum += temporary_value

        return _Reservation(
            node_values=node_values,
            edge_values=edge_values,
        )

    @staticmethod
    def _backup_reserved(
        simulation: _PendingSimulation,
        value: float,
        reservation: _Reservation,
    ) -> None:
        # Replace each temporary reservation value by the actual rollout value.
        for node, temporary_value in zip(
            simulation.path_nodes,
            reservation.node_values,
            strict=True,
        ):
            node.value_sum += value - temporary_value
        for edge, temporary_value in zip(
            simulation.path_edges,
            reservation.edge_values,
            strict=True,
        ):
            edge.value_sum += value - temporary_value

    def _zero_budget_result(self, state: StateHandle) -> SearchResult:
        node = self._make_node(state)
        return SearchResult(
            root_state_hash=node.exact_hash,
            root_observation_hash=node.observation.observation_hash,
            root_observation_json=node.observation.payload_json,
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
            ordered = sorted(unvisited, key=lambda edge: edge.action.action_id)
            return self._rng.choice(ordered)

        log_parent = math.log(max(1, node.visits))
        return max(
            node.edges.values(),
            key=lambda edge: (
                (edge.value_sum / edge.visits)
                + self._exploration * math.sqrt(log_parent / edge.visits),
                edge.action.action_id,
            ),
        )

    def _rollout_batch(self, starts: tuple[_Node, ...]) -> tuple[tuple[float, ...], int]:
        rollouts = [
            _RolloutState(
                handle=start.handle,
                observation=start.observation,
                legal_actions=start.legal_actions,
                value=self._value_fn(start.observation) if start.terminal else None,
            )
            for start in starts
        ]
        temporary: list[StateHandle] = []
        transitions_used = 0

        try:
            maximum_steps = self._rollout_depth + (
                self._combat_extension_depth
                if self._finish_combat_rollouts
                else 0
            )
            for step in range(maximum_steps):
                active_indices = []
                for index, rollout in enumerate(rollouts):
                    if rollout.value is not None:
                        continue
                    if not rollout.legal_actions:
                        rollout.value = self._value_fn(rollout.observation)
                        continue

                    within_base_horizon = step < self._rollout_depth
                    extending_combat = (
                        self._finish_combat_rollouts
                        and step >= self._rollout_depth
                        and _is_combat_observation(rollout.observation)
                    )
                    if within_base_horizon or extending_combat:
                        active_indices.append(index)
                    else:
                        rollout.value = self._value_fn(rollout.observation)

                if not active_indices:
                    break

                requests: list[tuple[StateHandle, LegalAction]] = []
                for index in active_indices:
                    rollout = rollouts[index]
                    decision = self._rollout_policy.choose(
                        rollout.observation,
                        rollout.legal_actions,
                    )
                    requests.append((rollout.handle, decision.action))

                frames = self._backend.batch_rollout_step_frame(
                    requests,
                    self._policy,
                )
                transitions_used += len(frames)
                temporary.extend(frame.transition.child for frame in frames)

                for index, frame in zip(
                    active_indices,
                    frames,
                    strict=True,
                ):
                    rollout = rollouts[index]
                    rollout.handle = frame.transition.child
                    rollout.observation = frame.observation
                    rollout.legal_actions = (
                        ()
                        if frame.transition.terminal
                        else frame.legal_actions
                    )
                    if frame.transition.terminal:
                        rollout.value = self._value_fn(frame.observation)

            for rollout in rollouts:
                if rollout.value is None:
                    rollout.value = self._value_fn(rollout.observation)

            return (
                tuple(cast(float, rollout.value) for rollout in rollouts),
                transitions_used,
            )
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


def _configured_search_version(
    *,
    rollout_policy: RolloutPolicy,
    value_fn: ValueFunction | None,
    exploration: float,
    rollout_depth: int,
    rollout_batch_size: int,
    virtual_loss: float | None,
    finish_combat_rollouts: bool,
    combat_extension_depth: int,
) -> str:
    rollout_id = getattr(
        rollout_policy,
        "policy_id",
        f"{type(rollout_policy).__module__}.{type(rollout_policy).__qualname__}",
    )
    actual_value_fn = value_fn or sts2_value
    value_id = (
        _CUTOFF_VALUE_VERSION
        if value_fn is None
        else getattr(
            actual_value_fn,
            "value_id",
            getattr(actual_value_fn, "__name__", type(actual_value_fn).__qualname__),
        )
    )
    return (
        f"{_SEARCH_ALGORITHM_VERSION}"
        f"|depth={rollout_depth}"
        f"|batch={rollout_batch_size}"
        f"|vl={_virtual_loss_id(virtual_loss)}"
        f"|finish-combat={int(finish_combat_rollouts)}"
        f"|combat-ext={combat_extension_depth}"
        f"|c={exploration:.6g}"
        f"|rollout={rollout_id}"
        f"|value={value_id}"
    )


def _is_combat_observation(observation: Observation) -> bool:
    try:
        raw = json.loads(observation.payload_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict):
        return False
    phase = raw.get("phase")
    return phase == 3 or phase == "Combat"


def _virtual_loss_id(virtual_loss: float | None) -> str:
    return "mean" if virtual_loss is None else f"{virtual_loss:.6g}"


def _mean_or_zero(value_sum: float, visits: int) -> float:
    return value_sum / visits if visits > 0 else 0.0


def sts2_value(observation: Observation) -> float:
    """Transparent cutoff value for the fixed three-act prototype ruleset.

    A combat on floor f is treated as progress from floor f-1 toward floor f.
    Remaining enemy HP supplies the within-combat progress signal that the first
    evaluator lacked, while player HP remains the survival signal.
    """

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

    # The current prototype ruleset has 3 acts x 6 floors. Keeping this explicit
    # is preferable to the old 20-floor reporting convention, which compressed
    # all of Act 1 into only one tenth of the value range.
    floors_per_act = 6.0
    total_acts = 3.0
    act = max(1.0, min(total_acts, _number(state.get("act"), 1.0)))
    floor = max(0.0, min(floors_per_act, _number(state.get("floor"), 0.0)))

    combat = state.get("combat")
    if isinstance(combat, dict) and floor > 0.0:
        enemies = _dict_items(combat.get("enemies"))
        remaining_enemy_hp = sum(
            max(0.0, _number(enemy.get("hp"), 0.0))
            for enemy in enemies
        )
        # Smoothly interpolates from roughly the previous completed floor at the
        # beginning of a fight to the current floor as enemy HP approaches zero.
        combat_completion = math.exp(-remaining_enemy_hp / 80.0)
        effective_floor = max(0.0, floor - 1.0 + combat_completion)
    else:
        effective_floor = floor

    progress = (
        ((act - 1.0) * floors_per_act) + effective_floor
    ) / (total_acts * floors_per_act)
    utility = (0.70 * progress) + (0.30 * hp_fraction)
    return max(-1.0, min(1.0, (2.0 * utility) - 1.0))


def _dict_items(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        cast(dict[str, Any], item)
        for item in value
        if isinstance(item, dict)
    )


def _number(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
