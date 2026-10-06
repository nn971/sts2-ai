from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass

from sts2_ai.emulator import (
    EmulatorBackend,
    ExpandableEmulatorBackend,
    InformationPolicy,
    ReleasableEmulatorBackend,
    StateHandle,
    Transition,
)
from sts2_ai.search.base import ActionEvaluation, SearchBudget, SearchResult
from sts2_ai.search.evaluator import StateEvaluator


@dataclass(frozen=True, slots=True)
class _FrontierNode:
    state: StateHandle
    root_action_id: str
    depth: int
    value: float
    terminal: bool


class BestFirstSearch:
    """Small deterministic best-first lookahead baseline.

    Search always evaluates every legal root action once before spending the remaining budget on
    descendants. This makes the returned root comparison complete even when the deeper node budget
    is small. Exact-state hashes suppress duplicate expansion within each root-action subtree.

    When the backend advertises sibling expansion, search uses it directly. Temporary state handles
    are released automatically when the backend supports explicit lifetime management.
    """

    VERSION = "best-first-v1"

    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        evaluator: StateEvaluator,
        information_policy: InformationPolicy,
    ) -> None:
        self._backend = backend
        self._evaluator = evaluator
        self._policy = information_policy

    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult:
        _validate_budget(budget)

        root_hash = self._backend.exact_hash(state)
        created_states: list[StateHandle] = []

        try:
            root_transitions = self._expand(state)
            created_states.extend(item.child for item in root_transitions)
            if not root_transitions:
                return SearchResult(
                    root_state_hash=root_hash,
                    evaluations=(),
                    expanded_nodes=0,
                    search_version=self.VERSION,
                )

            root_actions = tuple(item.action for item in root_transitions)
            best_values = {action.action_id: float("-inf") for action in root_actions}
            visits = {action.action_id: 0 for action in root_actions}
            actions_by_id = {action.action_id: action for action in root_actions}

            if len(actions_by_id) != len(root_actions):
                raise ValueError("Root legal-action IDs must be unique")

            start = time.monotonic()
            frontier: list[tuple[float, int, _FrontierNode]] = []
            tie_breaker = itertools.count()
            seen: set[tuple[str, str]] = set()
            expanded_nodes = 0

            # Root coverage is mandatory: every legal choice receives at least one comparable value.
            for transition in root_transitions:
                action = transition.action
                child_hash = self._transition_hash(transition)
                value = self._value(transition.child, transition.terminal)

                expanded_nodes += 1
                visits[action.action_id] += 1
                best_values[action.action_id] = max(best_values[action.action_id], value)
                seen.add((action.action_id, child_hash))

                if not transition.terminal and _can_descend(1, budget):
                    node = _FrontierNode(
                        state=transition.child,
                        root_action_id=action.action_id,
                        depth=1,
                        value=value,
                        terminal=False,
                    )
                    heapq.heappush(
                        frontier,
                        (-value, next(tie_breaker), node),
                    )

            while frontier:
                if budget.max_nodes is not None and expanded_nodes >= budget.max_nodes:
                    break
                if (
                    budget.max_seconds is not None
                    and time.monotonic() - start >= budget.max_seconds
                ):
                    break

                _, _, node = heapq.heappop(frontier)
                if node.terminal or not _can_descend(node.depth, budget):
                    continue

                transitions = self._expand(node.state)
                created_states.extend(item.child for item in transitions)

                for transition in transitions:
                    if budget.max_nodes is not None and expanded_nodes >= budget.max_nodes:
                        break
                    if (
                        budget.max_seconds is not None
                        and time.monotonic() - start >= budget.max_seconds
                    ):
                        break

                    child_hash = self._transition_hash(transition)
                    key = (node.root_action_id, child_hash)
                    if key in seen:
                        continue
                    seen.add(key)

                    value = self._value(transition.child, transition.terminal)
                    expanded_nodes += 1
                    visits[node.root_action_id] += 1
                    best_values[node.root_action_id] = max(
                        best_values[node.root_action_id],
                        value,
                    )

                    child_depth = node.depth + 1
                    if not transition.terminal and _can_descend(child_depth, budget):
                        child = _FrontierNode(
                            state=transition.child,
                            root_action_id=node.root_action_id,
                            depth=child_depth,
                            value=value,
                            terminal=False,
                        )
                        heapq.heappush(
                            frontier,
                            (-value, next(tie_breaker), child),
                        )

            evaluations = tuple(
                ActionEvaluation(
                    action=actions_by_id[action_id],
                    value=best_values[action_id],
                    visits=visits[action_id],
                )
                for action_id in sorted(actions_by_id)
            )

            return SearchResult(
                root_state_hash=root_hash,
                evaluations=evaluations,
                expanded_nodes=expanded_nodes,
                search_version=self.VERSION,
            )
        finally:
            if created_states and isinstance(self._backend, ReleasableEmulatorBackend):
                self._backend.release_many(created_states)

    def _expand(self, state: StateHandle) -> tuple[Transition, ...]:
        if isinstance(self._backend, ExpandableEmulatorBackend):
            return tuple(self._backend.expand(state))

        return tuple(
            self._backend.step(state, action)
            for action in self._backend.legal_actions(state)
        )

    def _transition_hash(self, transition: Transition) -> str:
        if transition.exact_hash is not None:
            return transition.exact_hash
        return self._backend.exact_hash(transition.child)

    def _value(self, state: StateHandle, terminal: bool) -> float:
        observation = self._backend.observe(state, self._policy)
        return self._evaluator.evaluate(observation, terminal=terminal)


def _can_descend(depth: int, budget: SearchBudget) -> bool:
    return budget.max_depth is None or depth < budget.max_depth


def _validate_budget(budget: SearchBudget) -> None:
    if budget.max_nodes is not None and budget.max_nodes <= 0:
        raise ValueError("max_nodes must be positive when set")
    if budget.max_seconds is not None and budget.max_seconds <= 0:
        raise ValueError("max_seconds must be positive when set")
    if budget.max_depth is not None and budget.max_depth <= 0:
        raise ValueError("max_depth must be positive when set")
