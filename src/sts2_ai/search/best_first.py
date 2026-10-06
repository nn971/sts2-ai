from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass

from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, StateHandle
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
    """

    VERSION = "best-first-v0"

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
        root_actions = tuple(self._backend.legal_actions(state))
        if not root_actions:
            return SearchResult(
                root_state_hash=root_hash,
                evaluations=(),
                expanded_nodes=0,
                search_version=self.VERSION,
            )

        start = time.monotonic()
        best_values = {action.action_id: float("-inf") for action in root_actions}
        visits = {action.action_id: 0 for action in root_actions}
        actions_by_id = {action.action_id: action for action in root_actions}

        if len(actions_by_id) != len(root_actions):
            raise ValueError("Root legal-action IDs must be unique")

        frontier: list[tuple[float, int, _FrontierNode]] = []
        tie_breaker = itertools.count()
        seen: set[tuple[str, str]] = set()
        expanded_nodes = 0

        # Root coverage is mandatory: every legal choice receives at least one comparable value.
        for action in root_actions:
            transition = self._backend.step(state, action)
            child_hash = self._backend.exact_hash(transition.child)
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
            if budget.max_seconds is not None and time.monotonic() - start >= budget.max_seconds:
                break

            _, _, node = heapq.heappop(frontier)
            if node.terminal or not _can_descend(node.depth, budget):
                continue

            legal_actions = self._backend.legal_actions(node.state)
            for action in legal_actions:
                if budget.max_nodes is not None and expanded_nodes >= budget.max_nodes:
                    break
                if (
                    budget.max_seconds is not None
                    and time.monotonic() - start >= budget.max_seconds
                ):
                    break

                transition = self._backend.step(node.state, action)
                child_hash = self._backend.exact_hash(transition.child)
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
