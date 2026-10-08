"""Player-information-keyed stochastic PUCT core with explicit chance sampling.

This module has no access to exact emulator state handles. An adapter must
certify that its transition distribution is conditioned on observable history;
the existing oracle-exact JSONL backend is intentionally ineligible.

A simulation returns a *terminal* success and optional bounded exploratory
auxiliaries. Censored/depth-capped paths fail closed; no fictitious defeat
labels or hidden-seed-conditioned search targets enter the statistics.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.training.bounded_targets import OnlineBoundedMoments

FAIR_TRANSITION_CAPABILITY_ID = "public-history-chance-transition-v1"


class FairPuctUnavailable(RuntimeError):
    """Attempted to run fair search without a verified chance-model adapter."""


class IncompleteChanceRollout(RuntimeError):
    """Cannot label an uncompleted stochastic simulation with success/auxiliary values."""


@dataclass(frozen=True, slots=True)
class TerminalReturn:
    success: bool
    progress: float | None = None
    hp: float | None = None

    def __post_init__(self) -> None:
        for x in (self.progress, self.hp):
            if x is not None and (not math.isfinite(x) or not 0.0 <= x <= 1.0):
                raise ValueError("Terminal auxiliaries must be normalized to [0, 1]")


@dataclass(frozen=True, slots=True)
class PublicSearchNode:
    """Full observable action/reveal history identity, independent of hidden seed."""

    information_key: str
    observation: Observation
    legal_actions: tuple[LegalAction, ...]
    terminal: TerminalReturn | None = None

    def __post_init__(self) -> None:
        if not self.information_key:
            raise ValueError("A full-public-history key is required")
        if len({action.action_id for action in self.legal_actions}) != len(self.legal_actions):
            raise ValueError("Legal actions have duplicate action IDs")
        if self.terminal is not None and self.legal_actions:
            raise ValueError("Terminal states cannot expose legal actions")


@runtime_checkable
class FairChanceTransitionModel(Protocol):
    """A validated adapter that samples exact *public-history-conditional* laws."""

    @property
    def fair_transition_capability_id(self) -> str: ...

    def sample_transition(
        self,
        node: PublicSearchNode,
        action: LegalAction,
        search_rng: random.Random,
    ) -> PublicSearchNode: ...


@dataclass(slots=True)
class _RunningOutcome:
    success: OnlineBoundedMoments = field(default_factory=OnlineBoundedMoments)
    progress: OnlineBoundedMoments = field(default_factory=OnlineBoundedMoments)
    hp: OnlineBoundedMoments = field(default_factory=OnlineBoundedMoments)

    def record(self, outcome: TerminalReturn) -> None:
        self.success.add(float(outcome.success))
        if outcome.progress is not None:
            self.progress.add(outcome.progress)
        if outcome.hp is not None:
            self.hp.add(outcome.hp)


@dataclass(slots=True)
class _Edge:
    action: LegalAction
    prior: float
    results: _RunningOutcome = field(default_factory=_RunningOutcome)


@dataclass(slots=True)
class _Node:
    frame: PublicSearchNode
    edges: tuple[_Edge, ...]


@dataclass(frozen=True, slots=True)
class PuctActionStatistics:
    action: LegalAction
    prior: float
    visits: int
    success_mean: float | None
    progress_mean: float | None
    progress_uniformized_variance: float | None
    hp_mean: float | None
    hp_uniformized_variance: float | None


@dataclass(frozen=True, slots=True)
class PuctSearchStatistics:
    information_key: str
    search_version: str
    simulations: int
    sampled_transitions: int
    expanded_information_sets: int
    actions: tuple[PuctActionStatistics, ...]


# Called only with an observation+legal actions; action-instance IDs are opaque.
PriorProvider = Callable[[Observation, tuple[LegalAction, ...]], tuple[float, ...]]


class StochasticPuct:
    """Single-player search over public *histories*, with chance outcomes sampled."""

    search_version = "fair-stochastic-puct-kernel-v1"

    def __init__(
        self,
        model: FairChanceTransitionModel,
        *,
        prior_provider: PriorProvider | None = None,
        exploration: float = 1.5,
        max_depth: int = 128,
        seed: int = 0,
    ) -> None:
        if (
            not isinstance(model, FairChanceTransitionModel)
            or model.fair_transition_capability_id != FAIR_TRANSITION_CAPABILITY_ID
        ):
            raise FairPuctUnavailable(
                "A validated public-history chance transition adapter is required; "
                "forked exact emulator states are oracle-only."
            )
        if not math.isfinite(exploration) or exploration <= 0:
            raise ValueError("exploration must be positive and finite")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self._model = model
        self._prior_provider = prior_provider
        self._exploration = exploration
        self._max_depth = max_depth
        self._rng = random.Random(seed)

    def _priors(self, node: PublicSearchNode) -> tuple[float, ...]:
        if not node.legal_actions:
            raise IncompleteChanceRollout("Nonterminal information state has no legal actions")
        if self._prior_provider is None:
            size = len(node.legal_actions)
            return tuple(1.0 / size for _ in node.legal_actions)
        values = self._prior_provider(node.observation, node.legal_actions)
        if len(values) != len(node.legal_actions):
            raise ValueError("Prior length does not match legal actions")
        if any(not math.isfinite(p) or p < 0.0 for p in values):
            raise ValueError("Action priors must be nonnegative and finite")
        total = math.fsum(values)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Action priors must sum to one")
        return tuple(p / total for p in values)

    def _make_node(self, frame: PublicSearchNode) -> _Node:
        probs = self._priors(frame)
        return _Node(
            frame,
            tuple(
                _Edge(action, p)
                for action, p in zip(frame.legal_actions, probs, strict=True)
            ),
        )

    def _choose(self, node: _Node) -> _Edge:
        total = sum(e.results.success.count for e in node.edges)
        sqrt_total = math.sqrt(max(1, total))
        return max(
            node.edges,
            key=lambda edge: (
                (
                    (edge.results.success.mean if edge.results.success.count else 0.5)
                    + self._exploration * edge.prior * sqrt_total
                    / (1 + edge.results.success.count)
                ),
                edge.action.action_id,
            ),
        )

    def _weighted_rollout_action(self, node: PublicSearchNode) -> LegalAction:
        probs = self._priors(node)
        index = self._rng.choices(range(len(probs)), weights=probs, k=1)[0]
        return node.legal_actions[index]

    def _next(
        self, frame: PublicSearchNode, action: LegalAction, path: set[str]
    ) -> PublicSearchNode:
        child = self._model.sample_transition(frame, action, self._rng)
        if not isinstance(child, PublicSearchNode):
            raise ValueError("Chance model must return a public search node")
        if child.observation.policy_id != frame.observation.policy_id:
            raise ValueError("Chance model changed the information policy")
        if child.information_key in path:
            raise ValueError("Public-history keys must advance on transitions")
        return child

    @staticmethod
    def _check_consistent(old: PublicSearchNode, new: PublicSearchNode) -> None:
        if (
            old.observation != new.observation
            or old.legal_actions != new.legal_actions
            or old.terminal != new.terminal
        ):
            raise ValueError("Inconsistent chance outcomes share a public-history key")

    def _simulation(
        self, root: PublicSearchNode, table: dict[str, _Node]
    ) -> tuple[TerminalReturn, tuple[_Edge, ...], int]:
        current = root
        path_keys = {root.information_key}
        selected: list[_Edge] = []
        transitions = 0
        in_rollout = False

        for _ in range(self._max_depth + 1):
            if current.terminal is not None:
                return current.terminal, tuple(selected), transitions
            if transitions >= self._max_depth:
                break

            tree_node = table.get(current.information_key)
            if tree_node is None:
                table[current.information_key] = self._make_node(current)
                in_rollout = True
            else:
                self._check_consistent(tree_node.frame, current)

            if in_rollout:
                action = self._weighted_rollout_action(current)
            else:
                edge = self._choose(tree_node)
                selected.append(edge)
                action = edge.action
            current = self._next(current, action, path_keys)
            transitions += 1
            path_keys.add(current.information_key)

        raise IncompleteChanceRollout(
            "A chance simulation reached max_depth before an observable terminal; "
            "increase the horizon or add a separately versioned trained cutoff."
        )

    def search(self, root: PublicSearchNode, *, simulations: int) -> PuctSearchStatistics:
        if type(simulations) is not int or simulations <= 0:
            raise ValueError("simulations must be positive")
        if root.terminal is not None:
            raise ValueError("PUCT requires a nonterminal decision root")
        table = {root.information_key: self._make_node(root)}
        transitions = 0
        for _ in range(simulations):
            outcome, selected, count = self._simulation(root, table)
            transitions += count
            for edge in selected:
                edge.results.record(outcome)

        rows: list[PuctActionStatistics] = []
        for edge in table[root.information_key].edges:
            success = edge.results.success
            progress = edge.results.progress
            hp = edge.results.hp
            rows.append(
                PuctActionStatistics(
                    action=edge.action,
                    prior=edge.prior,
                    visits=success.count,
                    success_mean=success.mean if success.count else None,
                    progress_mean=progress.mean if progress.count else None,
                    progress_uniformized_variance=(
                        progress.estimate(min_variance_samples=2).uniformized_variance
                        if progress.count >= 2 else None
                    ),
                    hp_mean=hp.mean if hp.count else None,
                    hp_uniformized_variance=(
                        hp.estimate(min_variance_samples=2).uniformized_variance
                        if hp.count >= 2 else None
                    ),
                )
            )
        return PuctSearchStatistics(
            information_key=root.information_key,
            search_version=self.search_version,
            simulations=simulations,
            sampled_transitions=transitions,
            expanded_information_sets=len(table),
            actions=tuple(rows),
        )
