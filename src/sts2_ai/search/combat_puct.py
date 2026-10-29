"""Stochastic, information-set combat PUCT with policy/value head guidance.

Each simulation uses an independent *certified* fair hidden continuation;
the solver sees only public frames. Exact-state forks of the live run cannot
implement this protocol. Terminal rewards are the existing H9 total-nonminion
HP score in [0,2], internally divided by two to fit the H9 value-head scale.

This search core intentionally does not claim an H8 native chance adapter:
the caller must provide one with the explicit capability and verify its
conditioning against replayed public histories.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sts2_ai.emulator.protocol import LegalAction, Observation

FAIR_COMBAT_CAPABILITY = "h15-public-history-combat-chance-v1"
H15_SEARCH_VERSION = "h15-combat-puct-v1"


@dataclass(frozen=True)
class CombatFrame:
    information_key: str
    observation: Observation
    legal_actions: tuple[LegalAction, ...]
    terminal_reward: float | None = None

    def __post_init__(self) -> None:
        if not self.information_key:
            raise ValueError("Missing public-history identity")
        if len(set(a.action_id for a in self.legal_actions)) != len(self.legal_actions):
            raise ValueError("Duplicate legal action IDs")
        if self.terminal_reward is None and not self.legal_actions:
            raise ValueError("Nonterminal combat has no legal actions")
        if self.terminal_reward is not None:
            if (self.legal_actions or not math.isfinite(self.terminal_reward)
                    or not 0 <= self.terminal_reward <= 2):
                raise ValueError("Terminal must have no legal moves and a valid H9 reward")


@runtime_checkable
class FairCombatSampler(Protocol):
    @property
    def fair_combat_capability_id(self) -> str: ...

    def begin_simulation(self, root: CombatFrame, search_rng: random.Random) -> None:
        """Condition a fresh continuation on the root's FULL public history."""
        ...

    def step(self, frame: CombatFrame, action: LegalAction,
             search_rng: random.Random) -> CombatFrame:
        """Use the same sampled hidden continuation for this entire simulation."""
        ...

    def end_simulation(self) -> None:
        """Release the sampled hidden continuation even on failure."""
        ...


# Return logits aligned to the public legal menu, and value in [0,1].
CombatEvaluator = Callable[[Observation, tuple[LegalAction, ...]], tuple[tuple[float, ...], float]]


@dataclass
class _Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class _Node:
    frame: CombatFrame
    prior_value: float
    edges: list[_Edge] = field(default_factory=list)


@dataclass(frozen=True)
class CombatSearchResult:
    search_version: str
    information_key: str
    action_ids: tuple[str, ...]
    visits: tuple[int, ...]
    mean_action_values: tuple[float | None, ...]
    root_mean_value: float
    simulations: int
    terminal_evaluations: int
    neural_cutoffs: int
    transitions: int

    @property
    def policy(self) -> tuple[float, ...]:
        total = sum(self.visits)
        if total <= 0:
            raise ValueError("No visited root actions")
        return tuple(v / total for v in self.visits)


class CombatPuct:
    """A search-policy teacher, never a source of fictitious terminal labels."""

    def __init__(self, sampler: FairCombatSampler, evaluator: CombatEvaluator, *,
                 c_puct: float = 1.5, max_depth: int = 256, seed: int = 0):
        if (not isinstance(sampler, FairCombatSampler) or
                sampler.fair_combat_capability_id != FAIR_COMBAT_CAPABILITY):
            raise ValueError("H15 requires a certified fair-combat sampler")
        if not math.isfinite(c_puct) or c_puct <= 0 or max_depth <= 0:
            raise ValueError("Invalid combat search configuration")
        self._sampler = sampler
        self._evaluator = evaluator
        self._c = c_puct
        self._depth = max_depth
        self._rng = random.Random(seed)

    def _expand(self, frame: CombatFrame) -> _Node:
        logits, raw_value = self._evaluator(frame.observation, frame.legal_actions)
        if (len(logits) != len(frame.legal_actions)
                or not all(math.isfinite(v) for v in logits)
                or not math.isfinite(raw_value) or not 0 <= raw_value <= 1):
            raise ValueError("H15 evaluator must provide finite policy and normalized value")
        m = max(logits)
        weights = [math.exp(x - m) for x in logits]
        z = math.fsum(weights)
        if not math.isfinite(z) or z <= 0:
            raise ValueError("Invalid policy normalization")
        return _Node(frame, raw_value, [_Edge(w / z) for w in weights])

    def _choose(self, node: _Node) -> int:
        total = sum(e.visits for e in node.edges)
        scale = math.sqrt(1 + total)
        return max(range(len(node.edges)), key=lambda i: (
            (node.edges[i].q if node.edges[i].visits else node.prior_value)
            + self._c * node.edges[i].prior * scale / (1 + node.edges[i].visits),
            node.frame.legal_actions[i].action_id,
        ))

    def search(self, root: CombatFrame, *, simulations: int) -> CombatSearchResult:
        if root.terminal_reward is not None or simulations <= 0:
            raise ValueError("H15 requires nonterminal root and positive budget")
        table: dict[str, _Node] = {root.information_key: self._expand(root)}
        transitions = terminals = cutoffs = 0
        root_value_sum = 0.0
        for _ in range(simulations):
            path: list[_Edge] = []
            frame = root
            seen = {root.information_key}
            try:
                self._sampler.begin_simulation(root, self._rng)
                result: float | None = None
                for depth in range(self._depth + 1):
                    if frame.terminal_reward is not None:
                        result = frame.terminal_reward / 2.0
                        terminals += 1
                        break
                    node = table.get(frame.information_key)
                    if node is None:
                        node = self._expand(frame)
                        table[frame.information_key] = node
                        result = node.prior_value
                        cutoffs += 1
                        break
                    if node.frame != frame:
                        raise ValueError("Same public-history key has contradictory observations")
                    if depth == self._depth:
                        result = node.prior_value
                        cutoffs += 1
                        break
                    action_index = self._choose(node)
                    path.append(node.edges[action_index])
                    action = frame.legal_actions[action_index]
                    child = self._sampler.step(frame, action, self._rng)
                    transitions += 1
                    if not isinstance(child, CombatFrame):
                        raise TypeError("Fair combat sampler returned an invalid frame")
                    if (child.information_key in seen
                            or child.observation.policy_id != root.observation.policy_id):
                        raise ValueError("Nonadvancing or policy-changing public history")
                    seen.add(child.information_key)
                    frame = child
                if result is None:
                    raise AssertionError("Every simulation must return a value")
                root_value_sum += result
                for edge in path:
                    edge.visits += 1
                    edge.value_sum += result
            finally:
                self._sampler.end_simulation()
        edges = table[root.information_key].edges
        return CombatSearchResult(
            H15_SEARCH_VERSION, root.information_key,
            tuple(a.action_id for a in root.legal_actions),
            tuple(e.visits for e in edges),
            tuple(e.q if e.visits else None for e in edges),
            root_value_sum / simulations, simulations, terminals, cutoffs,
            transitions,
        )
