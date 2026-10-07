from __future__ import annotations

import json
from collections.abc import Sequence

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation, StateHandle
from sts2_ai.search import SearchBudget, UctMcts


class OracleMctsAgent:
    """Explicitly unfair exact-state MCTS agent used as the first search baseline."""

    def __init__(self, search: UctMcts, *, simulations: int) -> None:
        if simulations < 0:
            raise ValueError("simulations must be non-negative")
        self._search = search
        self._simulations = simulations

    def choose_state(
        self,
        state: StateHandle,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> Decision:
        if not legal_actions:
            raise ValueError("Cannot choose from an empty legal-action set")

        if self._simulations == 0:
            # Budget zero is defined as heuristic-only in the roadmap. The caller
            # should normally instantiate HeuristicAgent directly, but keep this
            # path deterministic and explicit.
            action = min(legal_actions, key=lambda item: item.action_id)
            return Decision(
                action=action,
                policy_name="oracle-exact-uct-v1-budget-0",
                metadata_json='{"search_transitions":0}',
            )

        result = self._search.search(
            state,
            SearchBudget(max_simulations=self._simulations),
        )
        if result.root_observation_hash != observation.observation_hash:
            raise RuntimeError("MCTS root observation does not match caller observation")

        legal_by_id = {action.action_id: action for action in legal_actions}
        candidates = [
            evaluation
            for evaluation in result.evaluations
            if evaluation.action.action_id in legal_by_id
        ]
        if not candidates:
            raise RuntimeError("MCTS returned no legal root action")

        best = max(
            candidates,
            key=lambda item: (item.visits, item.value, item.action.action_id),
        )
        metadata = {
            "search_transitions": result.transitions,
            "expanded_nodes": result.expanded_nodes,
            "transposition_hits": result.transposition_hits,
            "root_exact_hash": result.root_state_hash,
            "search_version": result.search_version,
            "simulations": self._simulations,
        }
        return Decision(
            action=legal_by_id[best.action.action_id],
            policy_name=f"oracle-exact-uct-v1-{self._simulations}",
            metadata_json=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        )
