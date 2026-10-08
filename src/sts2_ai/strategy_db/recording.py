from __future__ import annotations

from sts2_ai.emulator import LegalAction
from sts2_ai.search import SearchResult

from .schema import SearchActionEvidence, SearchObservationEvidence, SearchRootEvidence
from .sqlite_store import SQLiteStrategyStore


def record_search_result(
    store: SQLiteStrategyStore,
    result: SearchResult,
    chosen_action: LegalAction,
    *,
    information_policy: str,
    search_regime: str,
    search_budget: int,
    emulator_revision: str,
    game_build: str,
    model_id: str | None = None,
    run_seed: str | None = None,
) -> None:
    store.upsert_search_observation(
        SearchObservationEvidence(
            observation_hash=result.root_observation_hash,
            information_policy=information_policy,
            payload_json=result.root_observation_json,
        )
    )

    root = SearchRootEvidence(
        state_hash=result.root_state_hash,
        observation_hash=result.root_observation_hash,
        information_policy=information_policy,
        search_regime=search_regime,
        legal_action_ids=tuple(
            evaluation.action.action_id for evaluation in result.evaluations
        ),
        chosen_action_id=chosen_action.action_id,
        search_budget=search_budget,
        expanded_nodes=result.expanded_nodes,
        transitions=result.transitions,
        transposition_hits=result.transposition_hits,
        search_version=result.search_version,
        model_id=model_id,
        emulator_revision=emulator_revision,
        game_build=game_build,
        rollout_count=result.rollout_count,
        terminal_rollouts=result.terminal_rollouts,
        boundary_rollouts=result.boundary_rollouts,
        cutoff_rollouts=result.cutoff_rollouts,
        rollout_steps=result.rollout_steps,
    )
    actions = tuple(
        SearchActionEvidence(
            state_hash=result.root_state_hash,
            information_policy=information_policy,
            search_regime=search_regime,
            action_id=evaluation.action.action_id,
            action_kind=evaluation.action.kind,
            action_payload_json=evaluation.action.payload_json,
            value=evaluation.value,
            visits=evaluation.visits,
            uncertainty=evaluation.uncertainty,
            search_budget=search_budget,
            search_version=result.search_version,
            model_id=model_id,
            emulator_revision=emulator_revision,
            game_build=game_build,
        )
        for evaluation in result.evaluations
    )
    store.upsert_search(root, actions)
    if run_seed is not None:
        store.add_search_origin(root, run_seed)
