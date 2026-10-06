from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from sts2_ai.evaluation import PrototypeSearchDecision

from .schema import ReplayScenario, ReplayScenarioAction

ARCHIVE_SCHEMA_ID = "prototype-replay-scenarios-v0"


def mine_close_search_scenarios(
    decisions: Sequence[PrototypeSearchDecision],
    *,
    limit: int,
    emulator_revision: str,
    information_policy: str,
    game_build: str = "prototype-unbound",
) -> tuple[ReplayScenario, ...]:
    """Keep the closest/highest-uncertainty searched decisions as replayable scenarios."""

    if limit < 0:
        raise ValueError("limit cannot be negative")
    if limit == 0:
        return ()

    candidates: list[ReplayScenario] = []
    for decision in decisions:
        if len(decision.evaluations) < 2:
            continue

        ordered = sorted(
            decision.evaluations,
            key=lambda item: (-item.value, item.action.action_id),
        )
        margin = max(0.0, ordered[0].value - ordered[1].value)
        uncertainties = [
            item.uncertainty
            for item in ordered
            if item.uncertainty is not None
        ]
        max_uncertainty = max(uncertainties) if uncertainties else None

        identity = (
            f"{emulator_revision}\0{game_build}\0{decision.seed}\0"
            f"{decision.state_hash}\0{'|'.join(decision.action_history)}"
        )
        scenario_id = hashlib.sha256(identity.encode()).hexdigest()[:20]

        candidates.append(
            ReplayScenario(
                scenario_id=scenario_id,
                run_seed=decision.seed,
                action_history=decision.action_history,
                decision_index=decision.decision_index,
                exact_state_hash=decision.state_hash,
                game_build=game_build,
                emulator_revision=emulator_revision,
                information_policy=information_policy,
                reason="close-search-decision",
                value_margin=margin,
                max_uncertainty=max_uncertainty,
                actions=tuple(
                    ReplayScenarioAction(
                        action_id=item.action.action_id,
                        kind=item.action.kind,
                        value=item.value,
                        visits=item.visits,
                        uncertainty=item.uncertainty,
                    )
                    for item in ordered
                ),
                tags=("search-mined", "close-decision"),
            )
        )

    candidates.sort(
        key=lambda scenario: (
            scenario.value_margin,
            -(scenario.max_uncertainty or 0.0),
            scenario.exact_state_hash,
        )
    )
    return tuple(candidates[:limit])


def write_replay_scenario_archive(
    path: Path,
    scenarios: Sequence[ReplayScenario],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": ARCHIVE_SCHEMA_ID,
        "count": len(scenarios),
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
