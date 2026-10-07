from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from sts2_ai.strategy_db import (
    SearchActionEvidence,
    SearchRootEvidence,
    SQLiteStrategyStore,
)

from .targets import TrainingExample, build_training_example


def build_training_examples(
    store: SQLiteStrategyStore,
    information_policy: str,
    *,
    search_regime: str = "oracle-exact",
    search_version: str | None = None,
    min_budget: int = 0,
    highest_budget_only: bool = True,
) -> tuple[TrainingExample, ...]:
    """Build deterministic supervised examples from persisted search evidence."""

    records = list(
        store.search_records(
            information_policy,
            search_regime=search_regime,
            search_version=search_version,
            min_budget=min_budget,
        )
    )
    if highest_budget_only:
        strongest: dict[
            tuple[str, str, str, str],
            tuple[SearchRootEvidence, tuple[SearchActionEvidence, ...]],
        ] = {}
        for root, actions in records:
            key = (
                root.state_hash,
                root.search_version,
                root.emulator_revision,
                root.game_build,
            )
            previous = strongest.get(key)
            if previous is None or root.search_budget > previous[0].search_budget:
                strongest[key] = (root, actions)
        records = list(strongest.values())

    examples = []
    for root, actions in records:
        observation = store.observation(
            root.observation_hash,
            root.information_policy,
        )
        if observation is None:
            raise RuntimeError(
                f"Missing observation payload for searched root {root.state_hash}"
            )
        examples.append(build_training_example(root, actions, observation))

    examples.sort(
        key=lambda example: (
            example.observation_hash,
            example.search_budget,
            example.source_search_id,
        )
    )
    return tuple(examples)


def write_training_jsonl(
    examples: Iterable[TrainingExample],
    output: Path,
) -> int:
    """Write canonical, line-delimited training records and return their count."""

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(
                json.dumps(
                    asdict(example),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count
