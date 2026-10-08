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

from .targets import PolicyTarget, TrainingExample, build_training_example


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


def load_training_jsonl(path: Path) -> tuple[TrainingExample, ...]:
    """Load JSONL emitted by write_training_jsonl."""

    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Training record on line {line_number} is not an object"
                )
            raw_targets = raw.get("policy_targets")
            if not isinstance(raw_targets, list):
                raise ValueError(
                    f"Training record on line {line_number} has no policy_targets"
                )
            targets = []
            for raw_target in raw_targets:
                if not isinstance(raw_target, dict):
                    raise ValueError(
                        f"Policy target on line {line_number} is not an object"
                    )
                targets.append(
                    PolicyTarget(
                        action_id=str(raw_target["action_id"]),
                        probability=float(raw_target["probability"]),
                        action_kind=str(raw_target.get("action_kind", "")),
                        action_payload_json=str(
                            raw_target.get("action_payload_json", "{}")
                        ),
                        search_value=float(raw_target.get("search_value", 0.0)),
                        visits=int(raw_target.get("visits", 0)),
                    )
                )

            examples.append(
                TrainingExample(
                    observation_hash=str(raw["observation_hash"]),
                    information_policy=str(raw["information_policy"]),
                    policy_targets=tuple(targets),
                    value_target=float(raw["value_target"]),
                    source_search_id=str(raw["source_search_id"]),
                    emulator_revision=str(raw["emulator_revision"]),
                    observation_json=str(raw.get("observation_json", "{}")),
                    source_state_hash=str(raw.get("source_state_hash", "")),
                    search_regime=str(raw.get("search_regime", "oracle-exact")),
                    search_budget=int(raw.get("search_budget", 0)),
                    search_version=str(raw.get("search_version", "")),
                    game_build=str(raw.get("game_build", "unknown")),
                )
            )
    return tuple(examples)


def split_training_examples(
    examples: tuple[TrainingExample, ...],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    """Deterministic exact-state grouped split, independent of input order.

    All search budgets, versions and observation variants of one exact state
    remain in one partition, preventing repeated-root validation leakage.
    """
    import hashlib

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    groups: dict[str, list[TrainingExample]] = {}
    for example in examples:
        key = example.source_state_hash or example.observation_hash
        groups.setdefault(key, []).append(example)
    if len(groups) < 2:
        raise ValueError("At least two distinct states are required for validation")

    keys = sorted(
        groups,
        key=lambda key: (
            hashlib.sha256(f"{seed}:{key}".encode()).digest(),
            key,
        ),
    )
    count = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:count])
    def sorted_group(key: str) -> list[TrainingExample]:
        # Sort *inside* each exact-state group too. Otherwise reversing the
        # source record order changes downstream model training order even
        # though the partitions and group membership stay identical.
        return sorted(
            groups[key],
            key=lambda example: json.dumps(
                asdict(example), sort_keys=True, separators=(",", ":")
            ),
        )

    train = tuple(
        example
        for key in sorted(groups)
        if key not in validation_keys
        for example in sorted_group(key)
    )
    validation = tuple(
        example
        for key in sorted(groups)
        if key in validation_keys
        for example in sorted_group(key)
    )
    return train, validation
