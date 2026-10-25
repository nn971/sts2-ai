"""Disjoint source namespaces for H6 native Overgrowth macro PPO."""

from __future__ import annotations

import hashlib

H6_SOURCE_SCHEMA = "sts2-h6-macro-overgrowth-a10-source-v1"


def run_sources(
    split: str, count: int, *, study_seed: int = 43,
    round_index: int = 0,
) -> tuple[str, ...]:
    if split not in {"train", "selection", "confirmation"}:
        raise ValueError("Unknown H6 run source split")
    if (type(count) is not int or count <= 0
            or type(study_seed) is not int or study_seed < 0
            or type(round_index) is not int or round_index < 0):
        raise ValueError("Invalid H6 source-bank arguments")
    if split != "train" and round_index != 0:
        raise ValueError("Selection and confirmation may not depend on training round")
    # Do not construct seeds by post-hoc rejection on game outcomes.
    return tuple(
        f"h6-a10-{split}-v1-{study_seed:08d}-{round_index:05d}-{i:08d}"
        for i in range(count)
    )


def source_digest(sources: tuple[str, ...]) -> str:
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate H6 source seeds")
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()
