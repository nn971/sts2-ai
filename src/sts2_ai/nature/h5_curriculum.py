"""H5 reproducible, source-conditioned Silent A10 combat curriculum.

This *only* uses fixtures already implemented by the pinned semantic emulator.
A seed changes the actual shuffle and declared Chance-Law outcomes, not the
fixed deck/relic/HP configuration. We do not claim native run distributions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sts2_ai.nature.combat_scenarios import (
    DECK_PRESETS, SCENARIOS, deck_for_seed, scenario_for_seed,
)

SCHEMA = "sts2-ai-h5-curated-reference-curriculum-v1"


@dataclass(frozen=True, order=True)
class CombatCell:
    scenario: str
    deck: str
    tier: str

    @property
    def key(self) -> str:
        return f"{self.scenario}::{self.deck}"


_WEAK = ("shrinker-a10", "nibbits-a10")
_NORMAL = ("mawler-a10", "cubex-a10", "vine-shambler-a10")
_BOSS = ("ceremonial-beast-boss-a10", "vantom-boss-a10")
_WEAK_DECKS = (
    "silent-starter-v1", "silent-strike-heavy-v1",
    "silent-defend-heavy-v1",
)
_NORMAL_DECKS = (
    "silent-strike-heavy-v1", "silent-defend-heavy-v1",
    "silent-poison-kit-v1",
)
_BOSS_DECKS = (
    "silent-boss-cycle-20-v1", "silent-boss-poison-20-v1",
)

SUPPORTED_CELLS: tuple[CombatCell, ...] = tuple(sorted((
    *(CombatCell(s, d, "weak") for s in _WEAK for d in _WEAK_DECKS),
    *(CombatCell(s, d, "normal") for s in _NORMAL for d in _NORMAL_DECKS),
    *(CombatCell(s, d, "boss") for s in _BOSS for d in _BOSS_DECKS),
)))
# Combinations are held out entirely, not just their random source seeds.
# This is a modest cross-combination test, not an unseen-card/encounter test.
_ZERO_SHOT_KEYS = frozenset({
    "nibbits-a10::silent-defend-heavy-v1",
    "cubex-a10::silent-poison-kit-v1",
    "vantom-boss-a10::silent-boss-poison-20-v1",
})
TRAIN_CELLS = tuple(c for c in SUPPORTED_CELLS if c.key not in _ZERO_SHOT_KEYS)
ZERO_SHOT_CELLS = tuple(c for c in SUPPORTED_CELLS if c.key in _ZERO_SHOT_KEYS)
SCENARIO_POOL = tuple(sorted({c.scenario for c in SUPPORTED_CELLS}))
DECK_POOL = tuple(sorted({c.deck for c in SUPPORTED_CELLS}))
assert len(SUPPORTED_CELLS) == 19 and len(TRAIN_CELLS) == 16
assert len(ZERO_SHOT_CELLS) == 3
assert all(SCENARIOS[c.scenario].ascension == 10
           and c.deck in DECK_PRESETS for c in SUPPORTED_CELLS)


def cell_for_source(seed: str) -> CombatCell | None:
    """Return curated cell, or None for unsupported cross-product pairing."""
    scenario = scenario_for_seed(seed, SCENARIO_POOL).scenario_id
    deck = deck_for_seed(seed, DECK_POOL)
    return next(
        (cell for cell in SUPPORTED_CELLS
         if cell.scenario == scenario and cell.deck == deck),
        None,
    )


def source_bank(
    split: str, count: int, *, study_seed: int = 29,
    round_index: int = 0,
) -> tuple[str, ...]:
    """Preselect exactly balanced contexts without observing any game outcome.

    Selection and confirmation use new stochastic worlds from TRAIN_CELLS;
    zero-shot evaluates held-out scenario/deck *combinations*. Training
    rotates any remainder across its 16 cells. Namespaces are disjoint.
    """
    if (split not in {"train", "selection", "confirmation", "zero-shot"}
            or type(count) is not int or count < 1
            or type(study_seed) is not int or study_seed < 0
            or type(round_index) is not int or round_index < 0):
        raise ValueError("Invalid H5 source-bank request")
    if split != "train" and round_index:
        raise ValueError("Holdout seeds must not depend on training round")
    cells = TRAIN_CELLS if split != "zero-shot" else ZERO_SHOT_CELLS
    if split != "train" and count % len(cells):
        raise ValueError(
            f"{split} requires a multiple of {len(cells)} combats"
        )
    quota, extra = divmod(count, len(cells))
    quotas = {
        cell.key: quota + (1 if (i - round_index * extra) % len(cells) < extra
                             else 0)
        for i, cell in enumerate(cells)
    }
    found: Counter[str] = Counter()
    accepted: list[str] = []
    probe = 0
    # Deliberately bounded rejection over *source IDs*, never realized draws.
    while len(accepted) < count:
        if probe >= count * 512:
            raise RuntimeError("H5 source selection exceeded bounded search")
        seed = f"h5-{split}-v1-{study_seed}-{round_index}-{probe:09d}"
        probe += 1
        cell = cell_for_source(seed)
        if cell is not None and cell.key in quotas and (
            found[cell.key] < quotas[cell.key]
        ):
            found[cell.key] += 1
            accepted.append(seed)
    return tuple(accepted)


def source_cells(seeds: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for seed in seeds:
        cell = cell_for_source(seed)
        if cell is None:
            raise ValueError("Uncurated source world")
        counts[cell.key] += 1
    return dict(sorted(counts.items()))
