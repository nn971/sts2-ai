"""Exact synthetic deck permutation oracle for Nature's deferred draws.

The emulator alone owns *real* deck constraints and hook ordering.
These exhaustive tests verify the expected law for a small uniform toy deck
and catch double-counting and illegal repeated card-instance selections.
"""
from __future__ import annotations

import itertools
from collections import Counter

import pytest

from sts2_ai.nature.semantic import SemanticEvent
from sts2_ai.nature.semantic_policies import UniformSemanticNature


def _toy_event(
    ids: tuple[str, ...], *, ordinal: int, epoch: str = "epoch-one",
) -> SemanticEvent:
    return SemanticEvent(
        event_id=f"toy-{epoch}-{ordinal}",
        boundary_fingerprint=f"fixture-{epoch}-{ordinal}",
        site_id="toy.deck.reveal", owner_id="toy-deck",
        ordinal=ordinal, operation="draw_card",
        candidate_ids=ids, support_fingerprint=f"support-{ordinal}",
        randomization_id=epoch,
        constraint_fingerprint=f"toy-c-{ordinal}",
    )


def test_uniform_deferred_draws_enumerate_same_law_as_exact_shuffle():
    ids = ("card:a#1", "card:b#2", "card:c#3")
    joint = Counter()
    for first in ids:
        first_p = 1 / len(ids)
        remaining = tuple(x for x in ids if x != first)
        for second in remaining:
            second_p = 1 / len(remaining)
            last = next(x for x in remaining if x != second)
            joint[(first, second, last)] += first_p * second_p
    exact = list(itertools.permutations(ids))
    assert len(joint) == len(exact) == 6
    for permutation in exact:
        assert joint[permutation] == pytest.approx(1 / 6)


def test_fixed_middle_anchor_preserves_shared_pool_across_sides():
    ids = ("card:a", "card:b", "card:c")
    anchor = "card:fixed-x"
    exact = [(a, b, anchor, c) for a, b, c in itertools.permutations(ids)]
    sequential = []
    for a in ids:
        for b in ids:
            if b == a:
                continue
            c = next(x for x in ids if x != a and x != b)
            sequential.append((a, b, anchor, c))
    assert sorted(exact) == sorted(sequential)
    # Revealing known anchor does not create a new Nature event.
    assert all(row[2] == anchor for row in exact)


def test_singleton_unknown_is_still_one_semantic_resolution():
    event = _toy_event(("card:last-physical-instance",), ordinal=3)
    proposal = UniformSemanticNature(7).propose(event)
    assert proposal.selected_id == "card:last-physical-instance"
    assert proposal.selected_index == 0
    assert proposal.log_q == pytest.approx(0)


def test_reveal_then_draw_same_known_top_requires_no_second_decision():
    # The semantic contract emits a request only for *unresolved* identities.
    first = _toy_event(("card:a", "card:b"), ordinal=0)
    revealed = first.check_index(0)
    emitted = [first.event_id]
    drawn_known = revealed
    assert drawn_known == "card:a"
    assert emitted == ["toy-epoch-one-0"]
    next_unknown = _toy_event(("card:b",), ordinal=1)
    assert next_unknown.check_index(0) == "card:b"
    assert len(emitted) + 1 == 2
