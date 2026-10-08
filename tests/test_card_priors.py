"""All counts in this module are synthetic; never production Skada rankings."""
from __future__ import annotations

import json
import math

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.search.card_priors import (
    CARD_PRIOR_FORMAT,
    CardPriorContext,
    CardPriorDataset,
    blended_card_reward_prior,
)

CONTEXT = CardPriorContext(
    game_build="synthetic-test-build", character="Silent", act=1, ascension=0
)


def dataset() -> CardPriorDataset:
    return CardPriorDataset.from_dict({
        "format": CARD_PRIOR_FORMAT,
        "provenance": {
            "source": "synthetic-fixture",
            "source_url": "https://example.org/fixtures",
            "collected_at": "2026-10-08",
            "game_build": "synthetic-test-build",
            "character": "Silent",
            "act": 1,
            "ascension_min": 0,
            "ascension_max": 0,
            "methodology": "Synthetic offered/picked counts; not real-game data",
        },
        "cards": [
            {"card_id": "A", "offered": 100, "picked": 90},
            {"card_id": "B", "offered": 100, "picked": 10},
        ],
    })


def reward() -> tuple[Observation, tuple[LegalAction, ...]]:
    obs = Observation(
        "prototype-fair-v0",
        json.dumps({"act": 1, "reward": {"card_options": ["A", "B", "C"]}}),
        "visible-reward",
    )
    actions = (
        LegalAction("0", "take_reward_card", '{"index":0}'),
        LegalAction("1", "take_reward_card", '{"index":1}'),
        LegalAction("2", "take_reward_card", '{"index":2}'),
        LegalAction("skip", "skip_reward_card", "{}"),
    )
    return obs, actions


def test_human_prior_ranks_known_cards_and_leaves_unknown_and_skip_neutral() -> None:
    obs, actions = reward()
    result = blended_card_reward_prior(
        obs, actions, (0.0,) * 4, context=CONTEXT, dataset=dataset(),
        human_weight=1.0,
    )
    assert result.used_external_data
    assert result.matched_cards == 2
    assert result.dataset_fingerprint == dataset().fingerprint
    assert result.probabilities[0] > result.probabilities[2]
    assert result.probabilities[2] == pytest.approx(result.probabilities[3])
    assert result.probabilities[2] > result.probabilities[1]
    assert sum(result.probabilities) == pytest.approx(1)


def test_zero_weight_and_version_mismatch_leave_neural_policy_unchanged() -> None:
    obs, actions = reward()
    logits = (2.0, 0.0, -2.0, 0.0)
    off = blended_card_reward_prior(
        obs, actions, logits, context=CONTEXT, dataset=dataset(), human_weight=0
    )
    mismatch = blended_card_reward_prior(
        obs, actions, logits,
        context=CardPriorContext("different-build", "Silent", 1, 0),
        dataset=dataset(), human_weight=0.8,
    )
    assert not off.used_external_data
    assert off.probabilities == mismatch.probabilities
    assert off.probabilities[0] > off.probabilities[1]


def test_nonreward_choices_and_unknown_cards_unaffected() -> None:
    obs, actions = reward()
    nonreward = (
        actions[0],
        LegalAction("end", "end_turn"),
    )
    result = blended_card_reward_prior(
        obs, nonreward, (0.0, 0.0),
        context=CONTEXT, dataset=dataset(), human_weight=1,
    )
    assert result.probabilities == (0.5, 0.5)
    assert not result.used_external_data
    unknown = blended_card_reward_prior(
        obs, actions[2:], (0.0, 0.0),
        context=CONTEXT, dataset=dataset(), human_weight=1,
    )
    assert unknown.probabilities == (0.5, 0.5)
    assert not unknown.used_external_data


def test_smoothing_limits_spurious_small_sample_preference() -> None:
    raw = {
        "format": CARD_PRIOR_FORMAT,
        "provenance": {
            "source": "synthetic", "source_url": "https://example.org",
            "collected_at": "2026-10-08", "game_build": "synthetic-test-build",
            "character": "Silent", "act": 1, "ascension_min": 0,
            "ascension_max": 0, "methodology": "toy",
        },
        "cards": [
            {"card_id": "A", "offered": 1, "picked": 1},
            {"card_id": "B", "offered": 100, "picked": 50},
        ],
    }
    obs, actions = reward()
    result = blended_card_reward_prior(
        obs, actions, (0.0,) * 4,
        context=CONTEXT, dataset=CardPriorDataset.from_dict(raw), human_weight=1,
    )
    assert 1 < result.probabilities[0] / result.probabilities[1] < 1.15


def test_invalid_import_and_nonfinite_logits_fail_closed() -> None:
    raw = {
        "format": CARD_PRIOR_FORMAT,
        "provenance": {
            "source": "fake", "source_url": "https://example.org",
            "collected_at": "2026-10-08", "game_build": "test", "character": "Silent",
            "act": 1, "ascension_min": 0, "ascension_max": 0,
            "methodology": "fixture",
        },
        "cards": [{"card_id": "A", "offered": 1, "picked": 2}],
    }
    with pytest.raises(ValueError):
        CardPriorDataset.from_dict(raw)
    obs, actions = reward()
    with pytest.raises(ValueError):
        blended_card_reward_prior(
            obs, actions, (math.nan, 0, 0, 0),
            context=CONTEXT, dataset=dataset(), human_weight=1,
        )
