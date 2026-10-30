"""The M15 PUCT-budget probe measures search contribution without re-training."""
from __future__ import annotations

import pytest

from tools.probe_h15_reference_search_budget import (
    summarize_search_q,
    validate_budgets,
)


def test_budget_parser_requires_positive_sorted_unique_budgets():
    assert validate_budgets("16,64,128") == (16, 64, 128)
    for invalid in ("", "16,16", "64,16", "0,16", "-1,16", "16,abc"):
        with pytest.raises(ValueError):
            validate_budgets(invalid)


def test_search_q_probe_reports_real_cutoffs_and_action_separation():
    sample = {
        "searched_root_diagnostics": [{
            "root_visits": [3, 1, 0],
            "root_mean_action_values": [0.71, 0.43, None],
            "search_simulations": 4,
            "search_neural_cutoffs": 4,
            "search_terminal_evaluations": 0,
            "search_expanded_information_sets": 3,
            "search_transitions": 6,
        }],
    }
    stats = summarize_search_q([sample])
    assert stats["mean_visited_action_q_spread"] == pytest.approx(0.28)
    assert stats["mean_expanded_public_histories"] == 3
    assert stats["total_neural_cutoffs"] == 4
    assert stats["total_terminal_evaluations"] == 0


def test_search_q_probe_rejects_stale_or_incomplete_telemetry():
    sample = {
        "searched_root_diagnostics": [{
            "root_visits": [3, 1],
            "root_mean_action_values": [0.7, None],
            "search_simulations": 4,
        }],
    }
    with pytest.raises(ValueError, match="Invalid selected"):
        summarize_search_q([sample])
