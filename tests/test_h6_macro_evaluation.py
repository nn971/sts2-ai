"""H6 confirmation must distinguish completed runs from censored runs."""
import pytest

from tools.evaluate_h6_macro import _paired, _summarize


def _row(seed, *, cleared=False, censored=False, progress=0.3):
    return {
        "seed": seed, "censored": censored, "act1_cleared": (
            None if censored else cleared
        ),
        "frontier_progress": progress,
    }


def test_h6_confirmation_does_not_count_censored_as_losses():
    rows = [
        _row("a", cleared=True, progress=0.9),
        _row("b", censored=True, progress=0.5),
        _row("c", progress=0.3),
    ]
    report = _summarize(rows)
    assert report["runs"] == 3
    assert report["completed"] == 2
    assert report["censored"] == 1
    assert report["act1_clears"] == 1
    assert report["completed_only_win_rate"] == pytest.approx(0.5)
    assert report["win_rate_lower_bound"] == pytest.approx(1 / 3)
    assert report["win_rate_upper_bound"] == pytest.approx(2 / 3)


def test_h6_comparison_pairs_only_jointly_completed_source_seeds():
    base = [_row("a"), _row("b", censored=True), _row("c")]
    new = [_row("a", cleared=True), _row("b"), _row("c")]
    result = _paired(base, new)
    assert result["pairs_jointly_completed"] == 2
    assert result["pairs_with_censoring"] == 1
    assert result["completed_pair_win_delta"] == pytest.approx(0.5)
    assert len(result["completed_pair_win_delta_bootstrap_95pct"]) == 2
    with pytest.raises(ValueError, match="Unmatched"):
        _paired(base, list(reversed(new)))
