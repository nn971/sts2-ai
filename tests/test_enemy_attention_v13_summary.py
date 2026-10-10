"""Do not confuse independent training repeats with repeated evaluation seeds."""
from __future__ import annotations

import pytest

from sts2_ai.evaluation.checkpoint_selection import GOAL, REPORT_SCHEMA
from tools.summarize_enemy_attention_v13 import markdown, summarize


def _report() -> dict:
    models = {f"{arch}-r{r}" for r in (1, 2, 3)
              for arch in ("h32", "h64", "attn")}
    records = []
    for i in range(4):
        seed = f"v13-fresh-{i}"
        by_model = {}
        for arch in models:
            # Attention beats width32 on exactly one additional common seed,
            # and wider MLP wins a different seed for two training replicas.
            won = (
                i == 0 or
                arch.startswith("attn") and i == 1 or
                arch.startswith("h64") and i == 2
            )
            by_model[arch] = {
                "seed": seed, "outcome": "victory" if won else "defeat",
                "episode_goal_version": GOAL,
                "act1_cleared": won, "censored": False,
                "boss_progress": {
                    "encounter_id": "kin", "initial_hp": 100,
                    "remaining_hp": 0 if won else 20,
                },
            }
        records.append({"index": i, "seed": seed, "models": by_model})
    return {
        "schema": REPORT_SCHEMA,
        "mode": "comparison",
        "identity": {
            "goal": GOAL, "seed_prefix": "v13-fresh", "seeds": 4,
            "emulator_revision": "test",
            "model_specs": {name: {} for name in models},
        },
        "runs": records,
    }


def test_v13_summary_reports_each_replica_and_paired_deltas() -> None:
    stats = summarize(_report())
    assert len(stats["replicas"]) == 3
    assert stats["mean_clear_rates"]["attn"] == pytest.approx(0.5)
    assert stats["mean_clear_rates"]["h32"] == pytest.approx(0.25)
    assert stats["mean_attention_advantages"]["h32"] == pytest.approx(0.25)
    assert stats["positive_pair_counts"]["h32"] == 3
    assert "25.0%" in markdown(stats)


def test_v13_summary_fails_closed_on_missing_models_and_mispaired_seeds() -> None:
    report = _report()
    del report["runs"][0]["models"]["h32-r1"]
    with pytest.raises(ValueError, match="Incomplete model"):
        summarize(report)
    report = _report()
    report["runs"][0]["models"]["h32-r1"]["seed"] = "other-seed"
    with pytest.raises(ValueError, match="Inconsistent"):
        summarize(report)
    report = _report()
    report["runs"][0]["models"]["attn-r3"]["censored"] = True
    with pytest.raises(ValueError):
        summarize(report)
