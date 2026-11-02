"""Full-combat H16 PUCT budget comparisons and uncapped search safety."""
from __future__ import annotations

import json
import random

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.search.combat_puct import (
    H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
    CombatFrame,
    CombatPuct,
    IncompleteCombatRollout,
)
from tools.diagnose_h16_full_combat import (
    SCHEMA,
    build_analysis,
    read_arm_journal,
    verify_depth8_budget,
)


class FiniteCombat:
    """Two root moves, then a deterministic eleven-action combat."""
    fair_combat_capability_id = H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY

    def __init__(self, length: int = 12) -> None:
        self.length = length
        self.starts = 0
        self.ends = 0

    def frame(self, n: int, suffix: str = "") -> CombatFrame:
        return CombatFrame(
            information_key=f"toy-{n}-{suffix}",
            observation=Observation(FAIR_POLICY_ID, "{}", f"hash-{n}"),
            legal_actions=(
                () if n == self.length
                else (LegalAction("a", "end_turn"), LegalAction("b", "end_turn"))
                if n == 0 else (LegalAction("advance", "end_turn"),)
            ),
            terminal_reward=2.0 if n == self.length else None,
        )

    def begin_simulation(self, root: CombatFrame, rng: random.Random) -> None:
        self.starts += 1

    def step(self, frame: CombatFrame, action: LegalAction,
             rng: random.Random) -> CombatFrame:
        n = int(frame.information_key.split("-")[1])
        suffix = action.action_id if n == 0 else frame.information_key.split("-", 2)[2]
        return self.frame(n + 1, suffix)

    def end_simulation(self) -> None:
        self.ends += 1


def evaluator(obs, actions):
    return (tuple(0.0 for _ in actions), 0.1)


def test_full_combat_search_reaches_actual_terminal_after_depth_eight():
    sampler = FiniteCombat(length=12)
    full = CombatPuct(
        sampler, evaluator, max_depth=None,
        tree_expansion_mode="full_depth",
        max_simulation_steps=32,
    ).search(sampler.frame(0), simulations=16)
    assert full.terminal_evaluations == 16
    assert full.neural_cutoffs == 0
    assert full.transitions == 16 * 12
    assert full.rollout_mode == "full_combat"
    assert full.maximum_expanded_depth == 11
    assert sum(full.visits) == 16
    assert full.root_mean_value == 1.0
    assert sampler.starts == sampler.ends == 16

    capped = CombatPuct(
        FiniteCombat(length=12), evaluator, max_depth=8,
        tree_expansion_mode="full_depth",
    ).search(sampler.frame(0), simulations=16)
    assert capped.neural_cutoffs == 16
    assert capped.terminal_evaluations == 0
    assert capped.transitions == 16 * 8


def test_full_combat_watchdog_raises_without_reward_substitution():
    sampler = FiniteCombat(length=12)
    with pytest.raises(IncompleteCombatRollout, match="without reaching"):
        CombatPuct(
            sampler, evaluator, max_depth=None,
            tree_expansion_mode="full_depth",
            max_simulation_steps=5,
        ).search(sampler.frame(0), simulations=2)
    assert sampler.starts == sampler.ends == 1
    with pytest.raises(ValueError, match="Invalid combat"):
        CombatPuct(sampler, evaluator, max_depth=None)
    with pytest.raises(ValueError, match="Invalid combat"):
        CombatPuct(sampler, evaluator, max_depth=None,
                   tree_expansion_mode="full_depth",
                   max_simulation_steps=0)


def result(seed: str, floor: int, won: bool, searches: int = 1):
    return {
        "seed": seed, "outcome": "defeat", "max_floor_reached": floor,
        "death_floor": floor, "floor7_reached": True, "floor7_won": won,
        "post_floor7_hp": 0.3 if won else None,
        "lost_fights": [{"floor": floor, "damage": 0.5}],
        "searches": searches, "search_transitions": 100,
        "search_overrides": 0, "decisions": 12,
    }


def test_full_combat_report_pairs_budgets_and_depths():
    ref = {
        "policy": [result("a", 11, True), result("b", 7, False)],
        "puct": [result("a", 7, False), result("b", 10, True)],
    }
    depth8_64 = [result("a", 7, False), result("b", 10, True)]
    full16 = [result("a", 10, True), result("b", 10, True)]
    full64 = [result("a", 12, True), result("b", 7, False)]
    report = build_analysis(ref, depth8_64, full16, full64)
    assert set(report["summary"]) == {
        "policy", "depth8_puct16", "depth8_puct64",
        "full_puct16", "full_puct64",
    }
    assert report["paired"]["full16_vs_depth8_16"]["floor7_only_right"] == 1
    assert report["paired"]["full64_vs_full16"]["floor7_only_left"] == 1
    assert report["paired"]["full64_vs_full16"][
        "mean_floor_delta_right_minus_left"
    ] == -0.5
    with pytest.raises(ValueError, match="identical seed order"):
        build_analysis(ref, depth8_64, full16, list(reversed(full64)))


def test_prior_depth8_budget_must_match_original_input():
    identity = {
        "horizon_lambda": 0.4, "emulator_revision": "emu",
        "macro_sha256": "macro", "checkpoints": {"79": {"sha256": "ckpt"}},
    }
    report = {
        "schema": "sts2-h16-puct-budget-ablation-v1",
        "identity": {
            "reference_sha256": "ref", "round": 79, "simulations": 64,
            "baseline_simulations": 16, "horizon_lambda": 0.4,
            "emulator_revision": "emu", "macro_sha256": "macro",
            "checkpoint_sha256": "ckpt",
        },
        "puct64_runs": [{"seed": "a", "result": result("a", 7, False)}],
    }
    assert len(verify_depth8_budget(
        report, reference_sha256="ref", reference_identity=identity,
        reference_seeds=["a"], round_number=79,
    )) == 1
    with pytest.raises(ValueError, match="different experiment"):
        verify_depth8_budget(
            report, reference_sha256="changed", reference_identity=identity,
            reference_seeds=["a"], round_number=79,
        )


def test_full_combat_journal_validates_resume(tmp_path):
    path = tmp_path / "test.partial.jsonl"
    identity = {"simulations": 16, "max_simulation_steps": 512}
    row = {"seed": "a", "result": result("a", 7, False)}
    path.write_text(
        json.dumps({"schema": SCHEMA, "identity": identity}) + "\n"
        + json.dumps(row) + "\n", encoding="utf-8",
    )
    assert read_arm_journal(path, identity, ["a", "b"]) == [row]
    with pytest.raises(ValueError, match="identity mismatch"):
        read_arm_journal(path, {**identity, "simulations": 64}, ["a", "b"])
    with pytest.raises(ValueError, match="reordered"):
        read_arm_journal(path, identity, ["b", "a"])
