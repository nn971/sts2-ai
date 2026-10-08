import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.search import SearchBudget, UctMcts
from sts2_ai.testing import MockLinearBackend
from sts2_ai.training.diagnostics import (
    CutoffSampler,
    load_cutoff_samples,
    observation_shift_report,
    teacher_policy_report,
)
from sts2_ai.training.targets import PolicyTarget, TrainingExample


def _observation(index: int, *, phase: int = 3) -> Observation:
    return Observation(
        policy_id="fair",
        payload_json=json.dumps(
            {"phase": phase, "act": 1, "floor": 2, "hp": 35, "max_hp": 70}
        ),
        observation_hash=f"hash-{index}",
    )


def _example(index: int, a: float, b: float, *, phase: int = 2) -> TrainingExample:
    observation = _observation(index, phase=phase)
    return TrainingExample(
        observation_hash=observation.observation_hash,
        information_policy="fair",
        policy_targets=(PolicyTarget("a", a), PolicyTarget("b", b)),
        value_target=0.0,
        source_search_id=f"search-{index}",
        emulator_revision="emu",
        observation_json=observation.payload_json,
        search_budget=32,
    )


def test_teacher_entropy_distinguishes_flat_from_clear_targets() -> None:
    report = teacher_policy_report((_example(0, 0.5, 0.5), _example(1, 1.0, 0.0)))
    assert report["examples"] == 2
    assert report["mean_entropy_nats"] == pytest.approx(math.log(2) / 2)
    assert report["mean_normalized_entropy"] == pytest.approx(0.5)
    assert report["mean_effective_actions"] == pytest.approx(1.5)
    assert report["fraction_multi_action_confident_ge_0_7"] == 0.5
    assert report["fraction_multi_action_near_uniform_ge_0_95"] == 0.5


def test_cutoff_sampler_is_deterministic_deduplicated_and_bounded(
    tmp_path: Path,
) -> None:
    sampler = CutoffSampler(every=2, max_unique=1)
    first = _observation(1)
    second = _observation(2)
    for item in (first, first, second, first, second, first):
        sampler.record(item, run_seed="seed-0")
    assert sampler.seen == 6
    assert sampler.sampled == 3
    output = tmp_path / "nested" / "cutoffs.jsonl"
    assert sampler.write_jsonl(output, provenance={"budget": 8}) == 1
    samples = load_cutoff_samples(output)
    assert len(samples) == 1
    assert samples[0].observation_hash == "hash-1"
    assert samples[0].sampled_occurrences == 3
    raw = json.loads(output.read_text().splitlines()[0])
    assert raw["provenance"]["budget"] == 8

    with pytest.raises(ValueError, match="positive"):
        CutoffSampler(every=0)


def test_shift_report_exposes_different_phase_and_overlap() -> None:
    examples = (_example(1, 1.0, 0.0, phase=1),)
    sampler = CutoffSampler(every=1)
    sampler.record(_observation(1, phase=1), run_seed="seed-0")
    sampler.record(_observation(2, phase=3), run_seed="seed-0")

    # Use the independent JSONL record reader rather than sampler internals.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        output = Path(directory) / "cutoffs.jsonl"
        sampler.write_jsonl(output, provenance={})
        samples = load_cutoff_samples(output)
    report = observation_shift_report(examples, samples)
    assert report["phase_total_variation_distance"] == pytest.approx(0.5)
    assert report["cutoff_unique_hash_overlap_with_roots"] == pytest.approx(0.5)
    assert report["cutoffs"]["combat_fraction"] == 0.0
    with pytest.raises(ValueError, match="policies"):
        observation_shift_report(
            examples, (replace(samples[0], information_policy="other"),)
        )


def test_mcts_cutoff_observer_is_passive() -> None:
    backend = MockLinearBackend(terminal_at=100)
    observed: list[Observation] = []
    options = dict(
        policy=InformationPolicy("fair-test"),
        rollout_policy=HeuristicAgent(),
        rollout_depth=0,
        rollout_batch_size=4,
        seed=19,
    )
    baseline = UctMcts(backend, **options)
    instrumented = UctMcts(backend, cutoff_observer=observed.append, **options)
    original = baseline.search("0", SearchBudget(max_simulations=12))
    captured = instrumented.search("0", SearchBudget(max_simulations=12))
    assert captured.evaluations == original.evaluations
    assert captured.search_version == original.search_version
    assert captured.cutoff_rollouts == len(observed)
    assert captured.cutoff_rollouts > 0


def test_mcts_terminal_rollouts_do_not_enter_cutoff_capture() -> None:
    backend = MockLinearBackend(terminal_at=1)
    observed: list[Observation] = []
    search = UctMcts(
        backend,
        policy=InformationPolicy("fair-test"),
        rollout_policy=HeuristicAgent(),
        rollout_depth=0,
        cutoff_observer=observed.append,
        seed=7,
    )
    result = search.search("0", SearchBudget(max_simulations=4))
    assert result.terminal_rollouts + result.cutoff_rollouts >= 0
    assert observed == []
