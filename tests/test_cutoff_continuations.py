import hashlib
import json
from pathlib import Path

import pytest

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.models import HashedLinearPolicyValueModel
from sts2_ai.search import SearchBudget, UctMcts
from sts2_ai.testing import MockLinearBackend
from sts2_ai.training.continuations import (
    CutoffContinuationCollector,
    continuation_report,
    load_continuations,
)


class OutcomeBackend(MockLinearBackend):
    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        raw: dict[str, object] = {
            "phase": 3,
            "act": 1,
            "floor": 1,
            "hp": 30,
            "max_hp": 70,
            "value": int(state),
        }
        if self.is_terminal(state):
            raw["terminal_outcome"] = "victory"
        payload = json.dumps(raw, sort_keys=True)
        digest = hashlib.sha256((policy.policy_id + payload).encode()).hexdigest()
        return Observation(policy.policy_id, payload, digest)


class DefeatBackend(OutcomeBackend):
    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        observation = super().observe(state, policy)
        raw = json.loads(observation.payload_json)
        if self.is_terminal(state):
            raw["terminal_outcome"] = "defeat"
        payload = json.dumps(raw, sort_keys=True)
        digest = hashlib.sha256((policy.policy_id + payload).encode()).hexdigest()
        return Observation(policy.policy_id, payload, digest)


def _collector(
    backend: MockLinearBackend, *, max_decisions: int = 8, every: int = 1
) -> CutoffContinuationCollector:
    return CutoffContinuationCollector(
        backend, policy=InformationPolicy("fair-test"),
        continuation_policy=HeuristicAgent(),
        max_decisions=max_decisions,
        every=every,
        max_unique=4,
    )


def test_independent_terminal_label_and_deduplication(tmp_path: Path) -> None:
    backend = OutcomeBackend(terminal_at=3)
    collector = _collector(backend)
    observation = backend.observe("0", InformationPolicy("fair-test"))
    collector.record(observation, "0", "depth", run_seed="seed-a")
    collector.record(observation, "0", "depth", run_seed="seed-a")
    assert collector.seen == 2
    assert collector.sampled == 2
    assert len(collector.records) == 1
    (record,) = collector.records
    assert record.terminal_value == 1.0
    assert record.outcome == "victory"
    assert 0 < record.continuation_decisions <= 3
    assert record.source_exact_hash == backend.exact_hash("0")
    assert record.continuation_policy_id == "heuristic-v3-claim-visible-rewards"

    output = tmp_path / "nested" / "labels.jsonl"
    assert collector.write_jsonl(output, provenance={"emulator_revision": "mock"}) == 1
    assert load_continuations(output) == (record,)
    report = continuation_report((record,))
    assert report["terminal_labels"] == 1
    assert report["victories"] == 1
    assert report["handcrafted_terminal_rmse"] is not None
    model = HashedLinearPolicyValueModel.zeros(dimension=64)
    learned_report = continuation_report((record,), model=model)
    assert learned_report["learned_terminal_rmse"] == pytest.approx(1.0)


def test_censoring_never_turns_into_failure_label(tmp_path: Path) -> None:
    backend = DefeatBackend(terminal_at=100)
    collector = _collector(backend, max_decisions=1, every=2)
    observation = backend.observe("0", InformationPolicy("fair-test"))
    collector.record(observation, "0", "depth", run_seed="seed-a")
    assert collector.records == ()
    collector.record(observation, "0", "depth", run_seed="seed-a")
    (record,) = collector.records
    assert record.outcome == "truncated"
    assert record.terminal_value is None
    assert record.continuation_decisions == 1
    report = continuation_report((record,))
    assert report["truncated"] == 1
    assert report["terminal_labels"] == 0
    assert report["handcrafted_terminal_rmse"] is None
    output = tmp_path / "labels.jsonl"
    collector.write_jsonl(output, provenance={})
    assert load_continuations(output)[0].terminal_value is None


def test_defeat_label_is_exact_terminal_outcome() -> None:
    backend = DefeatBackend(terminal_at=2)
    collector = _collector(backend)
    observation = backend.observe("0", InformationPolicy("fair-test"))
    collector.record(observation, "0", "depth", run_seed="seed-0")
    assert collector.records[0].outcome == "defeat"
    assert collector.records[0].terminal_value == -1.0


def test_rejects_illegal_capture_context() -> None:
    backend = OutcomeBackend(terminal_at=4)
    collector = _collector(backend)
    observation = backend.observe("0", InformationPolicy("fair-test"))
    with pytest.raises(ValueError, match="reason"):
        collector.record(observation, "0", "terminal", run_seed="seed-a")
    with pytest.raises(ValueError, match="policy"):
        collector.record(
            Observation("other", observation.payload_json, observation.observation_hash),
            "0", "depth", run_seed="seed-a"
        )


def test_mcts_continuation_hook_is_passive() -> None:
    backend = OutcomeBackend(terminal_at=100)
    options = {
        "policy": InformationPolicy("fair-test"),
        "rollout_policy": HeuristicAgent(),
        "rollout_depth": 0,
        "rollout_batch_size": 4,
        "seed": 19,
    }
    baseline = UctMcts(backend, **options)
    collector = _collector(backend, max_decisions=1)
    instrumented = UctMcts(
        backend,
        continuation_observer=lambda obs, handle, reason: collector.record(
            obs, handle, reason, run_seed="seed-a"
        ),
        **options,
    )
    first = baseline.search("0", SearchBudget(max_simulations=12))
    second = instrumented.search("0", SearchBudget(max_simulations=12))
    assert first.evaluations == second.evaluations
    assert first.search_version == second.search_version
    assert second.cutoff_rollouts == collector.seen
    assert collector.records
    assert all(record.terminal_value is None for record in collector.records)


def test_capture_skips_terminal_uct_leaves() -> None:
    backend = OutcomeBackend(terminal_at=1)
    collector = _collector(backend)
    search = UctMcts(
        backend,
        policy=InformationPolicy("fair-test"),
        rollout_policy=HeuristicAgent(),
        rollout_depth=0,
        continuation_observer=lambda obs, handle, reason: collector.record(
            obs, handle, reason, run_seed="test"
        ),
        seed=7,
    )
    search.search("0", SearchBudget(max_simulations=4))
    assert collector.seen == 0
