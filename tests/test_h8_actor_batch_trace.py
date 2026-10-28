"""Cheap deterministic tests for optional shared-actor batch trace accounting."""
from __future__ import annotations

import pytest

from sts2_ai.models.actor_batch_trace import ActorBatchTrace


def test_batch_trace_occupancy_padding_timing_and_warmup_reset() -> None:
    profiler = ActorBatchTrace()
    assert profiler.report()["batches"] == 0
    profiler.record(
        count=2, padded_objects=5, padded_actions=4,
        valid_objects=8, valid_actions=5,
        fill_ms=3.0, queue_to_batch_mean_ms=4.5,
        queue_to_batch_max_ms=7.0,
        collate_ms=1.0, forward_ms=5.0, reply_ms=0.3,
    )
    profiler.record(
        count=1, padded_objects=3, padded_actions=2,
        valid_objects=3, valid_actions=2,
        fill_ms=0.0, queue_to_batch_mean_ms=1.0,
        queue_to_batch_max_ms=1.0,
        collate_ms=0.7, forward_ms=2.0, reply_ms=0.2,
    )
    report = profiler.report()
    assert report["schema"] == "sts2-h8-shared-actor-batch-trace-v1"
    assert report["batches"] == 2
    assert report["requests"] == 3
    assert report["mean_batch_size"] == 1.5
    assert len(report["batch_records"]) == 2
    pair = report["by_batch_size"]["2"]
    assert pair["batches"] == 1
    assert pair["object_slots"] == 10
    assert pair["action_slots"] == 8
    assert pair["valid_objects"] == 8
    assert pair["valid_actions"] == 5
    assert pair["forward_ms_per_batch"] == 5.0
    assert pair["mean_queue_to_batch_ms"] == 4.5
    profiler.reset()
    assert profiler.report()["requests"] == 0
    assert profiler.report()["batch_records"] == []


@pytest.mark.parametrize("invalid", [
    {"count": 0}, {"padded_objects": 0}, {"padded_actions": 0},
    {"valid_objects": 1}, {"valid_actions": 1},
    {"valid_objects": 100}, {"valid_actions": 100},
    {"fill_ms": -1.0}, {"collate_ms": -0.01},
    {"queue_to_batch_mean_ms": -1.0}, {"reply_ms": -1.0},
])
def test_batch_trace_rejects_invalid_record(invalid: dict) -> None:
    record = dict(
        count=2, padded_objects=4, padded_actions=3,
        valid_objects=5, valid_actions=4,
        fill_ms=2.0, queue_to_batch_mean_ms=3.0,
        queue_to_batch_max_ms=4.0, collate_ms=1.0,
        forward_ms=3.0, reply_ms=0.1,
    )
    record.update(invalid)
    with pytest.raises(ValueError):
        ActorBatchTrace().record(**record)
