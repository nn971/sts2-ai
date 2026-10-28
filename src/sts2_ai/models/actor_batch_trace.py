"""Compact, optional batch-level telemetry for the shared H7/H8 actor.

No Torch dependencies. The trace records executed batches only. An H8
synchronized warmup barrier resets it alongside actor performance counters.

fill_ms starts when the first request of a batch is dequeued. It measures
batch assembly wait, NOT total request queue latency.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ActorBatchTrace:
    entries: list[dict[str, float | int]] = field(default_factory=list)

    def reset(self) -> None:
        self.entries.clear()

    def record(
        self, *, count: int, padded_objects: int, padded_actions: int,
        valid_objects: int, valid_actions: int, fill_ms: float,
        collate_ms: float, forward_ms: float, reply_ms: float,
    ) -> None:
        if (count <= 0 or padded_objects <= 0 or padded_actions <= 0 or
                valid_objects < count or valid_actions < count or
                valid_objects > count * padded_objects or
                valid_actions > count * padded_actions or
                min(fill_ms, collate_ms, forward_ms, reply_ms) < 0):
            raise ValueError("Invalid shared actor batch telemetry")
        self.entries.append({
            "count": count,
            "padded_objects": padded_objects,
            "padded_actions": padded_actions,
            "valid_objects": valid_objects,
            "valid_actions": valid_actions,
            "fill_ms": fill_ms,
            "collate_ms": collate_ms,
            "forward_ms": forward_ms,
            "reply_ms": reply_ms,
        })

    def report(self) -> dict:
        """Both per-batch records and grouped occupancy statistics."""
        bins: dict[int, dict[str, float | int]] = defaultdict(lambda: {
            "batches": 0, "object_slots": 0, "action_slots": 0,
            "valid_objects": 0, "valid_actions": 0,
            "fill_ms": 0.0, "collate_ms": 0.0,
            "forward_ms": 0.0, "reply_ms": 0.0,
        })
        for e in self.entries:
            size = int(e["count"])
            b = bins[size]
            b["batches"] += 1
            b["object_slots"] += size * int(e["padded_objects"])
            b["action_slots"] += size * int(e["padded_actions"])
            for name in ("valid_objects", "valid_actions", "fill_ms",
                         "collate_ms", "forward_ms", "reply_ms"):
                b[name] += e[name]
        total = sum(int(b["batches"]) for b in bins.values())
        requests = sum(size * int(b["batches"]) for size, b in bins.items())
        return {
            "schema": "sts2-h8-shared-actor-batch-trace-v1",
            "batches": total,
            "requests": requests,
            "mean_batch_size": requests / total if total else 0.0,
            "by_batch_size": {
                str(size): {
                    **dict(b),
                    "mean_padded_objects": (
                        b["object_slots"] / (size * b["batches"])),
                    "mean_padded_actions": (
                        b["action_slots"] / (size * b["batches"])),
                    "forward_ms_per_batch": b["forward_ms"] / b["batches"],
                    "collate_ms_per_batch": b["collate_ms"] / b["batches"],
                    "fill_ms_per_batch": b["fill_ms"] / b["batches"],
                }
                for size, b in sorted(bins.items())
            },
            "batch_records": list(self.entries),
            "interpretation": (
                "fill_ms starts after first request dequeue; it measures "
                "batch assembly delay, NOT upstream request queue wait. "
                "Object/action slots include pad; forward_ms includes "
                "GPU synchronize and output D2H, reply_ms includes per-slot "
                "queue serialization. Trace resets after synchronized warmup."),
        }
