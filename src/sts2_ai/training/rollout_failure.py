"""Portable, public-only diagnostics for failed emulator self-play steps.

Failure records are ordinary JSON-compatible data so they survive a spawned
process boundary. They contain public frames and the *chosen legal action IDs*
for deterministic replay, never exact state handles or hidden RNG states.
"""
from __future__ import annotations

from typing import Any

SCHEMA = "sts2-public-rollout-step-failure-v1"


class PublicRolloutFailure(RuntimeError):
    """An emulator transition failed; this episode has no reward target.

    Pass the single JSON-compatible record to Exception.__init__ so instances
    stay pickleable when ProcessPoolExecutor sends them to the learner.
    """

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(report)

    def __str__(self) -> str:
        record = self.report
        return (
            f"Emulator step failed (seed={record['seed']!r}, "
            f"environment={record['environment']!r}, "
            f"decision={record['decision_index']}, "
            f"action={record['failing_action']['action_id']!r}): "
            f"{record['error_type']}: {record['error_message']}"
        )
