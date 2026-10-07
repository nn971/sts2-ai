from __future__ import annotations

import statistics
from dataclasses import dataclass

from .run import RunSummary


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    label: str
    runs: int
    win_rate: float
    average_terminal_progress: float
    average_decisions: float
    transitions_per_decision: float
    average_wall_seconds: float
    average_agent_compute_seconds: float


def summarize_runs(label: str, summaries: list[RunSummary]) -> BenchmarkRow:
    if not summaries:
        raise ValueError("benchmark row requires at least one run")

    total_decisions = sum(summary.decisions for summary in summaries)
    total_transitions = sum(summary.emulator_transitions for summary in summaries)
    return BenchmarkRow(
        label=label,
        runs=len(summaries),
        win_rate=sum(summary.won for summary in summaries) / len(summaries),
        average_terminal_progress=statistics.fmean(
            summary.terminal_progress for summary in summaries
        ),
        average_decisions=statistics.fmean(summary.decisions for summary in summaries),
        transitions_per_decision=(
            total_transitions / total_decisions if total_decisions else 0.0
        ),
        average_wall_seconds=statistics.fmean(
            summary.wall_seconds for summary in summaries
        ),
        average_agent_compute_seconds=statistics.fmean(
            summary.agent_compute_seconds for summary in summaries
        ),
    )


def markdown_table(rows: list[BenchmarkRow]) -> str:
    lines = [
        "| Agent | Runs | Win % | Avg terminal progress | "
        "Emulator transitions/decision | Time/run |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row.label} | {row.runs} | {100.0 * row.win_rate:.1f} | "
        f"{row.average_terminal_progress:.2f} | "
        f"{row.transitions_per_decision:.1f} | {row.average_wall_seconds:.3f}s |"
        for row in rows
    )
    return "\n".join(lines)
