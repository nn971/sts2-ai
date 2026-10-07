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
    average_frontier_progress: float
    average_defeat_enemy_hp: float | None
    average_decisions: float
    transitions_per_decision: float
    average_wall_seconds: float
    average_agent_compute_seconds: float


def summarize_runs(label: str, summaries: list[RunSummary]) -> BenchmarkRow:
    if not summaries:
        raise ValueError("benchmark row requires at least one run")

    total_decisions = sum(summary.decisions for summary in summaries)
    total_transitions = sum(summary.emulator_transitions for summary in summaries)
    defeat_enemy_hp = [
        summary.frontier_enemy_hp
        for summary in summaries
        if summary.outcome == "defeat" and summary.frontier_enemy_hp is not None
    ]
    return BenchmarkRow(
        label=label,
        runs=len(summaries),
        win_rate=sum(summary.won for summary in summaries) / len(summaries),
        average_terminal_progress=statistics.fmean(
            summary.terminal_progress for summary in summaries
        ),
        average_frontier_progress=statistics.fmean(
            summary.frontier_progress for summary in summaries
        ),
        average_defeat_enemy_hp=(
            statistics.fmean(defeat_enemy_hp) if defeat_enemy_hp else None
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
        "| Agent | Runs | Win % | Avg terminal progress | Avg frontier progress | "
        "Avg defeat enemy HP | Emulator transitions/decision | Time/run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        defeat_hp = (
            f"{row.average_defeat_enemy_hp:.1f}"
            if row.average_defeat_enemy_hp is not None
            else "—"
        )
        lines.append(
            f"| {row.label} | {row.runs} | {100.0 * row.win_rate:.1f} | "
            f"{row.average_terminal_progress:.2f} | "
            f"{row.average_frontier_progress:.3f} | {defeat_hp} | "
            f"{row.transitions_per_decision:.1f} | {row.average_wall_seconds:.3f}s |"
        )
    return "\n".join(lines)
