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


@dataclass(frozen=True, slots=True)
class PairedComparison:
    baseline: str
    challenger: str
    runs: int
    mean_frontier_delta: float
    mean_terminal_delta: float
    challenger_ahead: int
    tied: int
    baseline_ahead: int


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


def compare_paired_runs(
    baseline: str,
    baseline_runs: list[RunSummary],
    challenger: str,
    challenger_runs: list[RunSummary],
) -> PairedComparison:
    baseline_by_seed = {run.seed: run for run in baseline_runs}
    challenger_by_seed = {run.seed: run for run in challenger_runs}
    if len(baseline_by_seed) != len(baseline_runs):
        raise ValueError("baseline contains duplicate seeds")
    if len(challenger_by_seed) != len(challenger_runs):
        raise ValueError("challenger contains duplicate seeds")
    if baseline_by_seed.keys() != challenger_by_seed.keys():
        raise ValueError("paired comparison requires identical seed sets")
    if not baseline_by_seed:
        raise ValueError("paired comparison requires at least one run")

    frontier_deltas = []
    terminal_deltas = []
    challenger_ahead = 0
    tied = 0
    baseline_ahead = 0
    for seed in sorted(baseline_by_seed):
        base = baseline_by_seed[seed]
        other = challenger_by_seed[seed]
        delta = other.frontier_progress - base.frontier_progress
        frontier_deltas.append(delta)
        terminal_deltas.append(other.terminal_progress - base.terminal_progress)
        if delta > 1e-12:
            challenger_ahead += 1
        elif delta < -1e-12:
            baseline_ahead += 1
        else:
            tied += 1

    return PairedComparison(
        baseline=baseline,
        challenger=challenger,
        runs=len(frontier_deltas),
        mean_frontier_delta=statistics.fmean(frontier_deltas),
        mean_terminal_delta=statistics.fmean(terminal_deltas),
        challenger_ahead=challenger_ahead,
        tied=tied,
        baseline_ahead=baseline_ahead,
    )


def paired_markdown_table(rows: list[PairedComparison]) -> str:
    lines = [
        "| Challenger | Baseline | Runs | Mean frontier Δ | Mean terminal Δ | "
        "Ahead / tied / behind |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.challenger} | {row.baseline} | {row.runs} | "
            f"{row.mean_frontier_delta:+.3f} | {row.mean_terminal_delta:+.2f} | "
            f"{row.challenger_ahead} / {row.tied} / {row.baseline_ahead} |"
        )
    return "\n".join(lines)
