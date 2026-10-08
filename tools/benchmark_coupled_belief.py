#!/usr/bin/env python3
"""Reproducible, seed-blind PUBLIC-HISTORY survival/performance benchmark.

Build an Overgrowth Silent transcript from a *declared synthetic* six-stream
seed (fixture only); conditioning components receive solely public frames,
legal menus and chosen actions. Compare reusable finite empirical particles
with bounded fresh *joint full-history rejection*. The latter is a different
posterior estimator, not an equal-quality apples-to-apples speed contest.

This program does not read hidden game RNG during conditioning. It does
not claim native STS2 posterior calibration.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    FactorizedRunStartPosteriorSampler,
    IncrementalCoupledParticlePosterior,
    IncrementalCoupledPosteriorExhausted,
    InformationPolicy,
    JsonlEmulatorBackend,
    LegalAction,
)
from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.coupled_factorized_rejection import (
    CoupledFactorizedHistoryRejectionSampler,
)
from sts2_ai.emulator.rejection import HistoryConditioningExhausted

BENCHMARK_SCHEMA = "synthetic-overgrowth-coupled-belief-benchmark-v1"
ORACLE_INITIAL_STREAMS = {
    "map": 11,
    "combat": 29,
    "combat_targets": 5,
    "reward": 7,
    "shop": 9,
    "event": 13,
}
MAP_SUPPORT = (11, 12, 13)
COMBAT_SUPPORT = (29, 30, 31, 32)


def _visible_action(actions: tuple[LegalAction, ...], *, map_entry: bool) -> LegalAction:
    """Deterministic policy using only the complete, publicly legal menu."""
    if not actions:
        raise ValueError("A terminal public frame cannot select an action")
    preferred = (
        ("choose_map_node",) if map_entry else
        ("play_card", "end_turn", "skip_reward", "leave_shop", "leave_event", "choose_map_node")
    )
    for kind in preferred:
        chosen = next((action for action in actions if action.kind == kind), None)
        if chosen is not None:
            return chosen
    return actions[0]


def _public_history(
    backend: JsonlEmulatorBackend, *,
    post_map_decisions: int,
) -> tuple[PublicHistoryStep, ...]:
    """Generate fixture frames, then permanently release ALL fixture handles."""
    policy = InformationPolicy(FAIR_POLICY_ID)
    state = backend.reset_factorized_hypothetical(ORACLE_INITIAL_STREAMS)
    owned = [state]
    history: list[PublicHistoryStep] = []
    try:
        for index in range(post_map_decisions + 2):
            actions = tuple(backend.legal_actions(state))
            observation = backend.observe(state, policy)
            # First action is start_run; second enters a visible map node.
            can_choose = bool(actions) and index < post_map_decisions + 1
            selected = _visible_action(actions, map_entry=index == 1) if can_choose else None
            history.append(PublicHistoryStep(observation, selected, actions))
            if selected is None:
                break
            child = backend.step(state, selected).child
            owned.append(child)
            state = child
        if len(history) < 3:
            raise RuntimeError("Fixture did not reach at least one Overgrowth room entry")
        if history[0].chosen_action is None or history[0].chosen_action.kind != "start_run":
            raise RuntimeError("Synthetic fixture did not start a run")
        if history[1].chosen_action is None or history[1].chosen_action.kind != "choose_map_node":
            raise RuntimeError("Synthetic fixture did not choose a map node")
        return tuple(history)
    finally:
        backend.release_many(owned)


def _operation_snapshot(backend: JsonlEmulatorBackend) -> dict[str, int]:
    return {name: item.calls for name, item in backend.operation_profile().items()}


def _operation_delta(start: dict[str, int], end: dict[str, int]) -> dict[str, int]:
    return {
        key: end.get(key, 0) - start.get(key, 0)
        for key in sorted(set(start) | set(end))
        if end.get(key, 0) != start.get(key, 0)
    }


def _phase(step: PublicHistoryStep) -> int | str | None:
    parsed: Any = json.loads(step.observation.payload_json)
    if not isinstance(parsed, dict):
        raise ValueError("Expected public observation JSON object")
    value = parsed.get("phase")
    return value if isinstance(value, (int, str)) else None


def _benchmark_cohort(
    backend: JsonlEmulatorBackend,
    history: tuple[PublicHistoryStep, ...],
    *,
    cohort_size: int,
    seed: int,
) -> dict[str, Any]:
    before = _operation_snapshot(backend)
    source = FactorizedRunStartPosteriorSampler(
        backend, map_states=MAP_SUPPORT, combat_states=COMBAT_SUPPORT
    )
    checkpoints: list[dict[str, Any]] = []
    with source, IncrementalCoupledParticlePosterior(
        backend, runstart_sampler=source
    ) as belief:
        initialized_at = time.perf_counter()
        belief.initialize(
            (
                history[0],
                PublicHistoryStep(
                    history[1].observation, None, history[1].legal_actions
                ),
            ),
            search_rng=random.Random(seed),
            cohort_size=cohort_size,
        )
        initialization_seconds = time.perf_counter() - initialized_at
        checkpoints.append({
            "public_frame": 1,
            "phase": _phase(history[1]),
            "chosen_action_kind": None,
            "survivors": belief.stats.surviving_particles,
            "simulator_steps": 0,
            "wall_seconds": 0.0,
            "status": "conditioned",
        })
        collapsed = False
        for index in range(2, len(history)):
            action = history[index - 1].chosen_action
            assert action is not None
            previous_steps = belief.stats.simulator_transitions
            started_at = time.perf_counter()
            status = "conditioned"
            if not collapsed:
                try:
                    assert history[index].legal_actions is not None
                    belief.advance(
                        action,
                        history[index].observation,
                        history[index].legal_actions,
                    )
                except IncrementalCoupledPosteriorExhausted:
                    collapsed = True
                    status = "cohort_collapsed"
            else:
                status = "unavailable_after_collapse"
            checkpoints.append({
                "public_frame": index,
                "phase": _phase(history[index]),
                "chosen_action_kind": action.kind,
                "survivors": belief.stats.surviving_particles,
                "simulator_steps": (
                    belief.stats.simulator_transitions - previous_steps
                ),
                "wall_seconds": time.perf_counter() - started_at if not (
                    status == "unavailable_after_collapse"
                ) else 0.0,
                "status": status,
            })
        final_stats = belief.stats
    return {
        "cohort_size": cohort_size,
        "initialization_seconds": initialization_seconds,
        "factorized_runstarts": source.stats.inspected_run_starts,
        "observed_transitions": final_stats.observed_transitions,
        "total_simulator_steps": final_stats.simulator_transitions,
        "surviving_particles": final_stats.surviving_particles,
        "collapsed": collapsed,
        "checkpoints": checkpoints,
        "bridge_operations": _operation_delta(before, _operation_snapshot(backend)),
    }


def _benchmark_rejection(
    backend: JsonlEmulatorBackend,
    history: tuple[PublicHistoryStep, ...],
    *,
    proposal_budget: int,
    probes: int,
    seed: int,
) -> list[dict[str, Any]]:
    source = FactorizedRunStartPosteriorSampler(
        backend, map_states=MAP_SUPPORT, combat_states=COMBAT_SUPPORT
    )
    records: list[dict[str, Any]] = []
    with source:
        sampler = CoupledFactorizedHistoryRejectionSampler(
            backend, runstart_sampler=source, max_candidates=proposal_budget
        )
        for index in range(2, min(len(history), 2 + probes)):
            before = _operation_snapshot(backend)
            started_at = time.perf_counter()
            accepted = False
            handles: tuple[str, ...] = ()
            try:
                # The benchmark knows what action was chosen at this
                # checkpoint in the *full* fixture trace, but a posterior
                # query here must end at the currently observed frame.
                # Do not include that future chosen action in the query.
                at_checkpoint = history[index]
                query = history[:index] + (
                    PublicHistoryStep(
                        at_checkpoint.observation, None, at_checkpoint.legal_actions
                    ),
                )
                handles = sampler.sample_fair_continuations(
                    query,
                    search_rng=random.Random(seed + index),
                    count=1,
                )
                accepted = True
                # A fresh rejection sample must match the complete public frame.
                assert backend.observe(
                    handles[0], InformationPolicy(FAIR_POLICY_ID)
                ) == history[index].observation
            except HistoryConditioningExhausted:
                pass
            finally:
                if handles:
                    backend.release_many(handles)
            records.append({
                "public_frame": index,
                "phase": _phase(history[index]),
                "accepted": accepted,
                "candidates": sampler.last_stats.candidates,
                "replay_steps": sampler.last_stats.replay_steps,
                "wall_seconds": time.perf_counter() - started_at,
                "bridge_operations": _operation_delta(
                    before, _operation_snapshot(backend)
                ),
            })
    return records


def run_benchmark(
    backend: JsonlEmulatorBackend,
    *,
    cohort_sizes: tuple[int, ...],
    post_map_decisions: int,
    proposal_budget: int,
    probes: int,
    search_seed: int,
) -> dict[str, Any]:
    history = _public_history(backend, post_map_decisions=post_map_decisions)
    # Only these public records reach the posterior components.
    report: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "emulator_revision": backend.emulator_revision,
        "prior": "explicit-independent-six-stream-research-prior",
        "target_public_frames": len(history),
        "target_action_kinds": [
            step.chosen_action.kind for step in history
            if step.chosen_action is not None
        ],
        "target_phases": [_phase(step) for step in history],
        "cohorts": [],
        "fresh_rejection": [],
    }
    report["cohorts"] = [
        _benchmark_cohort(
            backend, history, cohort_size=n, seed=search_seed + n
        ) for n in cohort_sizes
    ]
    report["fresh_rejection"] = _benchmark_rejection(
        backend, history,
        proposal_budget=proposal_budget,
        probes=probes,
        seed=search_seed + 0x57A,
    )
    return report


def _sizes(raw: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(piece) for piece in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cohorts must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("cohort sizes must be distinct positive integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", type=_sizes, default=(32, 128))
    parser.add_argument("--post-map-decisions", type=int, default=24)
    parser.add_argument("--rejection-budget", type=int, default=16)
    parser.add_argument("--rejection-probes", type=int, default=3)
    parser.add_argument("--search-seed", type=int, default=711)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.post_map_decisions < 1 or args.rejection_budget < 1 or args.rejection_probes < 0:
        parser.error("post-map decisions, budget must be positive; probes nonnegative")

    with JsonlEmulatorBackend(build=args.build) as backend:
        result = run_benchmark(
            backend,
            cohort_sizes=args.cohorts,
            post_map_decisions=args.post_map_decisions,
            proposal_budget=args.rejection_budget,
            probes=args.rejection_probes,
            search_seed=args.search_seed,
        )
    rendered = json.dumps(result, sort_keys=True, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")

    print(
        "Target public phases:", result["target_phases"],
        "| actions:", result["target_action_kinds"],
    )
    for cohort in result["cohorts"]:
        print(
            f"cohort {cohort['cohort_size']}: survivors "
            f"{[check['survivors'] for check in cohort['checkpoints']]}, "
            f"steps={cohort['total_simulator_steps']}, "
            f"initialize={cohort['initialization_seconds']:.3f}s, "
            f"collapsed={cohort['collapsed']}"
        )
    for probe in result["fresh_rejection"]:
        print(
            f"fresh rejection at frame {probe['public_frame']}: "
            f"accepted={probe['accepted']}, trials={probe['candidates']}, "
            f"replay_steps={probe['replay_steps']}, wall={probe['wall_seconds']:.3f}s"
        )
    if args.json_out is not None:
        print(f"Machine-readable report: {args.json_out}")


if __name__ == "__main__":
    main()
