"""Fair-observation temperature sweeps and Act-1 boss failure diagnostics.

No hidden handles, RNG seed observations, simulator introspection or teacher
actions are used. Identical emulator seed strings and paired random streams
make diagnostics reproducible, without claiming independent game randomness.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, deque
from collections.abc import Sequence
from typing import Any

from sts2_ai.agents import Decision, NeuralGreedyAgent
from sts2_ai.agents.neural_temperature import NeuralTemperatureAgent
from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, Observation
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, require_environment
from sts2_ai.models.neural import NeuralPolicyValueModel

from .run import RunSummary, play_run
from .selfplay_metrics import compare_completed_pairs, summarize_completed_runs

DIAGNOSTIC_SCHEMA = "sts2-neural-temperature-diagnostics-v1"


def _temperature_key(value: float | None) -> str:
    return "greedy" if value is None else f"t{value:g}"


def _rng_seed(model_id: str, run_seed: str) -> int:
    # The same action-selection random stream is used for a given emulator
    # seed at every temperature; policy trajectories may diverge.
    digest = hashlib.sha256(
        f"sts2-neural-temperature-v1:{model_id}:{run_seed}".encode()
    ).digest()
    return int.from_bytes(digest, "big")


def _public_enemy_snapshot(combat: dict[str, Any]) -> list[dict[str, Any]]:
    enemies = combat.get("enemies")
    if not isinstance(enemies, list):
        return []
    snapshots = []
    for raw in enemies:
        if not isinstance(raw, dict):
            continue
        # Copy public fields only. This explicitly excludes internal AI slots,
        # RNG states and exact-state handles.
        snapshots.append({
            key: raw[key]
            for key in (
                "enemy_id", "id", "name", "hp", "max_hp", "block",
                "intent", "intent_id", "move_id"
            )
            if key in raw and type(raw[key]) in (str, int, float, bool)
        })
    return snapshots


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def diagnose_neural_temperatures(
    backend: EmulatorBackend,
    model: NeuralPolicyValueModel,
    *,
    seeds: Sequence[str],
    temperatures: Sequence[float | None] = (1.0, 0.5, 0.25, None),
    environment: str = NATIVE_OVERGROWTH,
    policy_id: str = "prototype-fair-v0",
    max_decisions: int = 4096,
    max_boss_actions: int = 48,
) -> dict[str, Any]:
    """Evaluate inference policies without touching training or optimizer state.

    'greedy' uses the exact production NeuralGreedyAgent tie-breaking logic.
    Finite positive T samples from softmax(logits/T). The learner currently
    collects its own T=1 trajectories; this routine does not change it.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected nonempty unique diagnostic seeds")
    if not temperatures or len({_temperature_key(t) for t in temperatures}) != len(
        temperatures
    ):
        raise ValueError("Expected nonempty unique diagnostic temperatures")
    if max_decisions <= 0 or max_boss_actions <= 0:
        raise ValueError("Decision caps must be positive")
    for temperature in temperatures:
        if temperature is not None and (
            not math.isfinite(temperature) or temperature <= 0
        ):
            raise ValueError("Temperatures must be finite and positive, or greedy")

    require_environment(backend, environment)
    raw_runs: dict[str, list[RunSummary]] = {}
    run_diagnostics: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for temperature in temperatures:
        label = _temperature_key(temperature)
        trajectories: list[RunSummary] = []
        records: list[dict[str, Any]] = []
        all_action_kinds: Counter[str] = Counter()
        entropy_sum = max_probability_sum = chosen_probability_sum = 0.0
        sample_count = 0
        combat_sample_count = 0

        for seed in seeds:
            import random

            agent = (
                NeuralGreedyAgent(model, content_hash=model.model_id)
                if temperature is None
                else NeuralTemperatureAgent(
                    model,
                    temperature=temperature,
                    rng=random.Random(_rng_seed(model.model_id, seed)),
                )
            )
            action_kinds: Counter[str] = Counter()
            action_count = 0
            boss_floor_seen = False
            boss_combat_seen = False
            boss_decision_count = 0
            first_boss_actions: list[dict[str, Any]] = []
            last_boss_actions: deque[dict[str, Any]] = deque(
                maxlen=max_boss_actions
            )
            boss_start: dict[str, Any] | None = None
            boss_last: dict[str, Any] | None = None
            last_hp: int | None = None
            last_action_kind: str | None = None
            last_combat: dict[str, Any] | None = None

            def observe_decision(
                _state: str,
                observation: Observation,
                legal: tuple[LegalAction, ...],
                decision: Decision,
                *,
                _action_kinds: Counter[str] = action_kinds,
                _all_action_kinds: Counter[str] = all_action_kinds,
                _temperature: float | None = temperature,
                _first_boss_actions: list[dict[str, Any]] = first_boss_actions,
                _last_boss_actions: deque[dict[str, Any]] = last_boss_actions,
            ) -> None:
                nonlocal action_count, sample_count, combat_sample_count
                nonlocal entropy_sum, max_probability_sum, chosen_probability_sum
                nonlocal boss_floor_seen, boss_combat_seen, boss_decision_count
                nonlocal boss_start, boss_last, last_hp, last_action_kind
                nonlocal last_combat

                public = json.loads(observation.payload_json)
                if not isinstance(public, dict):
                    raise ValueError("Malformed public observation")
                hp = public.get("hp")
                if type(hp) is int:
                    last_hp = hp
                action_count += 1
                last_action_kind = decision.action.kind
                _action_kinds[decision.action.kind] += 1
                __all_action_kinds[decision.action.kind] += 1
                combat = public.get("combat")
                in_combat = isinstance(combat, dict)
                if in_combat:
                    combat_sample_count += 1
                if _temperature is None:
                    chosen_p = max_p = 1.0
                    entropy = 0.0
                else:
                    metadata = json.loads(decision.metadata_json)
                    chosen_p = float(metadata["chosen_probability"])
                    max_p = float(metadata["max_probability"])
                    entropy = float(metadata["normalized_entropy"])
                chosen_probability_sum += chosen_p
                max_probability_sum += max_p
                entropy_sum += entropy
                sample_count += 1

                if public.get("act") == 1 and public.get("floor") == 16:
                    boss_floor_seen = True
                    if in_combat:
                        assert isinstance(combat, dict)
                        boss_combat_seen = True
                        enemies = _public_enemy_snapshot(combat)
                        turn = combat.get("turn")
                        snapshot: dict[str, Any] = {
                            "decision": action_count,
                            "turn": turn if type(turn) is int else None,
                            "hp": hp if type(hp) is int else None,
                            "enemies": enemies,
                            "action_kind": decision.action.kind,
                            "action_id": decision.action.action_id,
                            "selected_probability": chosen_p,
                            "normalized_entropy": entropy,
                            "legal_actions": len(legal),
                        }
                        if boss_start is None:
                            boss_start = {
                                "hp": snapshot["hp"],
                                "enemies": enemies,
                                "turn": snapshot["turn"],
                            }
                        boss_last = snapshot
                        boss_decision_count += 1
                        if len(_first_boss_actions) < min(8, max_boss_actions):
                            _first_boss_actions.append(snapshot)
                        _last_boss_actions.append(snapshot)
                if in_combat:
                    assert isinstance(combat, dict)
                    last_combat = {
                        "turn": combat.get("turn"),
                        "enemies": _public_enemy_snapshot(combat),
                    }

            summary = play_run(
                backend,
                agent,
                seed=seed,
                policy=InformationPolicy(policy_id),
                max_decisions=max_decisions,
                environment=environment,
                decision_observer=observe_decision,
            )
            trajectories.append(summary)
            records.append({
                "seed": seed,
                "outcome": summary.outcome,
                "censored": summary.censored,
                "terminal_act": summary.terminal_act,
                "terminal_floor": summary.terminal_floor,
                "frontier_progress": summary.frontier_progress,
                "decisions": summary.decisions,
                "act1_cleared": summary.act1_cleared,
                "last_observed_hp_before_terminal": last_hp,
                "terminal_hp": (
                    summary.hp_trajectory[-1] if summary.hp_trajectory else None
                ),
                "last_selected_action_kind": last_action_kind,
                "last_public_combat": last_combat if summary.outcome == "defeat" else None,
                "action_kind_counts": dict(sorted(action_kinds.items())),
                "boss_floor_seen": boss_floor_seen or (
                    summary.terminal_act == 1
                    and summary.terminal_floor == 16
                ),
                "boss_combat_seen": boss_combat_seen,
                "boss_combat_decisions": boss_decision_count,
                "boss_initial_public_state": boss_start,
                "boss_last_public_state": boss_last,
                "boss_first_actions": first_boss_actions,
                "boss_last_actions": list(last_boss_actions),
            })

        raw_runs[label] = trajectories
        run_diagnostics[label] = records
        summary_row = summarize_completed_runs(trajectories)
        summaries[label] = {
            **summary_row,
            "mean_normalized_action_entropy": (
                entropy_sum / sample_count if sample_count else None
            ),
            "mean_selected_action_probability": (
                chosen_probability_sum / sample_count if sample_count else None
            ),
            "mean_max_action_probability": (
                max_probability_sum / sample_count if sample_count else None
            ),
            "observed_policy_decisions": sample_count,
            "observed_combat_decisions": combat_sample_count,
            "action_kind_counts": dict(sorted(all_action_kinds.items())),
            "act1_clears": sum(row.act1_cleared is True for row in trajectories),
            "boss_floor_reaches": sum(row["boss_floor_seen"] for row in records),
            "boss_combat_entries": sum(row["boss_combat_seen"] for row in records),
            "boss_combat_defeats": sum(
                row["boss_combat_seen"] and row["outcome"] == "defeat"
                and row["terminal_act"] == 1 and row["terminal_floor"] == 16
                for row in records
            ),
            "mean_terminal_floor_completed_only": _mean_or_none([
                float(row.terminal_floor)
                for row in trajectories
                if not row.censored and row.terminal_floor is not None
            ]),
        }

    paired = {}
    if "greedy" in raw_runs:
        for label, trajectories in raw_runs.items():
            if label != "greedy":
                paired[f"{label}_vs_greedy"] = compare_completed_pairs(
                    trajectories, raw_runs["greedy"]
                )

    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "emulator_revision": backend.emulator_revision,
        "environment": environment,
        "information_policy": policy_id,
        "model_id": model.model_id,
        "temperatures": [_temperature_key(t) for t in temperatures],
        "seeds": list(seeds),
        "max_decisions": max_decisions,
        "max_boss_actions": max_boss_actions,
        "summary": summaries,
        "paired_completed_only": paired,
        "runs": run_diagnostics,
        "method": (
            "Temperature is applied to public legal-action logits only. "
            "Each temperature shares the emulator seed and a deterministic "
            "random-number stream for sampling; diverging trajectories do not "
            "imply identical hidden chance draws. No training update performed."
        ),
    }
