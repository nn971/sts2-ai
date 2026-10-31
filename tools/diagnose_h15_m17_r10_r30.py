#!/usr/bin/env python3
"""Paired R10/R30 public-state + PUCT diagnosis on the pinned M17 holdout.

This is a DIAGNOSTIC, not another trainer. We replay both raw policies on
identical held-out source seeds, compare BOTH nets at EVERY encountered public
combat decision, then replay selected disagreement roots *exactly* and run
both PUCT teachers against the SAME engine-owned hidden realization.

No source Nature cursor is exposed to model inference. Root-conditioned PUCT
and optional fixed-R10 continuation tests are approximate/privileged teachers,
not independent public-history-posterior estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sts2_ai.emulator.h15_h8_reference import H8_DECLARED_LAW_MODEL
from sts2_ai.nature.combat_scenarios import deck_for_seed, scenario_for_seed
from sts2_ai.training.h15_search_supervision import DEFEAT_MARGIN_REWARD_VERSION

SCHEMA = "sts2-h15-m17-r10-r30-paired-public-state-diagnostic-v1"
MODEL_NAMES = ("r10", "r30")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed63(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 1


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def probabilities(logits: tuple[float, ...]) -> tuple[float, ...]:
    peak = max(logits)
    weights = [math.exp(v - peak) for v in logits]
    z = math.fsum(weights)
    return tuple(w / z for w in weights)


def greedy_index(logits: tuple[float, ...], menu: tuple[Any, ...]) -> int:
    return max(range(len(menu)), key=lambda j: (logits[j], menu[j].action_id))


def jensen_shannon(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Nats; symmetric and bounded by log(2), including zero probabilities."""
    if len(a) != len(b) or not a:
        raise ValueError("Distribution length mismatch")
    midpoint = [(x + y) * 0.5 for x, y in zip(a, b, strict=True)]
    return 0.5 * math.fsum(
        x * math.log(x / m) for x, m in zip(a, midpoint, strict=True) if x > 0
    ) + 0.5 * math.fsum(
        y * math.log(y / m) for y, m in zip(b, midpoint, strict=True) if y > 0
    )


def action_descriptor(action: Any, public: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(action.payload_json)
    except (ValueError, TypeError):
        payload = {}
    card_id = None
    if isinstance(payload, dict):
        instance = payload.get("CardInstanceId")
        combat = public.get("combat")
        hand = combat.get("hand", []) if isinstance(combat, dict) else []
        if isinstance(hand, list):
            for card in hand:
                if isinstance(card, dict) and card.get("instance_id") == instance:
                    card_id = card.get("card_id")
                    break
    return {"action_id": action.action_id, "kind": action.kind, "card_id": card_id}


def paired_prediction(models: dict[str, Any], observation: Any,
                      menu: tuple[Any, ...]) -> dict[str, Any]:
    state = json.loads(observation.payload_json)
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Expected public combat decision")
    predictions = {name: model.predict(observation, menu)
                   for name, model in models.items()}
    probs = {name: probabilities(pred.logits)
             for name, pred in predictions.items()}
    chosen = {name: greedy_index(pred.logits, menu)
              for name, pred in predictions.items()}
    enemy_hp = [enemy["hp"] for enemy in combat.get("enemies", [])
                if isinstance(enemy, dict)
                and type(enemy.get("hp")) is int and enemy["hp"] > 0]
    return {
        "observation_hash": observation.observation_hash,
        "turn": combat.get("turn"),
        "player_hp": state.get("hp"),
        "min_enemy_hp": min(enemy_hp) if enemy_hp else None,
        "total_enemy_hp_visible": sum(enemy_hp),
        "legal_action_count": len(menu),
        "legal_action_ids": [a.action_id for a in menu],
        "nested_choice": "h15_player_choice_kind" in state,
        "r10_raw": action_descriptor(menu[chosen["r10"]], state),
        "r30_raw": action_descriptor(menu[chosen["r30"]], state),
        "r10_value": predictions["r10"].value,
        "r30_value": predictions["r30"].value,
        "r10_unclipped_value": predictions["r10"].unclipped_value,
        "r30_unclipped_value": predictions["r30"].unclipped_value,
        "r10_argmax_probability": probs["r10"][chosen["r10"]],
        "r30_argmax_probability": probs["r30"][chosen["r30"]],
        "policy_js_nats": jensen_shannon(probs["r10"], probs["r30"]),
        "policy_total_variation": 0.5 * sum(
            abs(a - b) for a, b in zip(
                probs["r10"], probs["r30"], strict=True
            )
        ),
        "raw_disagreement": chosen["r10"] != chosen["r30"],
    }


def fixture_entry(cfg: dict[str, Any], case_id: str) -> Any:
    from sts2_ai.emulator.h15_h8_reference import H8FixtureEntry
    scenario = scenario_for_seed(case_id, tuple(cfg["scenarios"]))
    deck = deck_for_seed(case_id, tuple(cfg["decks"]))
    return H8FixtureEntry(
        seed=cfg["fixture_seed"], encounter_id=scenario.encounter_id,
        native_entry=scenario.native_entry, act=scenario.act,
        total_floor=scenario.total_floor, ascension=scenario.ascension,
        first_monster_combat_id=scenario.first_monster_combat_id,
        map_column=scenario.map_column, map_row=scenario.map_row,
        fixture_deck_preset_id=deck,
    )


def trace_raw_trajectory(backend: Any, nature_seed: int,
                         models: dict[str, Any], source_name: str, *,
                         max_decisions: int) -> dict[str, Any]:
    from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
    from sts2_ai.emulator.protocol import InformationPolicy
    from sts2_ai.search.h15_public_combat_adapter import H15PublicCombatAdapter

    handle = backend.begin_reference(nature_seed)
    decisions = []
    policy = InformationPolicy(FAIR_POLICY_ID)
    try:
        for i in range(max_decisions + 1):
            observed = backend.observe(handle, policy)
            public = json.loads(observed.payload_json)
            if not isinstance(public.get("combat"), dict):
                terminal_reward = H15PublicCombatAdapter._combat_result(
                    public, defeat_margin_beta=0.5,
                )
                outcome = (
                    "defeat" if public.get("terminal_outcome") == "defeat"
                    or public.get("hp") == 0 else "victory"
                )
                return {
                    "outcome": outcome,
                    "terminal_reward": terminal_reward,
                    "exit_hp": public.get("hp"),
                    "decisions": decisions,
                }
            if i >= max_decisions:
                raise RuntimeError("Censored diagnostic trajectory")
            menu = backend.legal_actions(handle)
            prediction = paired_prediction(models, observed, menu)
            selected = prediction[f"{source_name}_raw"]["action_id"]
            action = next(a for a in menu if a.action_id == selected)
            prediction["decision_index"] = i
            prediction["source_action_id"] = selected
            prediction["source_action_kind"] = action.kind
            decisions.append(prediction)
            backend.step(handle, action)
        raise AssertionError("Unreachable diagnostic loop exit")
    finally:
        backend.release_many((handle,))


def choose_roots(case: dict[str, Any], *, max_roots: int) -> list[dict[str, Any]]:
    """Only current observations/recorded prefixes; selection is outcome-stratified.

    This is retrospective *diagnostic* state selection, never an online
    policy input or a training target. First disagreement is emphasized.
    """
    if max_roots < 0:
        raise ValueError("Negative diagnostic root budget")
    candidates = []
    for source in MODEL_NAMES:
        trajectory = case["trajectories"][source]["decisions"]
        for i, row in enumerate(trajectory):
            if row["nested_choice"] or row["legal_action_count"] < 2:
                continue
            candidates.append({
                "source_trajectory": source,
                "decision_index": i,
                "row": row,
                "prefix": tuple(d["source_action_id"] for d in trajectory[:i]),
            })
    selected = []
    seen: set[tuple[str, ...]] = set()

    def pick(item: dict[str, Any] | None, reason: str) -> None:
        if item is None or len(selected) >= max_roots:
            return
        if item["prefix"] not in seen:
            seen.add(item["prefix"])
            selected.append({
                "source_trajectory": item["source_trajectory"],
                "decision_index": item["decision_index"],
                "reason": reason,
                "observation_hash": item["row"]["observation_hash"],
                "raw_disagreement": item["row"]["raw_disagreement"],
                "policy_js_nats": item["row"]["policy_js_nats"],
            })

    disagreed = [c for c in candidates if c["row"]["raw_disagreement"]]
    if disagreed:
        # Earliest reachable policy fork (in either actual trajectory).
        pick(min(disagreed, key=lambda c: (
            c["decision_index"], c["source_trajectory"],
        )), "first_policy_disagreement")
        # A later, highly changed policy in a potentially lethal position.
        lethal = [c for c in disagreed if
                  c["row"]["min_enemy_hp"] is not None
                  and c["row"]["min_enemy_hp"] <= 20]
        if lethal:
            pick(max(lethal, key=lambda c: (
                c["row"]["policy_js_nats"], -c["decision_index"],
            )), "near_lethal_disagreement")
        pick(max(disagreed, key=lambda c: c["row"]["policy_js_nats"]),
             "largest_policy_disagreement")
    # Controls are important: test what search does even when both raw
    # checkpoints agree, especially near the terminal combat turn.
    late = [c for c in candidates if
            c["row"]["turn"] is not None and c["row"]["turn"] >= 7]
    if late:
        pick(max(late, key=lambda c: (
            c["row"]["policy_js_nats"], c["decision_index"],
        )), "largest_late_shift")
    for c in sorted(candidates, key=lambda c: (
            -c["row"]["policy_js_nats"], c["source_trajectory"],
            c["decision_index"],
    )):
        pick(c, "largest_remaining_shift")
    return selected


def replay_selected_root(backend: Any, nature_seed: int,
                         trajectory: dict[str, Any],
                         decision_index: int) -> tuple[str, Any, tuple[Any, ...]]:
    from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
    from sts2_ai.emulator.protocol import InformationPolicy

    path = trajectory["decisions"]
    if not 0 <= decision_index < len(path):
        raise ValueError("Selected root outside saved public trajectory")
    handle = backend.begin_reference(nature_seed)
    try:
        policy = InformationPolicy(FAIR_POLICY_ID)
        for i in range(decision_index + 1):
            observed = backend.observe(handle, policy)
            menu = backend.legal_actions(handle)
            if (observed.observation_hash != path[i]["observation_hash"]
                    or [a.action_id for a in menu] !=
                    path[i]["legal_action_ids"]):
                raise RuntimeError(
                    f"Replay public observation/menu mismatched at decision {i}"
                )
            if i == decision_index:
                return handle, observed, menu
            matching = [action for action in menu if
                        action.action_id == path[i]["source_action_id"]]
            if len(matching) != 1:
                raise RuntimeError("Replay public legal menu changed")
            backend.step(handle, matching[0])
        raise AssertionError("Unreachable public replay")
    except BaseException:
        backend.release_many((handle,))
        raise


def sample_fixed_r10_continuation(
    adapter: Any, root: Any, first_action: Any, r10_model: Any, *,
    repetitions: int, seed: int, max_decisions: int,
) -> dict[str, Any]:
    """Paired *root-conditioned* forced-first-action outcomes.

    After the forced action, sample from the SAME frozen R10 continuation
    policy for both actions. This is a common-policy diagnostic, not an
    unbiased causal estimate under the public hidden-state posterior.
    """
    rewards = []
    for i in range(repetitions):
        rng = random.Random(seed63(f"{seed}:continuation:{i}"))
        adapter.begin_simulation(root, rng)
        try:
            frame = adapter.step(root, first_action, rng)
            for _ in range(max_decisions):
                if frame.terminal_reward is not None:
                    rewards.append(frame.terminal_reward)
                    break
                prior = probabilities(
                    r10_model.predict(
                        frame.observation, frame.legal_actions,
                    ).logits
                )
                chosen = rng.choices(frame.legal_actions, weights=prior, k=1)[0]
                frame = adapter.step(frame, chosen, rng)
            else:
                raise RuntimeError("Censored forced-action continuation")
        finally:
            adapter.end_simulation()
    return {
        "rollouts": repetitions,
        "mean_reward": sum(rewards) / len(rewards),
        "win_fraction": sum(x >= 1.0 for x in rewards) / len(rewards),
    }


def analyze_root(backend: Any, nature_seed: int, case: dict[str, Any],
                 selection: dict[str, Any], models: dict[str, Any], *,
                 simulations: int, tree_depth: int, rollout_depth: int,
                 counterfactual_rollouts: int) -> dict[str, Any]:
    from sts2_ai.emulator.chance import PublicHistoryStep
    from sts2_ai.search.combat_puct import CombatPuct
    from sts2_ai.search.h15_public_combat_adapter import (
        H15RootConditionedCombatAdapter,
    )

    handle, observed, menu = replay_selected_root(
        backend, nature_seed,
        case["trajectories"][selection["source_trajectory"]],
        selection["decision_index"],
    )
    try:
        adapter = H15RootConditionedCombatAdapter(
            backend, handle, defeat_margin_beta=0.5,
        )
        root = adapter.root((PublicHistoryStep(observed, None, menu),))
        prediction = paired_prediction(models, observed, menu)
        search_results = {}
        diagnostics = {}
        public = json.loads(observed.payload_json)
        for name in MODEL_NAMES:
            search_seed = seed63(
                f"{case['case_id']}:{selection['source_trajectory']}:"
                f"{selection['decision_index']}:paired-search"
            )
            result = CombatPuct(
                adapter, models[name],
                seed=search_seed,
                max_depth=tree_depth,
                rollout_depth=rollout_depth,
                leaf_mode="terminal",
                tree_expansion_mode="full_depth",
            ).search(root, simulations=simulations)
            if (result.terminal_evaluations != simulations
                    or result.neural_cutoffs):
                raise RuntimeError("Paired terminal PUCT did not terminate")
            search_results[name] = result
            best = max(range(len(menu)), key=lambda i: (
                result.visits[i], menu[i].action_id,
            ))
            diagnostics[name] = {
                "selected_action": action_descriptor(menu[best], public),
                "expanded_information_sets": result.expanded_information_sets,
                "maximum_expanded_depth": result.maximum_expanded_depth,
                "total_transitions": result.transitions,
                "root_mean_value": result.root_mean_value,
                "root_visits": list(result.visits),
                "root_q_values": list(result.mean_action_values),
                "visit_probability": list(result.policy),
            }
        priors = {
            name: probabilities(
                models[name].predict(observed, menu).logits,
            )
            for name in MODEL_NAMES
        }
        actions = []
        for i, action in enumerate(menu):
            actions.append({
                **action_descriptor(action, public),
                **{
                    f"{name}_prior_probability": priors[name][i]
                    for name in MODEL_NAMES
                },
                **{
                    f"{name}_puct_visits": search_results[name].visits[i]
                    for name in MODEL_NAMES
                },
                **{
                    f"{name}_puct_q": search_results[name].mean_action_values[i]
                    for name in MODEL_NAMES
                },
            })
        row = {
            "case_id": case["case_id"],
            "r10_outcome": case["trajectories"]["r10"]["outcome"],
            "r30_outcome": case["trajectories"]["r30"]["outcome"],
            **selection,
            **prediction,
            "search_r10": diagnostics["r10"],
            "search_r30": diagnostics["r30"],
            "search_disagreement": (
                diagnostics["r10"]["selected_action"]["action_id"]
                != diagnostics["r30"]["selected_action"]["action_id"]
            ),
            "actions": actions,
            "root_conditioning": (
                "same H8 public observation, mechanics and ReferenceNature "
                "cursor; NOT a public posterior"
            ),
        }
        raw_a = prediction["r10_raw"]["action_id"]
        raw_b = prediction["r30_raw"]["action_id"]
        if counterfactual_rollouts and raw_a != raw_b:
            action_a = next(a for a in menu if a.action_id == raw_a)
            action_b = next(a for a in menu if a.action_id == raw_b)
            common_seed = seed63(
                f"{case['case_id']}:{selection['decision_index']}:paired-actions"
            )
            row["forced_action_fixed_r10_continuations"] = {
                "r10_raw_action": sample_fixed_r10_continuation(
                    adapter, root, action_a, models["r10"],
                    repetitions=counterfactual_rollouts,
                    seed=common_seed, max_decisions=rollout_depth,
                ),
                "r30_raw_action": sample_fixed_r10_continuation(
                    adapter, root, action_b, models["r10"],
                    repetitions=counterfactual_rollouts,
                    seed=common_seed, max_decisions=rollout_depth,
                ),
            }
        return row
    finally:
        backend.release_many((handle,))


def process_cases(
    stage: str, cfg: dict[str, Any], models: dict[str, str],
    case_ids: list[str], kwargs: dict[str, Any],
    trace_cases: dict[str, Any] | None,
) -> dict[str, Any]:
    """One worker owns one engine JSONL session and two frozen CPU models."""
    import torch
    from sts2_ai.emulator.h15_h8_reference import H8ReferenceCombatBackend
    from sts2_ai.models.h15_policy_value import H15CombatEvaluator

    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(cfg["emulator_root"]) / "bindings/python"))
    from sts2_emulator import Session  # type: ignore[import-not-found]

    evaluators = {
        name: H15CombatEvaluator.load(Path(models[name]), device="cpu")
        for name in MODEL_NAMES
    }
    started = time.monotonic()
    cases = []
    with Session(["dotnet", cfg["emulator_dll"], "prototype-ai-jsonl"]) as session:
        for case_id in case_ids:
            backend = H8ReferenceCombatBackend(
                session, fixture_entry(cfg, case_id),
                emulator_revision=cfg["emulator_revision"],
                use_declared_chance_laws=True,
            )
            nature_seed = seed63(case_id + ":nature")
            try:
                if stage == "traces":
                    trajectories = {
                        name: trace_raw_trajectory(
                            backend, nature_seed, evaluators, name,
                            max_decisions=cfg["max_decisions"],
                        )
                        for name in MODEL_NAMES
                    }
                    cases.append({"case_id": case_id,
                                  "trajectories": trajectories})
                else:
                    if trace_cases is None or case_id not in trace_cases:
                        raise ValueError("Missing recorded source trajectory")
                    case = trace_cases[case_id]
                    roots = [
                        analyze_root(
                            backend, nature_seed, case, selection, evaluators,
                            simulations=kwargs["simulations"],
                            tree_depth=kwargs["tree_depth"],
                            rollout_depth=kwargs["rollout_depth"],
                            counterfactual_rollouts=kwargs["counterfactual_rollouts"],
                        )
                        for selection in choose_roots(
                            case, max_roots=kwargs["roots_per_case"],
                        )
                    ]
                    cases.append({"case_id": case_id, "roots": roots})
            finally:
                backend.close()
                if backend.live_handles:
                    raise RuntimeError("Paired comparison leaked H8 handles")
    return {
        "stage": stage, "cases": cases,
        "case_ids": case_ids,
        "checkpoint_hashes": {name: digest(Path(models[name]))
                              for name in MODEL_NAMES},
        "seconds": time.monotonic() - started,
    }


def _worker_safe(*args: Any) -> dict[str, Any]:
    try:
        return process_cases(*args)
    except Exception:
        return {"worker_error": traceback.format_exc()}


def parallel_cases(
    stage: str, cfg: dict[str, Any], checkpoints: dict[str, str],
    case_ids: list[str], kwargs: dict[str, Any],
    trace_cases: dict[str, Any] | None, out: Path, workers: int,
) -> list[dict[str, Any]]:
    if not case_ids:
        return []
    destination = out / stage
    destination.mkdir(parents=True, exist_ok=True)
    groups = [case_ids[i::workers] for i in range(min(workers, len(case_ids)))]
    hashes = {name: digest(Path(checkpoints[name])) for name in MODEL_NAMES}
    all_cases = {}
    pending = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, group in enumerate(groups):
            filename = destination / f"worker-{i:02d}.json"
            if filename.exists():
                saved = json.loads(filename.read_text(encoding="utf-8"))
                if (saved.get("case_ids") != group
                        or saved.get("checkpoint_hashes") != hashes
                        or saved.get("stage") != stage):
                    raise RuntimeError("Incompatible paired diagnostic worker cache")
                for case in saved["cases"]:
                    all_cases[case["case_id"]] = case
            else:
                future = pool.submit(
                    _worker_safe, stage, cfg, checkpoints, group, kwargs,
                    trace_cases if stage == "search" else None,
                )
                pending.append((future, filename))
        for future in as_completed([entry[0] for entry in pending]):
            filename = next(path for f, path in pending if f is future)
            report = future.result()
            if "worker_error" in report:
                raise RuntimeError(
                    "Paired H15 diagnostic failed in "
                    + filename.name + "\n" + report["worker_error"]
                )
            if report["checkpoint_hashes"] != hashes:
                raise RuntimeError("Worker checkpoint hash changed")
            write_json(filename, report)
            for case in report["cases"]:
                all_cases[case["case_id"]] = case
            print(f"[h15-paired-{stage}] {filename.name} "
                  f"cases={len(report['cases'])} "
                  f"seconds={report['seconds']:.1f}", flush=True)
    if set(all_cases) != set(case_ids):
        raise RuntimeError("Paired diagnostic lost/duplicated source cases")
    return [all_cases[c] for c in case_ids]


def choose_search_cases(traces: list[dict[str, Any]], limit: int) -> list[str]:
    """Prefer win->loss regressions, then recovered wins and control cases."""
    def priority(case: dict[str, Any]) -> tuple[Any, ...]:
        a = case["trajectories"]["r10"]["outcome"]
        b = case["trajectories"]["r30"]["outcome"]
        all_rows = [
            row for source in MODEL_NAMES
            for row in case["trajectories"][source]["decisions"]
            if not row["nested_choice"]
        ]
        max_js = max((row["policy_js_nats"] for row in all_rows), default=0)
        cls = 0 if (a, b) == ("victory", "defeat") else (
            1 if (a, b) == ("defeat", "victory") else 2
        )
        return cls, -max_js, case["case_id"]
    return [case["case_id"] for case in sorted(traces, key=priority)[:limit]]


def make_report(
    trace_cases: list[dict[str, Any]],
    search_cases: list[dict[str, Any]],
    cfg: dict[str, Any], checkpoint_hashes: dict[str, str],
) -> dict[str, Any]:
    n = len(trace_cases)
    r10_wins = sum(x["trajectories"]["r10"]["outcome"] == "victory"
                   for x in trace_cases)
    r30_wins = sum(x["trajectories"]["r30"]["outcome"] == "victory"
                   for x in trace_cases)
    regressions = [c["case_id"] for c in trace_cases if
                   (c["trajectories"]["r10"]["outcome"],
                    c["trajectories"]["r30"]["outcome"])
                   == ("victory", "defeat")]
    improvements = [c["case_id"] for c in trace_cases if
                    (c["trajectories"]["r10"]["outcome"],
                     c["trajectories"]["r30"]["outcome"])
                    == ("defeat", "victory")]
    rows = [d for case in trace_cases for name in MODEL_NAMES
            for d in case["trajectories"][name]["decisions"]
            if not d["nested_choice"]]
    roots = [root for case in search_cases for root in case["roots"]]
    def avg(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None
    forced = [
        r for r in roots if "forced_action_fixed_r10_continuations" in r
    ]
    return {
        "schema": SCHEMA,
        "confidence": "diagnostic / single training seed; source-run-disjoint heldout",
        "chance_prior": H8_DECLARED_LAW_MODEL,
        "search_provider": "root_conditioned (privileged approximate)",
        "reward_version": DEFEAT_MARGIN_REWARD_VERSION,
        "checkpoint_hashes": checkpoint_hashes,
        "fixture_seed": cfg["fixture_seed"],
        "seed_prefix": cfg["seed_prefix"],
        "holdout_cases_replayed": n,
        "r10_wins": r10_wins,
        "r30_wins": r30_wins,
        "r10_win_rate": r10_wins / n,
        "r30_win_rate": r30_wins / n,
        "regressed_cases_count": len(regressions),
        "improved_cases_count": len(improvements),
        "regressed_case_ids": regressions,
        "improved_case_ids": improvements,
        "public_decisions_paired": len(rows),
        "raw_action_disagreement_fraction": avg([
            float(x["raw_disagreement"]) for x in rows
        ]),
        "mean_policy_js_nats": avg([x["policy_js_nats"] for x in rows]),
        "mean_normalized_value_change_r30_minus_r10": avg([
            x["r30_value"] - x["r10_value"] for x in rows
        ]),
        "searched_shared_public_roots": len(roots),
        "search_action_disagreement_fraction": avg([
            float(r["search_disagreement"]) for r in roots
        ]),
        "mean_search_depth_r10": avg([
            r["search_r10"]["maximum_expanded_depth"] for r in roots
        ]),
        "mean_search_depth_r30": avg([
            r["search_r30"]["maximum_expanded_depth"] for r in roots
        ]),
        "mean_search_root_value_r10": avg([
            r["search_r10"]["root_mean_value"] for r in roots
        ]),
        "mean_search_root_value_r30": avg([
            r["search_r30"]["root_mean_value"] for r in roots
        ]),
        "paired_fixed_r10_rollout_roots": len(forced),
        "paired_fixed_r10_rollout_mean_reward_advantage_of_r10_action": avg([
            r["forced_action_fixed_r10_continuations"]["r10_raw_action"][
                "mean_reward"
            ] - r["forced_action_fixed_r10_continuations"]["r30_raw_action"][
                "mean_reward"
            ] for r in forced
        ]),
        "interpretation_note": (
            "This report identifies where trained policies disagree and "
            "what their root-conditioned searches prefer. Rollout-based "
            "action quality is conditioned on ONE hidden realization and "
            "a fixed R10 continuation policy; small samples cannot prove "
            "which action is optimal under unseen hidden states."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path)
    p.add_argument("--emulator-dll", type=Path)
    p.add_argument("--cases", type=int, default=64)
    p.add_argument("--search-cases", type=int, default=12)
    p.add_argument("--roots-per-case", type=int, default=3)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--tree-depth", type=int, default=16)
    p.add_argument("--rollout-depth", type=int, default=256)
    p.add_argument("--counterfactual-rollouts", type=int, default=8)
    p.add_argument("--max-decisions", type=int, default=512)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if min(args.cases, args.search_cases, args.roots_per_case,
           args.simulations, args.tree_depth, args.rollout_depth,
           args.max_decisions, args.workers) < 1 or (
           args.counterfactual_rollouts < 0 or args.search_cases > args.cases):
        p.error("Invalid paired diagnostic budget")

    run = args.run_root.resolve()
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if (cfg.get("reward_version") != DEFEAT_MARGIN_REWARD_VERSION
            or cfg.get("defeat_margin_beta") != 0.5
            or cfg.get("search_provider") != "root_conditioned"
            or cfg.get("tree_expansion_mode") != "full_depth"):
        p.error("Expected pinned M17 beta05 root-conditioned full-depth experiment")
    cfg["emulator_root"] = str(
        args.emulator_root.resolve() if args.emulator_root
        else Path(cfg["emulator_root"])
    )
    cfg["emulator_dll"] = str(
        args.emulator_dll.resolve() if args.emulator_dll
        else Path(cfg["emulator_dll"])
    )
    for file in ("emulator_root", "emulator_dll"):
        if not Path(cfg[file]).exists():
            p.error(f"Missing {file}: {cfg[file]}")
    checkpoints = {
        name: run / f"round-{r:04d}" / "student.pt"
        for name, r in (("r10", 10), ("r30", 30))
    }
    if any(not f.is_file() for f in checkpoints.values()):
        p.error("Both R10 and R30 checkpoints must exist in the M17 run")
    hashes = {name: digest(path) for name, path in checkpoints.items()}
    if (run / "progress.json").exists():
        progress = json.loads((run / "progress.json").read_text())
        for name, r in (("r10", 10), ("r30", 30)):
            entry = next(
                (row for row in progress["rounds"] if row["round"] == r),
                None,
            )
            if entry is None or not entry.get("evaluation"):
                p.error(f"Missing evaluated checkpoint for {name}")
            report_path = run / f"round-{r:04d}" / "round.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("updated_sha256") != hashes[name]:
                    p.error(f"Checkpoint SHA mismatch: {name}")
    output = args.output_root.resolve()
    config = {
        "schema": SCHEMA,
        "run_root": str(run),
        "emulator_revision": cfg["emulator_revision"],
        "emulator_dll_sha256": digest(Path(cfg["emulator_dll"])),
        "checkpoint_hashes": hashes,
        "source_configuration_sha256": digest(run / "config.json"),
        "cases": args.cases,
        "search_cases": args.search_cases,
        "roots_per_case": args.roots_per_case,
        "simulations": args.simulations,
        "tree_depth": args.tree_depth,
        "rollout_depth": args.rollout_depth,
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "workers": args.workers,
        "seed_prefix": cfg["seed_prefix"],
    }
    print("[h15-paired-preflight] " + json.dumps(config, sort_keys=True),
          flush=True)
    if not args.execute:
        print("[h15-paired-preflight] add --execute to replay and search",
              flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise RuntimeError("Paired diagnostic resume configuration changed")
    else:
        write_json(config_file, config)
    case_ids = [
        f"{cfg['seed_prefix']}:heldout:i{i:05d}" for i in range(args.cases)
    ]
    paths = {name: str(path) for name, path in checkpoints.items()}
    traced = parallel_cases(
        "traces", cfg, paths, case_ids, {}, None, output, args.workers,
    )
    by_case = {case["case_id"]: case for case in traced}
    # Check that identical training-script source fixtures were replayed.
    if args.cases == cfg["eval_combats"]:
        for name, round_number in (("r10", 10), ("r30", 30)):
            round_report = json.loads(
                (run / f"round-{round_number:04d}" / "round.json").read_text()
            )
            expected = round_report["evaluation"]["h15_raw"]["victories"]
            actual = sum(x["trajectories"][name]["outcome"] == "victory"
                         for x in traced)
            if actual != expected:
                raise RuntimeError(
                    f"R{round_number} reproduced {actual} wins, expected "
                    f"{expected}. Check emulator/model/seed identity."
                )
    selected = choose_search_cases(traced, args.search_cases)
    searched = parallel_cases(
        "search", cfg, paths, selected, {
            "roots_per_case": args.roots_per_case,
            "simulations": args.simulations,
            "tree_depth": args.tree_depth,
            "rollout_depth": args.rollout_depth,
            "counterfactual_rollouts": args.counterfactual_rollouts,
        }, by_case, output, args.workers,
    )
    report = make_report(traced, searched, cfg, hashes)
    write_json(output / "paired-report.json", report)
    with (output / "paired-roots.jsonl").open("w", encoding="utf-8") as f:
        for item in searched:
            for root in item["roots"]:
                f.write(json.dumps(root, sort_keys=True) + "\n")
    with (output / "public-decisions.jsonl").open("w", encoding="utf-8") as f:
        for case in traced:
            for name in MODEL_NAMES:
                for row in case["trajectories"][name]["decisions"]:
                    f.write(json.dumps({
                        "case_id": case["case_id"],
                        "source_trajectory": name,
                        **row,
                    }, sort_keys=True) + "\n")
    print("[h15-paired-complete] " + json.dumps(report, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
