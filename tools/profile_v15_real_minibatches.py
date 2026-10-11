#!/usr/bin/env python3
"""Profile real public PPO combat minibatches on CPU/CUDA (no model training).

Collect a frozen, reusable PUBLIC-ONLY corpus once from the pinned emulator,
using completed combat HP-preservation GAE. All subsequent device and batch
comparisons replay the exact same observations, legal menus, and PPO targets.
A benchmark optimizer changes only throwaway in-memory weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import (
    ObjectSet,
    extract_actionable_combat_objects,
    legal_action_object_bindings,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.models.relational_objects import RelationalObjectTorch
from sts2_ai.training.parallel_rollouts import episode_actor_seed
from sts2_ai.training.pooled_ppo_v15 import prepare_pooled_samples
from sts2_ai.training.selfplay import Episode, collect_public_episode

SCHEMA = "sts2-v15g-public-combat-ppo-corpus-v1"
REPORT_SCHEMA = "sts2-v15g-real-minibatch-profile-v1"


def _digest(obj: Any) -> str:
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_corpus(
    episodes: list[Episode], base: PhaseSplitNeuralModel,
    manifest: dict[str, Any], *, revision: str, seed: int = 71,
    temperature: float = 0.75,
) -> dict[str, Any]:
    """Compute real on-policy combat targets once; retain *only* public data."""
    import torch

    decisions = [
        episode.decisions[i]
        for episode in episodes
        for segment in episode.combat_outcomes
        for i in range(segment.start_decision, segment.end_decision)
    ]
    if not decisions:
        raise ValueError("No resolved combat segments; increase --runs")
    objects: list[ObjectSet] = []
    for decision in decisions:
        state = json.loads(decision.observation.payload_json)
        if not isinstance(state, dict) or not isinstance(state.get("combat"), dict):
            raise ValueError("Noncombat observation in resolved segment")
        objects.append(extract_actionable_combat_objects(
            state, [(a.kind, a.payload_json) for a in decision.legal_actions],
        ))
    vocabulary = ObjectVocabulary.from_object_sets(objects, manifest=manifest)
    torch.manual_seed(seed)
    head = RelationalObjectTorch(vocabulary, width=16)
    preparation_started = time.perf_counter()
    samples, _ = prepare_pooled_samples(
        episodes, base.combat, head,
        temperature=temperature, gamma=0.995, gae_lambda=0.98,
    )
    target_preparation_seconds = time.perf_counter() - preparation_started
    if len(samples) != len(decisions):
        raise ValueError("Resolved combat samples and public frames do not align")
    records = []
    for decision, sample in zip(decisions, samples, strict=True):
        records.append({
            "observation": decision.observation.payload_json,
            "legal_actions": [
                {
                    "kind": action.kind,
                    "payload_json": action.payload_json,
                }
                for action in decision.legal_actions
            ],
            "baseline": {
                "value": sample.baseline.value,
                "action_logits": list(sample.baseline.action_logits),
            },
            "chosen_index": sample.chosen_index,
            "old_logprob": sample.old_logprob,
            "advantage": sample.advantage,
            "target": sample.target,
        })
    data = {
        "schema": SCHEMA, "emulator_revision": revision,
        "base_sha256": _digest(base.to_dict()),
        "temperature": temperature, "seed": seed,
        "manifest": manifest, "records": records,
        "resolved_combats": sum(len(ep.combat_outcomes) for ep in episodes),
        "source_runs": len(episodes),
        "target_preparation_seconds": target_preparation_seconds,
    }
    return {**data, "sha256": _digest(data)}


def load_corpus(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("Invalid v15g public corpus schema")
    contents = {k: v for k, v in raw.items() if k != "sha256"}
    if not isinstance(raw.get("sha256"), str) or _digest(contents) != raw["sha256"]:
        raise ValueError("Public corpus checksum mismatch")
    if not isinstance(raw.get("records"), list) or not raw["records"]:
        raise ValueError("Public corpus contains no resolved combats")
    if not isinstance(raw.get("manifest"), dict):
        raise ValueError("Public corpus is missing content manifest")
    return raw


def prepare_records(records: list[dict[str, Any]]):
    """Time decoding, public-object extraction and legal-action linking.

    Results can be reused unchanged by both CPU and GPU runs. The JSONL
    corpus contains no canonical state hash, private RNG, or draw order.
    """
    objects = []
    bindings = []
    baselines = []
    timings: dict[str, float] = defaultdict(float)
    for record in records:
        tick = time.perf_counter()
        state = json.loads(record["observation"])
        timings["json_decode"] += time.perf_counter() - tick
        tick = time.perf_counter()
        obj = extract_actionable_combat_objects(
            state,
            [(a["kind"], a["payload_json"]) for a in record["legal_actions"]],
        )
        obj.index()
        timings["object_extraction"] += time.perf_counter() - tick
        tick = time.perf_counter()
        menu = tuple(
            (
                action["kind"],
                legal_action_object_bindings(
                    action["kind"], action["payload_json"], obj,
                ),
            )
            for action in record["legal_actions"]
        )
        timings["action_binding"] += time.perf_counter() - tick
        logit_values = record["baseline"]["action_logits"]
        if not (len(menu) == len(logit_values)
                and 0 <= record["chosen_index"] < len(menu)):
            raise ValueError("Malformed combat action menu in corpus")
        objects.append(obj)
        bindings.append(menu)
        baselines.append(PolicyValueEstimate(
            tuple(float(v) for v in logit_values),
            float(record["baseline"]["value"]),
        ))
    return objects, bindings, baselines, dict(timings)


def _normalized_advantages(rows: list[dict[str, Any]]) -> list[float]:
    """Match v15 PPO normalization over the entire selected replay buffer."""
    values = [float(row["advantage"]) for row in rows]
    avg = statistics.fmean(values)
    scale = math.sqrt(
        statistics.fmean((x - avg) ** 2 for x in values) + 1e-8
    )
    return [max(-5.0, min(5.0, (x - avg) / scale)) for x in values]


def _ppo_loss(torch: Any, values: Any, menus: tuple[Any, ...],
              rows: list[dict[str, Any]], temperature: float,
              *, normalized_advantages: list[float] | None = None) -> Any:
    norm = (
        normalized_advantages if normalized_advantages is not None
        else _normalized_advantages(rows)
    )
    if len(norm) != len(rows):
        raise ValueError("PPO advantages and legal menus differ in size")
    losses = []
    for i, (menu, row) in enumerate(zip(menus, rows, strict=True)):
        logp = torch.nn.functional.log_softmax(menu / temperature, dim=0)
        picked = logp[row["chosen_index"]]
        ratio = (picked - row["old_logprob"]).clamp(-20.0, 20.0).exp()
        clipped = ratio.clamp(0.8, 1.2)
        actor = -torch.minimum(ratio * norm[i], clipped * norm[i])
        critic = (values[i] - row["target"]).square()
        entropy = -(logp.exp() * logp).sum()
        losses.append(actor + 0.5 * critic - 0.002 * entropy)
    return torch.stack(losses).mean()


def profile_minibatches(
    corpus: dict[str, Any], *, devices: list[str], batch_sizes: list[int],
    width: int = 64, trials: int = 5, warmups: int = 2,
    seed: int = 71,
) -> dict[str, Any]:
    """Measure actual PPO loss/grad/AdamW, plus synchronized forward phases."""
    import torch

    records = corpus["records"]
    normalized = _normalized_advantages(records)
    objects, bindings, baselines, preparation = prepare_records(records)
    vocabulary = ObjectVocabulary.from_object_sets(
        objects, manifest=corpus["manifest"],
    )
    oov = vocabulary.oov_report(objects)
    torch.manual_seed(seed)
    source = RelationalObjectTorch(vocabulary, width=width)
    initial = copy.deepcopy(source.module.state_dict())
    rng = random.Random(seed)
    max_batch = max(batch_sizes)
    # Same real examples and order on both devices and all trials.
    indices = (
        rng.sample(range(len(records)), max_batch)
        if len(records) >= max_batch
        else [rng.randrange(len(records)) for _ in range(max_batch)]
    )
    rows = []
    for device_name in devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        device = torch.device(device_name)

        def sync() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for size in batch_sizes:
            picked = indices[:size]
            selected_obj = [objects[i] for i in picked]
            selected_menu = [bindings[i] for i in picked]
            selected_base = [baselines[i] for i in picked]
            selected_raw = [records[i] for i in picked]
            selected_adv = [normalized[i] for i in picked]
            head = RelationalObjectTorch(vocabulary, width=width)
            head.module.load_state_dict(initial)
            head.module.to(device)
            optimizer = torch.optim.AdamW(head.module.parameters(), lr=0.0003)
            samples: list[dict[str, float]] = []
            peaks = []

            def step(instrument: bool) -> dict[str, float]:
                elapsed: dict[str, float] = {}
                sync()
                began = previous = time.perf_counter()

                def mark(label: str) -> None:
                    nonlocal previous
                    if not instrument:
                        return
                    sync()
                    now = time.perf_counter()
                    elapsed[label] = now - previous
                    previous = now

                optimizer.zero_grad(set_to_none=True)
                mark("zero_grad")
                values, logits = head.forward_batch_prepared(
                    selected_obj, selected_menu, selected_base,
                    stage_hook=mark if instrument else None,
                )
                loss = _ppo_loss(
                    torch, values, logits, selected_raw, corpus["temperature"],
                    normalized_advantages=selected_adv,
                )
                mark("ppo_loss")
                loss.backward()
                mark("backward")
                torch.nn.utils.clip_grad_norm_(head.module.parameters(), 1.0)
                mark("gradient_clip")
                optimizer.step()
                mark("adamw_step")
                sync()
                elapsed["total"] = time.perf_counter() - began
                return elapsed

            for _ in range(warmups):
                step(False)
            uninstrumented: list[float] = []
            for _ in range(trials):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                uninstrumented.append(step(False)["total"])
                if device.type == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated(device))
            for _ in range(trials):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                samples.append(step(True))
                if device.type == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated(device))
            keys = sorted(set().union(*(sample.keys() for sample in samples)))
            medians = {
                key: statistics.median(sample.get(key, 0.0) for sample in samples)
                for key in keys
            }
            result = {
                "device": device_name, "batch_size": size,
                "median_seconds": medians,
                "uninstrumented_total_seconds": statistics.median(uninstrumented),
                "uninstrumented_trials_seconds": uninstrumented,
                "decisions_per_second": size / statistics.median(uninstrumented),
                "instrumentation_overhead_ratio": (
                    medians["total"] / statistics.median(uninstrumented)
                ),
                "peak_cuda_allocated_bytes": max(peaks) if peaks else None,
                "tokens": {
                    "mean": statistics.fmean(len(obj.tokens) for obj in selected_obj),
                    "max": max(len(obj.tokens) for obj in selected_obj),
                },
                "legal_actions": {
                    "mean": statistics.fmean(len(menu) for menu in selected_menu),
                    "max": max(len(menu) for menu in selected_menu),
                },
                "reused_observations": len(set(picked)) != len(picked),
            }
            rows.append(result)
            print(
                f"[v15g] {device_name} batch={size} "
                f"ppo={statistics.median(uninstrumented) * 1000:.2f}ms "
                f"throughput={result['decisions_per_second']:.1f}/s "
                f"embed={medians.get('object_embedding', 0)*1000:.2f}ms "
                f"attention={medians.get('self_attention', 0)*1000:.2f}ms",
                flush=True,
            )
    return {
        "schema": REPORT_SCHEMA,
        "corpus_sha256": corpus["sha256"],
        "emulator_revision": corpus["emulator_revision"],
        "source_runs": corpus["source_runs"],
        "resolved_combats": corpus["resolved_combats"],
        "target_preparation_seconds": corpus.get("target_preparation_seconds"),
        "real_combat_decisions": len(records),
        "sample_preparation_seconds": preparation,
        "sample_preparation_per_decision_ms": {
            key: value * 1000 / len(records)
            for key, value in preparation.items()
        },
        "vocabulary_oov": oov,
        "torch_version": torch.__version__,
        "width": width, "trials": trials, "warmups": warmups,
        "rows": rows,
        "notes": (
            "Throughput uses uninstrumented wall timing. Synchronized phase " 
            "timing is reported separately and includes CUDA sync overhead. "
            "Only throwaway benchmark optimizer weights are updated; "
            "rollout collection and public PPO targets are reused verbatim. "
            "This is NOT gameplay win-rate or total training throughput."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--collect", action="store_true",
                        help="Collect and replace public corpus even if it exists")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"),
                        default=["cpu"])
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[32, 64, 128])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if (min(args.runs, args.max_decisions, args.threads, args.width,
            args.trials, *args.batch_sizes) <= 0
            or args.warmups < 0 or args.width % 4
            or not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error("Invalid sizes, warmups, width or temperature")

    import torch
    torch.set_num_threads(args.threads)
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.collect or not args.corpus.is_file():
        if args.base_model is None:
            parser.error("--base-model required when collecting a new corpus")
        base = PhaseSplitNeuralModel.load(args.base_model)
        if base.combat_value_objective != "hp_preservation":
            parser.error("Base critic must estimate HP preservation")
        episodes = []
        start = time.perf_counter()
        with JsonlEmulatorBackend(build=args.build) as backend:
            for i in range(args.runs):
                run_seed = f"v15g-real-ppo-{args.seed}-{i}"
                episode = collect_public_episode(
                    backend, base, seed=run_seed,
                    actor_rng=random.Random(episode_actor_seed(args.seed, run_seed)),
                    max_decisions=args.max_decisions,
                    environment=NATIVE_OVERGROWTH,
                    temperature=args.temperature,
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                )
                episodes.append(episode)
                print(
                    f"[v15g] collected run={i+1}/{args.runs} "
                    f"decisions={len(episode.decisions)} "
                    f"resolved_combats={len(episode.combat_outcomes)}",
                    flush=True,
                )
            corpus = build_corpus(
                episodes, base, dict(backend.capability_manifest),
                revision=backend.emulator_revision, seed=args.seed,
                temperature=args.temperature,
            )
        args.corpus.parent.mkdir(parents=True, exist_ok=True)
        args.corpus.write_text(
            json.dumps(corpus, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(
            f"[v15g] saved {len(corpus['records'])} real PPO samples "
            f"to {args.corpus}; collection_s={time.perf_counter()-start:.2f}",
            flush=True,
        )
    else:
        corpus = load_corpus(args.corpus)
        print(
            f"[v15g] loaded frozen public corpus {args.corpus} "
            f"samples={len(corpus['records'])}", flush=True,
        )
    if args.base_model is not None:
        base = PhaseSplitNeuralModel.load(args.base_model)
        if _digest(base.to_dict()) != corpus["base_sha256"]:
            parser.error("Corpus/base-model fingerprint mismatch")
    results = profile_minibatches(
        corpus, devices=args.devices, batch_sizes=args.batch_sizes,
        width=args.width, warmups=args.warmups, trials=args.trials, seed=args.seed,
    )
    results["corpus_path"] = str(args.corpus)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[v15g] saved benchmark report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
