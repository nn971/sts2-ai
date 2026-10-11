#!/usr/bin/env python3
"""Benchmark v15f batched vs scalar relational attention; NO agent training.

Synthetic *public-structure-shaped* object observations isolate network cost.
This does not estimate game strength or emulator/JSONL throughput. Metrics
include forward-only and forward+backward time for several variable-size
decision cohorts. CUDA peak *allocated* tensor memory is reported when used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import (
    ActionBinding,
    ObjectRef,
    ObjectSet,
    ObjectToken,
    OBJECT_SCHEMA,
)
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.models.relational_objects import RelationalObjectTorch


def _batch(size: int, seed: int):
    rng = random.Random(seed)
    objects = []
    menus = []
    baselines = []
    for i in range(size):
        tokens = [
            ObjectToken(
                "global", "player", "combat", ObjectRef("global", "player"),
                None, (("hp", float(30 + i % 30)), ("energy", 3.0)), (),
            )
        ]
        cards, enemies = [], []
        for j in range(rng.randint(4, 12)):
            ref = ObjectRef("card", i * 1000 + j + 1)
            cards.append(ref)
            tokens.append(ObjectToken(
                "card", f"proto.card.{j % 8}", "hand", ref, None,
                (("upgrade_level", float(j % 3)),), (),
            ))
        for j in range(rng.randint(1, 5)):
            ref = ObjectRef("enemy", i * 1000 + j + 1)
            enemies.append(ref)
            tokens.append(ObjectToken(
                "enemy", f"proto.enemy.{j % 4}", "living", ref, None,
                (("hp", float(8 + j * 7)), ("block", float(j % 4))), (),
            ))
            for p in range(j % 3):
                ref_power = ObjectRef("power", (str(ref), "proto.power.strangle", p))
                tokens.append(ObjectToken(
                    "power", "proto.power.strangle", "active", ref_power, ref,
                    (("stacks", float(1 + p)),), (),
                ))
        if i % 2 == 0:
            tokens.append(ObjectToken(
                "relic", "proto.relic.demo", "owned",
                ObjectRef("relic", "proto.relic.demo"), None, (), (),
            ))
        tokens.append(ObjectToken(
            "potion", "proto.potion.demo", "held",
            ObjectRef("potion", 0), None, (), (),
        ))
        objects.append(ObjectSet(OBJECT_SCHEMA, tuple(tokens)))
        actions = []
        for j in range(rng.randint(3, 15)):
            mode = j % 5
            if mode == 0:
                actions.append(("end_turn", ()))
            elif mode == 1:
                actions.append(("select_cards", ()))
            elif mode == 2:
                actions.append((
                    "select_cards", tuple(
                        ActionBinding("selected_card", ref)
                        for ref in rng.sample(cards, k=min(len(cards), 2 + (j % 3)))
                    ),
                ))
            elif mode == 3:
                actions.append((
                    "play_card", (
                        ActionBinding("source_card", rng.choice(cards)),
                        ActionBinding("target_enemy", rng.choice(enemies)),
                    ),
                ))
            else:
                actions.append((
                    "use_potion", (
                        ActionBinding("source_potion", ObjectRef("potion", 0)),
                        ActionBinding("target_enemy", rng.choice(enemies)),
                    ),
                ))
        menus.append(tuple(actions))
        baselines.append(
            PolicyValueEstimate(
                tuple(0.02 * j for j in range(len(actions))),
                0.1 + (i % 5) * 0.05,
            )
        )
    return objects, menus, baselines


def _timer(head, objects, menus, baselines, *, batched, train, trials, warmups):
    torch = head.torch
    device = head.module.policy_gate.device

    def one_run():
        head.module.zero_grad(set_to_none=True)
        if batched:
            values, logits = head.forward_batch_prepared(objects, menus, baselines)
            loss = (values.square().sum() + sum(
                lg.square().mean() for lg in logits
            )) / len(objects)
        else:
            predictions = [
                head.forward_prepared(obj, actions, base)
                for obj, actions, base in zip(
                    objects, menus, baselines, strict=True,
                )
            ]
            loss = sum(
                val.square() + lg.square().mean()
                for val, lg in predictions
            ) / len(objects)
        if train:
            loss.backward()
        return loss

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmups):
        one_run()
        synchronize()
    measurements = []
    peak = None
    for _ in range(trials):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize()
        start = time.perf_counter()
        one_run()
        synchronize()
        measurements.append(time.perf_counter() - start)
        if device.type == "cuda":
            used = torch.cuda.max_memory_allocated(device)
            peak = max(used, peak or 0)
    return {
        "median_seconds": statistics.median(measurements),
        "decisions_per_second": len(objects) / statistics.median(measurements),
        "all_seconds": measurements,
        "peak_cuda_allocated_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--report", type=Path,
                        default=Path("results/v15f-attention-benchmark.json"))
    args = parser.parse_args()
    if (min(*args.batch_sizes, args.trials, args.threads, args.width) <= 0
            or args.warmups < 0 or args.width % 4):
        parser.error("Invalid batch size, trials, threads, warmups or attention width")
    import torch

    torch.set_num_threads(args.threads)
    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    if "cuda" in devices and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    max_size = max(args.batch_sizes)
    all_objects, all_actions, all_baselines = _batch(max_size, args.seed)
    vocab = ObjectVocabulary.from_object_sets(all_objects)
    rows = []
    for device in devices:
        torch.manual_seed(args.seed)
        head = RelationalObjectTorch(vocab, width=args.width)
        head.module.to(device)
        with torch.no_grad():
            head.module.policy_gate.fill_(0.42)
            head.module.value_gate.fill_(0.31)
        for size in args.batch_sizes:
            for train in (False, True):
                objects = all_objects[:size]
                actions = all_actions[:size]
                baselines = all_baselines[:size]
                reference = _timer(
                    head, objects, actions, baselines, batched=False,
                    train=train, trials=args.trials, warmups=args.warmups,
                )
                batched = _timer(
                    head, objects, actions, baselines, batched=True,
                    train=train, trials=args.trials, warmups=args.warmups,
                )
                row = {
                    "device": device, "batch_size": size,
                    "phase": "forward_backward" if train else "forward",
                    "reference": reference, "batched": batched,
                    "speedup": (
                        reference["median_seconds"] / batched["median_seconds"]
                    ),
                }
                rows.append(row)
                print(
                    f"[v15f] {device} {row['phase']} batch={size} "
                    f"scalar={reference['median_seconds']:.3f}s "
                    f"batched={batched['median_seconds']:.3f}s "
                    f"speedup={row['speedup']:.2f}x",
                    flush=True,
                )
    result = {
        "schema": "sts2-v15f-relational-batch-benchmark-v1",
        "synthetic_public_structure": True, "seed": args.seed,
        "threads": args.threads, "width": args.width,
        "torch_version": torch.__version__, "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"[v15f] saved report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
