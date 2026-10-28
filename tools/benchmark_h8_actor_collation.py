"""Microbenchmark exact H7 actor host collation + device transfer.

For a reproducible real-combat end-to-end comparison, also run
benchmark_h8_h7_policy_pilot.py with --actor-collation packed on the
identical 1,024-source checkpoint and emulator revision.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sts2_ai.models.packed_actor_collation import collate_h7_actor_packed
from sts2_ai.models.shared_game_objects import (
    NUMERIC_FIELDS, SHARED_OBJECT_SCHEMA, ActionToken, EncodedDecision, ObjectToken,
)
from sts2_ai.models.shared_transformer import collate_h7, vars_as_dict


def make_decisions(batch: int, objects: int, actions: int) -> list[EncodedDecision]:
    numbers = len(NUMERIC_FIELDS)
    result = []
    for row in range(batch):
        nobj = max(2, objects - row % 5)
        nact = max(1, actions - row % 3)
        tokens = tuple(ObjectToken(
            kind=i % 8,
            identity=(i * 13 + row) % 4096,
            zone=i % 12,
            numeric=tuple((i - j) / 30.0 for j in range(numbers)),
            observed=tuple((i + j) % 2 for j in range(numbers)),
            owner=(i - 1 if i > 0 else -1),
        ) for i in range(nobj))
        acts = tuple(ActionToken(
            kind=(j * 19) % 2048,
            semantic=(j + row) % 4096,
            source=(j % (nobj - 1) + 1) if j % 2 else -1,
            target=(nobj - 1 if j % 3 == 0 else -1),
        ) for j in range(nact))
        result.append(EncodedDecision(
            SHARED_OBJECT_SCHEMA, "combat", tokens, acts))
    return result


def measure(fn, *, iterations: int, device: str) -> list[float]:
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        batch = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        del batch
    return timings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--objects", type=int, default=40)
    ap.add_argument("--actions", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if min(args.batch, args.objects, args.actions, args.iterations) < 1 or args.warmup < 0:
        ap.error("Positive batch/object/action/iteration sizes required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA requested without a CUDA device")
    torch.set_num_threads(1)
    inputs = make_decisions(args.batch, args.objects, args.actions)
    old_fn = lambda: collate_h7(inputs).to(args.device)
    new_fn = lambda: collate_h7_actor_packed(inputs, args.device)
    old = old_fn()
    new = new_fn()
    for key, expected in vars_as_dict(old).items():
        if not torch.equal(expected, getattr(new, key)):
            raise RuntimeError("Packed tensor mismatch: " + key)
    for _ in range(args.warmup):
        old_fn()
        new_fn()
    # Interleave opposite order to control transient host/device load.
    measurements = {"legacy": [], "packed": []}
    for ordinal, name in enumerate(("legacy", "packed", "packed", "legacy")):
        fn = old_fn if name == "legacy" else new_fn
        measurements[name].extend(measure(
            fn, iterations=args.iterations, device=args.device))
    medians = {name: statistics.median(values)
               for name, values in measurements.items()}
    result = {
        "schema": "sts2-h8-actor-collation-microbench-v1",
        "device": args.device, "batch": args.batch,
        "objects": args.objects, "actions": args.actions,
        "iterations_per_leg": args.iterations,
        "tensor_parity": True, "order": ["legacy", "packed", "packed", "legacy"],
        "median_ms": {name: v * 1000.0 for name, v in medians.items()},
        "speedup": medians["legacy"] / medians["packed"],
        "interpretation": (
            "Isolated tensor collation and H2D transfer; includes CUDA "
            "synchronization per batch. Not a full combat rollout speedup."),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
