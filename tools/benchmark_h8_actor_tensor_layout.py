"""Isolate packing versus metadata strides for the frozen H7/H8 Transformer.

Perform a three-way A-B-C-C-B-A test: legacy tensor construction, packed
interleaved metadata, and packed metadata made contiguous *after* host-to-device
transfer. Measure collation and forward/output stages separately, using
synchronized timings on CUDA. This is a synthetic batch microbenchmark:
only the real 1,024-combat source-valid pilot can establish rollout gains.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from benchmark_h8_actor_collation import make_decisions
from sts2_ai.models.packed_actor_collation import collate_h7_actor_packed
from sts2_ai.models.shared_transformer import (
    H7Config, SharedObjectTransformer, collate_h7, vars_as_dict,
)

MODES = ("legacy", "packed", "packed-contiguous")
ORDER = ("legacy", "packed", "packed-contiguous",
         "packed-contiguous", "packed", "legacy")


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path,
                    help="Optional frozen real H7 Transformer checkpoint")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--objects", type=int, default=40)
    ap.add_argument("--actions", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=100,
                    help="Iterations for each of the six alternating legs")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--feedforward", type=int, default=1536)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if (min(args.batch, args.objects, args.actions, args.iterations,
            args.width, args.heads, args.layers, args.feedforward) < 1
            or args.warmup < 0):
        ap.error("Positive dimensions and iterations required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA requested without CUDA")
    torch.set_num_threads(1)
    inputs = make_decisions(args.batch, args.objects, args.actions)
    funcs = {
        "legacy": lambda: collate_h7(inputs).to(args.device),
        "packed": lambda: collate_h7_actor_packed(inputs, args.device),
        "packed-contiguous": lambda: collate_h7_actor_packed(
            inputs, args.device, contiguous_metadata=True),
    }
    if args.checkpoint:
        model = SharedObjectTransformer.load_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(73127)
        model = SharedObjectTransformer(H7Config(
            width=args.width, heads=args.heads, layers=args.layers,
            feedforward=args.feedforward,
        ))
    model = model.to(args.device).eval()

    def evaluate(mode: str):
        t0 = time.perf_counter()
        batch = funcs[mode]()
        sync(args.device)
        collate = time.perf_counter() - t0
        t1 = time.perf_counter()
        logits, values = model(batch)
        sync(args.device)
        logits = logits.detach().float().cpu()
        values = values.detach().float().cpu()
        sync(args.device)
        forward = time.perf_counter() - t1
        return collate, forward, logits, values, batch

    with torch.inference_mode():
        expected_fields = vars_as_dict(funcs["legacy"]())
        for mode in MODES:
            got = funcs[mode]()
            for name, expected in expected_fields.items():
                actual = getattr(got, name)
                if not torch.equal(expected, actual):
                    raise RuntimeError(f"Input parity failed: {mode} / {name}")
        # All output differences must be below the fixed tolerance, even
        # though CUDA kernels can select different reduction schedules.
        ref = evaluate("legacy")
        diffs = {}
        for mode in MODES:
            _, _, logits, values, _ = evaluate(mode)
            tolerances = dict(atol=1e-5, rtol=1e-5) if args.device.startswith("cuda") else dict(
                atol=0.0, rtol=0.0)
            torch.testing.assert_close(logits, ref[2], **tolerances)
            torch.testing.assert_close(values, ref[3], **tolerances)
            diffs[mode] = max(
                float((logits - ref[2]).abs().max()),
                float((values - ref[3]).abs().max()),
            )
        for _ in range(args.warmup):
            for mode in MODES:
                evaluate(mode)
        measurements = {mode: {"collate": [], "forward": [], "total": []}
                        for mode in MODES}
        leg_medians = []
        for mode in ORDER:
            c, f, whole = [], [], []
            for _ in range(args.iterations):
                col, fw, *_ = evaluate(mode)
                c.append(col)
                f.append(fw)
                whole.append(col + fw)
            measurements[mode]["collate"].extend(c)
            measurements[mode]["forward"].extend(f)
            measurements[mode]["total"].extend(whole)
            leg_medians.append({
                "mode": mode,
                "collate_ms": statistics.median(c) * 1000,
                "forward_ms": statistics.median(f) * 1000,
                "total_ms": statistics.median(whole) * 1000,
            })
    medians = {
        mode: {k + "_ms": statistics.median(v) * 1000
               for k, v in records.items()}
        for mode, records in measurements.items()
    }
    result = {
        "schema": "sts2-h8-three-way-tensor-layout-v1",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_config": vars(model.config),
        "device": args.device,
        "torch_version": torch.__version__,
        "batch": args.batch,
        "objects": args.objects,
        "actions": args.actions,
        "iterations_per_leg": args.iterations,
        "warmup_per_mode": args.warmup,
        "order": list(ORDER),
        "input_parity": True,
        "model_output_parity": True,
        "max_abs_model_output_diff": diffs,
        "median_ms": medians,
        "leg_medians": leg_medians,
        "contiguous_vs_packed_total_speedup": (
            medians["packed"]["total_ms"] /
            medians["packed-contiguous"]["total_ms"]),
        "contiguous_vs_packed_forward_speedup": (
            medians["packed"]["forward_ms"] /
            medians["packed-contiguous"]["forward_ms"]),
        "notes": (
            "Per-batch collation and forward/output transfer, CUDA "
            "synchronized. Different tensor strides may select different "
            "GPU kernels. Use full combat ABCCBA to confirm any gain."),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
