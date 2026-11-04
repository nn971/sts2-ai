"""Opt-in H16 actor service: batched public-token inference on one GPU.

The emulator and all player-visible encoding remain in worker processes.
Each request carries one H16-encoded public decision. One frozen H16
checkpoint and one horizon lambda are shared across a collection cohort.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sts2_ai.models.packed_actor_collation import prepare_h7_actor_packed
from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet


@dataclass
class H16ActorTransport:
    requests: Any
    replies: tuple[Any, ...]
    slot_counter: Any
    profile_actor: bool = False

    def allocate_slot(self) -> int:
        with self.slot_counter.get_lock():
            slot = int(self.slot_counter.value)
            if slot >= len(self.replies):
                raise RuntimeError("H16 shared actor worker capacity exceeded")
            self.slot_counter.value += 1
        return slot


class RemoteH16Actor:
    """Worker-owned synchronous client; at most one outstanding request."""

    def __init__(self, transport: H16ActorTransport, *, timeout: float = 120.0):
        self.transport = transport
        self.slot = transport.allocate_slot()
        self.timeout = timeout
        self.sequence = 0
        self.wait_seconds = 0.0

    def infer(self, encoded: EncodedDecision) -> tuple[tuple[float, ...], float]:
        self.sequence += 1
        seq = self.sequence
        started = time.perf_counter()
        try:
            request = (
                (self.slot, seq, encoded, started)
                if self.transport.profile_actor else (self.slot, seq, encoded)
            )
            self.transport.requests.put(request, timeout=self.timeout)
            answer = self.transport.replies[self.slot].get(timeout=self.timeout)
        except (queue.Full, queue.Empty) as exc:
            raise RuntimeError("H16 shared actor request/reply stalled") from exc
        finally:
            self.wait_seconds += time.perf_counter() - started
        returned, status, payload = answer
        if returned != seq:
            raise RuntimeError("H16 shared actor reply sequence mismatch")
        if status != "ok":
            raise RuntimeError(f"H16 shared actor inference failed: {payload}")
        scores, value = payload
        if (len(scores) != len(encoded.actions)
                or not all(math.isfinite(x) for x in (*scores, value))):
            raise RuntimeError("H16 shared actor returned invalid output")
        return tuple(scores), float(value)


def _serve(
    checkpoint: str, checkpoint_sha: str, device: str,
    horizon_lambda: float, requests: Any, replies: tuple[Any, ...],
    startup: Any, statistics: Any,
    max_batch: int, max_delay_ms: float,
    profile_actor: bool = False,
    verify_batch_parity: bool = False,
) -> None:
    stats: dict[str, Any] = {
        "device": device, "requests": 0, "batches": 0,
        "max_batch_observed": 0, "model_load_seconds": 0.0,
        "host_collate_seconds": 0.0, "transfer_seconds": 0.0,
        "forward_seconds": 0.0,
    }
    # P2-only bounded samples; disabled in production H16 execution.
    samples: dict[str, list[float]] = {
        "actor_queue_delay_ms": [], "host_collate_ms": [],
        "host_to_device_ms": [], "forward_sync_ms": [],
    }
    batch_histogram: dict[str, int] = {}
    # Explicit P2 diagnostic: the same decision evaluated alone versus in
    # the original variable-shaped mixed batch. Only recorded when requested;
    # extra GPU forwards invalidate this run as a throughput measurement.
    parity: dict[str, Any] = {
        "schema": "sts2-p2-actor-batch-numerical-parity-v1",
        "samples": 0,
        "changed_logits": 0,
        "changed_value": 0,
        "changed_argmax": 0,
        "max_abs_logit_difference": 0.0,
        "max_abs_value_difference": 0.0,
        "examples": [],
        "limit": 256,
    }

    def observe(name: str, seconds: float) -> None:
        if profile_actor and len(samples[name]) < 8192:
            samples[name].append(seconds * 1000.0)

    def distribution(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        sorted_values = sorted(values)
        return {
            "samples": len(sorted_values),
            "p50_ms": sorted_values[int((len(sorted_values) - 1) * 0.50)],
            "p95_ms": sorted_values[int((len(sorted_values) - 1) * 0.95)],
            "max_ms": sorted_values[-1],
        }

    try:
        torch.set_num_threads(1)
        started = time.perf_counter()
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != checkpoint_sha:
            raise ValueError("H16 shared actor checkpoint SHA256 mismatch")
        model = JointActPilotNet.load(Path(checkpoint), device=device).eval()
        stats["model_load_seconds"] = time.perf_counter() - started
        startup.put(("ready", None))
        stopping = False
        with torch.inference_mode():
            while not stopping:
                first = requests.get()
                if first is None:
                    break
                work = [first]
                deadline = time.perf_counter() + max_delay_ms / 1000.0
                while len(work) < max_batch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        entry = requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if entry is None:
                        stopping = True
                        break
                    work.append(entry)
                try:
                    frames = [entry[2] for entry in work]
                    if any(not isinstance(item, EncodedDecision) for item in frames):
                        raise ValueError("H16 actor expects public EncodedDecision tokens")
                    started = time.perf_counter()
                    if profile_actor:
                        ready_at = started
                        for entry in work:
                            if len(entry) == 4:
                                observe("actor_queue_delay_ms", ready_at - entry[3])
                        count_key = str(len(work))
                        batch_histogram[count_key] = batch_histogram.get(count_key, 0) + 1
                    host = prepare_h7_actor_packed(frames)
                    host_elapsed = time.perf_counter() - started
                    stats["host_collate_seconds"] += host_elapsed
                    observe("host_collate_ms", host_elapsed)
                    started = time.perf_counter()
                    batch = host.to(device)
                    if profile_actor and device.startswith("cuda"):
                        # Synchronize ONLY in the explicitly profiled arm to
                        # separate transfer from model+readback latency.
                        torch.cuda.synchronize(device)
                    transfer_elapsed = time.perf_counter() - started
                    stats["transfer_seconds"] += transfer_elapsed
                    observe("host_to_device_ms", transfer_elapsed)
                    started = time.perf_counter()
                    logits, _, hp = model(batch, horizon_lambda=horizon_lambda)
                    # Read back once per batch to synchronize CUDA.
                    cpu_logits = logits.detach().float().cpu()
                    cpu_hp = hp.detach().float().cpu()
                    forward_elapsed = time.perf_counter() - started
                    stats["forward_seconds"] += forward_elapsed
                    observe("forward_sync_ms", forward_elapsed)
                    if verify_batch_parity and len(work) > 1 and (
                        parity["samples"] < parity["limit"]
                    ):
                        # Use the same frozen model/encoded tokens on the
                        # same GPU. Differing batch shape and padding are
                        # the only intended variables.
                        sample = frames[0]
                        single = prepare_h7_actor_packed([sample]).to(device)
                        single_logits, _, single_hp = model(
                            single, horizon_lambda=horizon_lambda,
                        )
                        count = len(sample.actions)
                        reference_logits = (
                            single_logits[0, :count].detach().float().cpu()
                        )
                        reference_hp = float(single_hp[0].detach().float().cpu())
                        mixed_logits = cpu_logits[0, :count]
                        mixed_hp = float(cpu_hp[0])
                        logit_delta = float(
                            torch.max(torch.abs(reference_logits - mixed_logits))
                        )
                        value_delta = abs(reference_hp - mixed_hp)
                        argmax_changed = (
                            int(torch.argmax(reference_logits))
                            != int(torch.argmax(mixed_logits))
                        )
                        parity["samples"] += 1
                        parity["changed_logits"] += int(logit_delta != 0)
                        parity["changed_value"] += int(value_delta != 0)
                        parity["changed_argmax"] += int(argmax_changed)
                        parity["max_abs_logit_difference"] = max(
                            parity["max_abs_logit_difference"], logit_delta,
                        )
                        parity["max_abs_value_difference"] = max(
                            parity["max_abs_value_difference"], value_delta,
                        )
                        if len(parity["examples"]) < 12 and (
                            logit_delta or value_delta
                        ):
                            parity["examples"].append({
                                "batch_size": len(frames),
                                "objects": len(sample.objects),
                                "actions": count,
                                "batch_objects": int(batch.kinds.shape[1]),
                                "batch_actions": int(batch.action_kinds.shape[1]),
                                "max_abs_logit_difference": logit_delta,
                                "abs_value_difference": value_delta,
                                "argmax_changed": argmax_changed,
                            })
                    stats["batches"] += 1
                    stats["requests"] += len(work)
                    stats["max_batch_observed"] = max(
                        stats["max_batch_observed"], len(work),
                    )
                    for row, entry in enumerate(work):
                        slot, seq, encoded = entry[:3]
                        count = len(encoded.actions)
                        replies[slot].put((
                            seq, "ok",
                            (tuple(cpu_logits[row, :count].tolist()),
                             float(cpu_hp[row])),
                        ))
                except Exception:
                    error = traceback.format_exc()
                    for entry in work:
                        slot, seq = entry[:2]
                        replies[slot].put((seq, "error", error))
        if verify_batch_parity:
            stats["p2_batch_numerical_parity"] = parity
        if profile_actor:
            stats["p2_actor_distributions"] = {
                key: distribution(values) for key, values in samples.items()
            }
            stats["p2_actor_sample_cap"] = 8192
            stats["p2_batch_size_histogram"] = batch_histogram
            if device.startswith("cuda"):
                stats["p2_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                stats["p2_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        statistics.put(("ok", stats))
    except BaseException:
        error = traceback.format_exc()
        try:
            startup.put(("error", error))
        except Exception:
            pass
        statistics.put(("error", error))


class SharedH16Service:
    """A cohort-scoped frozen actor process and worker-specific response queues."""

    def __init__(
        self, *, checkpoint: Path, digest: str, workers: int,
        device: str, horizon_lambda: float,
        max_batch: int = 16, max_delay_ms: float = 2.0,
        profile_actor: bool = False,
        verify_batch_parity: bool = False,
    ) -> None:
        if (workers < 1 or max_batch < 1 or max_delay_ms < 0
                or not math.isfinite(max_delay_ms)
                or not math.isfinite(horizon_lambda)
                or not 0 <= horizon_lambda <= 1
                or not (device == "cpu" or device.startswith("cuda"))
                or (verify_batch_parity and not profile_actor)):
            raise ValueError("Invalid H16 shared actor configuration")
        ctx = multiprocessing.get_context("spawn")
        self.requests = ctx.Queue(maxsize=workers * 4)
        self.replies = tuple(ctx.Queue(maxsize=2) for _ in range(workers))
        self.startup = ctx.Queue(maxsize=2)
        self.statistics = ctx.Queue(maxsize=2)
        self.transport = H16ActorTransport(
            self.requests, self.replies, ctx.Value("i", 0), profile_actor,
        )
        self.process = ctx.Process(
            target=_serve,
            args=(str(checkpoint), digest, device, horizon_lambda,
                  self.requests, self.replies, self.startup, self.statistics,
                  max_batch, max_delay_ms, profile_actor,
                  verify_batch_parity),
            daemon=True,
        )
        self.metrics: dict[str, Any] = {}

    def __enter__(self) -> SharedH16Service:
        self.process.start()
        try:
            status, payload = self.startup.get(timeout=120)
        except queue.Empty as exc:
            self.process.terminate()
            self.process.join(timeout=10)
            raise RuntimeError("H16 shared actor process failed to start") from exc
        if status != "ready":
            self.process.join(timeout=10)
            raise RuntimeError(f"H16 shared actor startup failed: {payload}")
        return self

    def __exit__(self, *_: object) -> None:
        try:
            self.requests.put(None, timeout=5)
            try:
                status, report = self.statistics.get(timeout=60)
            except queue.Empty as exc:
                raise RuntimeError("H16 shared actor produced no statistics") from exc
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
                raise RuntimeError("H16 shared actor did not exit cleanly")
            if self.process.exitcode != 0 or status != "ok":
                raise RuntimeError(f"H16 shared actor failed: {report}")
            self.metrics = dict(report)
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
            self.requests.close()
            for reply in self.replies:
                reply.close()
            self.startup.close()
            self.statistics.close()
