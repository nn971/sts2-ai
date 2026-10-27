"""Opt-in single-process batched H7 actor inference for semantic emulator workers.

Workers send only source-public EncodedDecision tokens. The server owns a
SHA-pinned frozen checkpoint for the cohort and has no emulator/hidden state.
The existing FrozenP0Player retains source-seeded sampling and PPO capture.
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

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.protocol import PolicyValueEstimate

from .shared_game_objects import EncodedDecision, encode_public_decision
from .shared_transformer import SharedObjectTransformer, collate_h7


@dataclass
class H7Transport:
    requests: Any
    replies: tuple[Any, ...]
    slot_counter: Any
    model_id: str

    def allocate_slot(self) -> int:
        with self.slot_counter.get_lock():
            slot = int(self.slot_counter.value)
            if slot >= len(self.replies):
                raise RuntimeError("H7 shared inference worker capacity exceeded")
            self.slot_counter.value += 1
        return slot


class RemoteH7Policy:
    """PublicPolicy adapter; each worker has exactly one outstanding request."""

    def __init__(self, transport: H7Transport, *, slot: int, timeout: float = 120.0):
        self.transport = transport
        self.slot = slot
        self.timeout = timeout
        self.model_id = transport.model_id
        self.sequence = 0
        self.calls = 0
        self.encoding_seconds = 0.0
        self.wait_seconds = 0.0

    def timings_snapshot(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "encoding_seconds": self.encoding_seconds,
            "wait_seconds": self.wait_seconds,
        }

    def evaluate(
        self, observation: Observation, legal_actions: tuple[LegalAction, ...],
    ) -> PolicyValueEstimate:
        start = time.perf_counter()
        import json
        frame = json.loads(observation.payload_json)
        if not isinstance(frame, dict):
            raise ValueError("Malformed H7 public observation")
        encoded = encode_public_decision(frame, legal_actions)
        self.encoding_seconds += time.perf_counter() - start
        self.sequence += 1
        seq = self.sequence
        start = time.perf_counter()
        try:
            self.transport.requests.put((self.slot, seq, encoded), timeout=self.timeout)
            answer = self.transport.replies[self.slot].get(timeout=self.timeout)
        except queue.Full as exc:
            raise RuntimeError("H7 shared inference request queue stalled") from exc
        except queue.Empty as exc:
            raise RuntimeError("H7 shared inference reply timed out") from exc
        finally:
            self.wait_seconds += time.perf_counter() - start
        response_seq, status, payload = answer
        if response_seq != seq:
            raise RuntimeError("H7 shared inference response correlation mismatch")
        if status != "ok":
            raise RuntimeError(f"H7 shared inference error: {payload}")
        logits, value = payload
        if (len(logits) != len(legal_actions)
                or not all(math.isfinite(x) for x in (*logits, value))):
            raise RuntimeError("H7 shared inference returned invalid action logits")
        self.calls += 1
        return PolicyValueEstimate(action_logits=tuple(logits), value=float(value))


def _serve(
    checkpoint: str, digest: str, device: str, requests: Any,
    replies: tuple[Any, ...], startup: Any, statistics: Any,
    max_batch: int, max_delay_ms: float,
) -> None:
    stats: dict[str, float | int | str] = {
        "batches": 0, "requests": 0, "forward_seconds": 0.0,
        "model_load_seconds": 0.0, "max_batch_observed": 0,
        "device": device,
    }
    try:
        torch.set_num_threads(1)
        loaded = time.perf_counter()
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != digest:
            raise ValueError("H7 shared actor checkpoint SHA256 mismatch")
        model = SharedObjectTransformer.load_checkpoint(Path(checkpoint))
        model = model.to(device).eval()
        stats["model_load_seconds"] = time.perf_counter() - loaded
        startup.put(("ready", None))
        stopping = False
        with torch.inference_mode():
            while not stopping:
                request = requests.get()
                if request is None:
                    break
                work = [request]
                deadline = time.perf_counter() + max_delay_ms / 1000.0
                while len(work) < max_batch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        more = requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if more is None:
                        stopping = True
                        break
                    work.append(more)
                try:
                    frames = [entry[2] for entry in work]
                    if not all(isinstance(item, EncodedDecision) for item in frames):
                        raise ValueError("H7 inference received non-public token request")
                    batch = collate_h7(frames).to(device)
                    begun = time.perf_counter()
                    logits, values = model(batch)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize(device)
                    cpu_logits = logits.detach().float().cpu()
                    cpu_values = values.detach().float().cpu()
                    stats["forward_seconds"] = float(stats["forward_seconds"]) + (
                        time.perf_counter() - begun
                    )
                    stats["batches"] = int(stats["batches"]) + 1
                    stats["requests"] = int(stats["requests"]) + len(work)
                    stats["max_batch_observed"] = max(
                        int(stats["max_batch_observed"]), len(work)
                    )
                    for row, (slot, seq, frame) in enumerate(work):
                        count = len(frame.actions)
                        replies[slot].put((
                            seq, "ok",
                            (tuple(cpu_logits[row, :count].tolist()),
                             float(cpu_values[row].item())),
                        ))
                except Exception:
                    error = traceback.format_exc()
                    for slot, seq, _ in work:
                        replies[slot].put((seq, "error", error))
        statistics.put(("ok", stats))
    except BaseException:
        error = traceback.format_exc()
        try:
            startup.put(("error", error))
        except Exception:
            pass
        statistics.put(("error", error))


class SharedH7Service:
    """Owns one spawn-safe Torch process and a queue per emulator worker."""

    def __init__(
        self, *, checkpoint: Path, digest: str, workers: int,
        device: str = "cpu", max_batch: int = 16, max_delay_ms: float = 3.0,
    ) -> None:
        if (workers < 1 or max_batch < 1 or max_delay_ms < 0
                or not math.isfinite(max_delay_ms)
                or not (device == "cpu" or device.startswith("cuda"))):
            raise ValueError("Invalid H7 shared inference configuration")
        self.ctx = multiprocessing.get_context("spawn")
        self.requests = self.ctx.Queue(maxsize=workers * 4)
        self.replies = tuple(self.ctx.Queue(maxsize=2) for _ in range(workers))
        self.startup = self.ctx.Queue(maxsize=2)
        self.statistics = self.ctx.Queue(maxsize=2)
        self.transport = H7Transport(
            self.requests, self.replies, self.ctx.Value("i", 0),
            model_id=f"h7-{digest[:16]}",
        )
        self.process = self.ctx.Process(
            target=_serve,
            args=(
                str(checkpoint), digest, device,
                self.requests, self.replies, self.startup, self.statistics,
                max_batch, max_delay_ms,
            ),
            daemon=True,
        )
        self.metrics: dict[str, Any] = {}

    def __enter__(self) -> SharedH7Service:
        self.process.start()
        try:
            status, payload = self.startup.get(timeout=120)
        except queue.Empty as exc:
            self.process.terminate()
            self.process.join(timeout=10)
            raise RuntimeError("H7 inference process failed to start") from exc
        if status != "ready":
            self.process.join(timeout=10)
            raise RuntimeError(f"H7 inference process initialization failed: {payload}")
        return self

    def __exit__(self, *_: object) -> None:
        try:
            self.requests.put(None, timeout=5)
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
                raise RuntimeError("H7 shared inference server did not exit cleanly")
            if self.process.exitcode != 0:
                raise RuntimeError(f"H7 shared inference exited with {self.process.exitcode}")
            try:
                status, report = self.statistics.get(timeout=5)
            except queue.Empty as exc:
                raise RuntimeError("H7 inference server produced no statistics") from exc
            if status != "ok":
                raise RuntimeError(f"H7 inference server failed: {report}")
            self.metrics = dict(report)
        finally:
            self.requests.close()
            for response in self.replies:
                response.close()
            self.startup.close()
            self.statistics.close()
