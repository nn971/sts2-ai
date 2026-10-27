# H7 shared batched actor inference — experiment (2026-10-27)

This branch introduces an **opt-in** inference backend for the H7 14.7M
shared-object Transformer. Local per-worker inference remains the default.

## Evidence and expected benefit

The uploaded 128-source H7 A10 benchmarks with identical model, seed bank,
and emulator binary showed 14.68 player decisions/s with 4 workers and
18.59 decisions/s with 15 workers (1.27x). This suggests resource contention
or expensive repeated inference but does not identify the bottleneck.

The shared backend keeps 15 independent external C# emulator workers and their
per-source RNG samplers, but moves **only the frozen Transformer inference**
to one spawned Python Torch service. Each worker encodes its source-public
Observation and legal-action menu into `EncodedDecision` and sends those
tokens over a bounded request queue. The service batches variable-size
objects and action menus, returns padded-mask-filtered logits plus value on a
worker-specific response queue, and never observes hidden deck order, Nature
choices or internal emulator state. No distillation, MCTS or trained Nature.

The service pins/checks the checkpoint SHA-256, identifies actor revisions
by model digest, bounds the batching delay and backlog, and reports batch
size, forward time and loading time. The worker retains the existing
`FrozenP0Player` seeded action sampler, full public-menu provenance and
behavior value/log-probability capture used by PPO.

## Replayable benchmark commands (fish)

Start from the PR #116 branch. Use the built H5-compatible emulator at
`e56779ccd86725582af580925ca8242826ba2229`, not the H6 macro gitlink.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h7-shared-actor-inference-20261027
source .venv/bin/activate.fish

set rev e56779ccd86725582af580925ca8242826ba2229
set h5 /tmp/sts2-h5-emulator.zIOUjm
set dll $h5/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings $h5/bindings/python

python -m pytest -q tests/test_h7_batch_actor.py tests/test_h7_readiness.py

# Existing direct CPU path, now with per-episode actor timings.
python -u tools/benchmark_h7_rollouts.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 --episodes 128 \
  --actor-backend local --output /tmp/h7-inference-local-128.json

# One central CPU inference process with dynamic batches.
python -u tools/benchmark_h7_rollouts.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 --episodes 128 \
  --actor-backend shared --actor-device cpu \
  --actor-max-batch 16 --actor-delay-ms 3 \
  --output /tmp/h7-inference-shared-cpu-128.json

# Optional: only if Python Torch has CUDA support and a CUDA device is visible.
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -u tools/benchmark_h7_rollouts.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 --episodes 128 \
  --actor-backend shared --actor-device cuda \
  --actor-max-batch 16 --actor-delay-ms 3 \
  --output /tmp/h7-inference-shared-cuda-128.json
```

The three benchmark reports must match `source_digest`,
`model_sha256`, `emulator_dll_sha256`, and cell quotas. Source-seeded
actions should usually match between CPU implementations; numerical
differences between batched and single inference can change a sampled
action near a stochastic threshold, and CPU/CUDA outputs are only required
to meet the PPO source-logprob tolerance.

Reports add `actor_metrics`: local encoding/collation/forward time or
shared encoding/roundtrip time, total worker-episode wall time and server
forward/batch statistics. The sum of worker-episode wall times overlaps
across workers, so it is *not* elapsed cohort time. Server `forward_seconds`
includes CPU transfer after inference, while roundtrip time includes IPC,
server queueing and inference. These do not individually measure .NET
emulator CPU time; treat any remainder as an upper bound, not a precise
emulator profile.

Run training on the same interface with:

```fish
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --output-root results/h7-shared-smoke \
  --device cuda --actor-backend shared --actor-device cuda \
  --workers 15 --width 384 --layers 6 --heads 8 \
  --feedforward 1536 --rounds 1 --episodes-per-round 16 \
  --eval-every 1 --eval-episodes 16 \
  --profile-actor --execute
```

**This is a mechanical smoke command, not an approved large training run.**
Use a fresh output directory, and use `--device cpu --actor-device cpu`
if CUDA is unavailable. The local/default trainer config retains its
previous fingerprint, keeping historical local H7 checkpoint resume
compatible; a shared run records a distinct backend-aware fingerprint.

## Acceptance before substantive training

1. Focused lint, strict mypy and all existing H7 suite gates pass.
2. Native H7 shared two-worker PPO smoke passes, including on-policy
   captured actor log-probabilities and checkpoint output.
3. At least one 128-source, 15-worker **full-size** A10 benchmark shows
   measured throughput and timing breakdown on shared CPU and/or CUDA;
   benchmark against local with identical source/model revisions.
4. Verify fresh/continued optimizer/source replay on a pinned emulator;
   synthetic next-round AdamW restore parity already exists.
5. Determine if batching helps. If the emulator dominates, stop optimizing
   neural inference and prioritize environment throughput instead.
6. Separately expand genuinely supported A10 fixtures beyond the current
   16 train cells; see sts2-emulator issue #185. Inspector parity and exact
   game RNG remain *outside* the training gate.
