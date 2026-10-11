# v15g — Profile real PPO minibatches on CPU and CUDA

## Why this test

The v15f synthetic benchmark established a significant batching speedup but
also found the Ryzen CPU faster than the RTX 5070 on *that particular*
64-wide, two-layer attention workload. The next question is which stages
dominate PPO **on actual native-Overgrowth public combat decisions**.

This test does **not** optimize the agent's tactics, use hidden game state,
or repeat the 739-snapshot win-rate evaluation.

## What is implemented

- `tools/profile_v15_real_minibatches.py` collects native-overgrowth runs
  through the pinned `JsonlEmulatorBackend`, using the frozen
  `PhaseSplitNeuralModel` and exactly the existing public selfplay loop.
- Only **resolved combat segments** enter the replay corpus. Real
  HP-preservation combat targets, terminal-combat GAE advantages, actual
  legal actions and on-policy baseline log probabilities are produced by
  the existing `prepare_pooled_samples` pipeline.
- The resulting corpus JSON contains **only public observation JSON,
  action kinds and payloads, baseline estimates, selected legal-action index
  and their PPO values/advantages**. It contains no emulator handles,
  canonical state hashes, RNG streams or internal draw-pile order.
- A SHA-256 checksum protects the corpus from accidental changes. Its
  pinned emulator revision and frozen baseline checkpoint hash travel in
  the file. Re-running profiling from this corpus does not require .NET
  or generate different encounters.
- The CPU preparation pass measures JSON decoding, public object
  extraction, and validated legal-action/object binding.
- An opt-in stage hook in `RelationalObjectTorch.forward_batch_prepared`
  measures object embeddings, public ownership/relation tensor setup,
  two-layer object attention, value context, action metadata, action
  cross-attention and policy head. The hook is **absent in normal training**,
  where it introduces no synchronization.
- PPO steps include real old log probabilities, normalized actual GAE
  advantages, clipped actor objective, value regression, entropy term,
  gradient backpropagation, gradient clipping and AdamW update. These
  updates are **throwaway benchmark-only model weights** and do not
  modify any user checkpoint.
- Each device/batch uses the **same sample indices and initial weights**.
  To ensure fair CUDA measurements, the profiler synchronizes device
  execution at each instrumented stage. It also reports
  **separate uninstrumented end-to-end timings and decisions/s**. Use
  the latter, not the instrumented totals, for throughput comparisons.

## Recommended command (Fish in Ubuntu WSL)

Start with this single collection + CPU/CUDA test:

```fish
git fetch origin
git switch agent/v15g-real-minibatch-profiling
git submodule update --init --recursive

python -m pytest -q \
    tests/test_real_minibatch_profile_v15.py \
    tests/test_batched_relational_v15.py \
    tests/test_relational_ppo_v15.py

python -u tools/profile_v15_real_minibatches.py \
    --base-model results/ppo-v8-batched-40/models/stage-0120.json \
    --corpus results/v15g-public-combat-ppo-corpus.json \
    --report results/v15g-real-minibatch-profile.json \
    --runs 4 --max-decisions 512 \
    --devices cpu cuda \
    --batch-sizes 32 64 128 \
    --threads 1 --width 64 \
    --warmups 2 --trials 5 --build
```

**If the corpus already exists** it is automatically loaded, and combat
collection is skipped. That makes iterating across devices, thread counts,
batch sizes and profiling patches reproducible. To intentionally recollect
it with the same base model, pass `--collect`.

**If CUDA is unavailable**, use `--devices cpu` for the CPU portion.

The saved corpus can be relatively large because it contains complete
public JSON observations and all legal action payloads. This is
intentional; avoid including this generated research data in Git commits.

## How to interpret the report

`sample_preparation_per_decision_ms` is CPU-only, measured once across
all resolved combat samples:

- `json_decode`: JSON parsing of public observations.
- `object_extraction`: tokenizing visible cards, enemies, powers, etc.
- `action_binding`: validating and linking legal actions to their objects.

Each `rows[]` entry has the selected real sample's mean/max token count,
mean/max legal-action count and:

- `uninstrumented_total_seconds`, `decisions_per_second`: preferred
  throughput numbers. Include forward, clipped PPO loss, backward,
  gradient clipping and optimizer step; exclude emulator rollout/collection
  and the one-time CPU preparation pass.
- `median_seconds.object_embedding` through
  `median_seconds.policy_head`: synchronized forward stages.
- `median_seconds.ppo_loss`, `backward`, `gradient_clip`,
  `adamw_step`: training-update breakdown.
- `instrumentation_overhead_ratio`: approximate slowdown from
  synchronized phase hooks. Do not sum independently taken medians and
  interpret them as an exact measured training step.
- `peak_cuda_allocated_bytes`: peak allocated GPU memory for the timed
  minibatches (not total system/video-memory utilization).
- `reused_observations`: warns if fewer than the requested batch size
  have been collected and sampling with replacement was necessary.

The profiler is **sequential**, not the full selfplay worker pool.
If GPU is still slower, this identifies the exact stage to optimize
(e.g. PyTorch tensor construction in relation setup vs. actual attention
matrix multiplications), rather than assuming that a larger batch or
faster GPU must help.

## Pass/fail and safety

- The corpus is checksum-protected; invalid public object/action links
  or empty resolved-combat data fail explicitly.
- Focused CI uses a **public toy emulator** for corpus and full CPU PPO
  profile checks. No GPU is required for repository CI.
- Each stage hook is opt-in, so ordinary PPO remains untouched.
- Same model architecture, vocabulary and serialization as v15f; existing
  frozen checkpoints continue to load.
- Real, pinned native-runtime profiling must be run on the user's WSL
  machine; do not interpret toy/CI measurements as that result.

Upload `results/v15g-real-minibatch-profile.json` to choose the next
optimization based on the real bottleneck.
