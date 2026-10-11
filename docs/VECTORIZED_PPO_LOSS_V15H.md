# v15h — Vectorized PPO loss on actual public combat minibatches

The v15g real native PPO benchmark reported 620 decisions from 29 resolved combats at pinned emulator revision 9117b4af. At width 64 and batch 64, *uninstrumented* CPU reached ~1312 decisions/s and CUDA ~721 decisions/s. On the GPU, **actual owner-aware attention was ~1.1 ms**, but the prior per-decision PPO loss consumed ~11.5 ms, its backward graph ~53 ms, and relation/mask setup ~8.8 ms. These are measurements of the previous v15g code, **not v15h performance claims**.

## Architecture-preserving change

`src/sts2_ai/training/pooled_ppo_v15.py` now provides
`batched_ppo_loss`: batched padded action logits, one masked log-softmax,
one chosen-action gather, one vectorized clipped actor/value/entropy
objective across the minibatch. Padded action positions receive
`torch.finfo(dtype).min` and are masked from entropy. A decision with one
legal action has entropy exactly zero. Old log probabilities, GAE
advantages, value targets, temperature, clipping, gradient clipping and
optimizer semantics remain unchanged.

Only relational-v15d PPO minibatches use this path. Pooled-v15c PPO still
runs the prior reference implementation. There are **no new model weights,
checkpoint schema versions or hidden emulator fields**.

Tests in `tests/test_vectorized_ppo_loss_v15.py` compare per-decision
reference and vectorized PPO: every component of the loss, gradients
through variable-width action logits and critic outputs, mixed positive
and negative advantages, one-action menus, invalid indices, and
different temperatures. The existing relational checkpoint resume test
remains in the focused CI workflow.

## Next test: same captured native corpus, old vs new PPO loss

Reuse the v15g corpus you already collected. **Do not recollect combats**
or rerun fidelity/win-rate evaluations.

In WSL (Fish):

~~~
git fetch origin
git switch agent/v15h-vectorized-ppo-loss
git submodule update --init --recursive

python -m pytest -q tests/test_vectorized_ppo_loss_v15.py \
    tests/test_relational_ppo_v15.py \
    tests/test_real_minibatch_profile_v15.py

python -u tools/profile_v15_real_minibatches.py \
    --corpus results/v15g-public-combat-ppo-corpus.json \
    --report results/v15h-ppo-loss-comparison.json \
    --devices cpu cuda \
    --batch-sizes 32 64 128 \
    --loss-modes loop batched \
    --threads 1 --width 64 \
    --warmups 2 --trials 5
~~~

The profiler resets the identical network weights before each combination
of device, batch size and loss mode; all modes share the same sampled
combat decisions. Each JSON row now records `loss_mode` (`loop` or
`batched`), uninstrumented full PPO update time and decisions/s, and
the same synchronized phase breakdown. Compare **uninstrumented total
time first**, then `median_seconds.ppo_loss` and `backward`. GPU peak
allocation may differ slightly because of the padded action logits.

Do not interpret phase timings as actual end-to-end runtime without
their synchronization overhead. Single-threaded CPU results are the
relevant v15g baseline; optimizations may behave differently with more
threads or different legal-action counts.

## What is next?

If this reduces GPU PPO loss and backward enough, revisit CPU vs CUDA
throughput. If GPU still trails substantially, a separate optimization
should precompute immutable public relation/action topology once per
sample rather than rebuilding attention-bias matrices on-device every
minibatch. This is another **execution-only** change with scalar/batch
parity tests.

The temporary unknown-choice tokens from v15g remain explicitly
reported. Actual ChoicePool public projection is tracked in
sts2-emulator issue #99; this patch does not claim to restore the
identities of unavailable generated choice cards.
