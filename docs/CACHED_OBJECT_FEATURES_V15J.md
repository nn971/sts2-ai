# v15j — Compile immutable public object features once per PPO decision

## Baseline observation (user-provided v15i corpus)

At model width 64 on the same **620 native public combat decisions** and 29
resolved combats, v15i compiled relation/action topology reduced RTX 5070
end-to-end PPO minibatch costs by 7–9%:

| Batch | GPU dynamic topology | GPU cached topology |
|---|---:|---:|
| 32 | 1077 decisions/s | 1159 decisions/s |
| 64 | 1380 decisions/s | 1482 decisions/s |
| 128 | 1460 decisions/s | 1597 decisions/s |

CPU results were mixed, and caching at CPU batch 128 was substantially
slower. The GPU's remaining expensive stages at batch 128 are ~58 ms backward,
~9 ms relation setup and ~8 ms embedding. These are **v15i measurements**,
not v15j results.

## What this experiment changes

`TypedObjectEncoder.prepare` compiles **only immutable public token data**
for a single captured decision into a frozen `CompiledObjectFeatures` object:

- Kind, card/enemy identity and role *vocabulary indices*
- Categorical tag indices, token-row memberships and multiplicities
- Numeric field indices and signed-log-transformed **public observed
  numeric values**, grouped by object kind, preserving all multiplicities

`TypedObjectEncoder.encode_many_compiled` evaluates those pre-indexed
features using **the current trainable embedding tables, numeric projections,
layer normalization and autograd graph on every optimizer step**. It does
not cache actual neural embeddings, model activations or gradients. Raw
instance integers are omitted. The previous `encode_many` is maintained
as a fully independent reference, and the relational forward accepts an
optional `features` argument.

PPO sample preparation compiles the public features alongside the
v15i topology exactly once per decision, then reuses both throughout
the PPO epochs. The v15c pooled control and portable inference retain
their previous paths. No weights or model checkpoint formats change.

As usual, this optimization applies only to **immutable replay samples**:
a later observation with different cards or card state needs its own
compiled features. Runtime inference remains live and uncached.

## Numerical verification

The dedicated test `tests/test_cached_object_features_v15.py` compares:

1. Dynamic and cached embeddings on real toy-observation features,
   including kinds, categorical tags, numeric projections and gradients.
2. End-to-end relational attention values, logits and parameter gradients
   with the same cached topology and either dynamic or compiled features.
3. Re-evaluating a compiled sample after changing neural weights; the
   output **must** change, proving no stale activation reuse.
4. Empty token batches, invalid feature batch lengths and token counts.
5. Actual resolved-combat PPO sample caching and toy CPU profiler replay
   with both feature modes.

## CPU/CUDA benchmark (Fish, WSL)

The profile corpus is unchanged and already exists. Do not recollect or
rerun fidelity/win-rate tests. Switch to this stacked PR branch and run:

~~~
git fetch origin
git switch agent/v15j-cached-object-features
git submodule update --init --recursive

python -m pytest -q \
    tests/test_cached_object_features_v15.py \
    tests/test_cached_relational_topology_v15.py \
    tests/test_relational_ppo_v15.py

python -u tools/profile_v15_real_minibatches.py \
    --corpus results/v15g-public-combat-ppo-corpus.json \
    --report results/v15j-object-feature-comparison.json \
    --devices cpu cuda \
    --batch-sizes 32 64 128 \
    --loss-modes batched \
    --topology-modes cached \
    --feature-modes dynamic compiled \
    --threads 1 --width 64 --warmups 2 --trials 5
~~~

Each pair uses **identical decisions, model weights, loss, topology and
optimizer**, differing only in object-feature compilation. The report includes
a new `feature_mode` per row and a one-time
`feature_preparation_seconds` cost. Use
`uninstrumented_total_seconds` or `decisions_per_second` for device
comparisons, and `median_seconds.object_embedding` to isolate feature
preparation overhead. Synchronized CUDA stage timings include explicit
barriers.

The full native GPU speedup remains to be measured. A new v15j benchmark
file is not an agent-strength result and cannot establish training
convergence, gameplay skill or fidelity.

### Accept or revert?

If the compiled feature path improves the GPU without a material CPU
regression, retain it for PPO. If it loses across devices, prefer the
existing dynamic path instead of optimizing for theoretical elegance.
The existing emulator ChoicePool public-visibility gap (sts2-emulator#99)
and explicit unknown-choice placeholders remain unchanged.
