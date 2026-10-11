# v15i: cache public relation topology across PPO epochs

## Motivation

On the **same captured v15g public combat corpus** (620 decisions, 29 resolved combats),
v15h's masked batched PPO loss reduced CUDA PPO loss time at minibatch 128
from 23.66 ms to 2.34 ms and end-to-end update time from 164.73 ms to
103.98 ms. Remaining CUDA costs include ~15.42 ms of relation setup and
~71.54 ms of full backward propagation; actual two-layer object
attention itself is ~1.11 ms. These are measured v15h results, **not a
claim of v15i speedup**.

## What changed

The network weights, portable model, public observation representation,
legal action semantics, PPO loss, optimizer, checkpoint schema and
frozen baseline model are unchanged.

- A new immutable RelationalTopology stores public object count, global
  player token position, a directed NxN relation matrix, action-kind
  indices and per-action referenced-object row/semantic-role pairs.
  Raw card/enemy instance integers are used only for object lookup
  during preparation, **never** as neural input features.
- PPO compiles this topology **once per sampled combat decision** while
  caching behavior policy values and advantages, then reuses it in each
  shuffled PPO minibatch across all epochs.
- The batched relational forward accepts an optional topologies array.
  When supplied, it constructs a single padded relation tensor per
  minibatch, with one host-to-device transfer rather than reconstructing
  directed ownership relationships or looking up all action refs on every
  gradient step.
- Calling forward_batch_prepared without topologies retains the
  original dynamic implementation for numerical equivalence and
  profiling comparisons. Torch and portable single-decision inference
  do not require persistent topology.
- The v15g real PPO profiler now supports --topology-modes dynamic cached,
  paired with --loss-modes loop batched. The replay uses the same
  checksum-protected corpus, model initialization and decision indices.

## Verification

The dedicated v15 tests include:

- Dynamic vs cached logits, values and chosen parameter gradients on
  mixed object counts, repeated effects and variable action menus.
- A no-recomputation test, which explicitly raises if the cached
  forward calls the _relations builder.
- Reject inconsistent topology batch/menu lengths.
- Full PPO sample preparation and profiler runs with both topology modes.
- Existing v15h scalar-vs-batched PPO loss, portable parity, PPO
  resume and ChoicePool placeholder tests.

Caching introduces no new trainable parameters and does not rewrite
model artifacts. The dedicated repository workflow uses CPU PyTorch;
the user's RTX 5070 remains necessary to quantify GPU acceleration.

## Rerun *without* collecting combats (Fish on WSL)

~~~
git fetch origin
git switch agent/v15i-cached-relational-topology
git submodule update --init --recursive

python -m pytest -q tests/test_cached_relational_topology_v15.py \
    tests/test_batched_relational_v15.py \
    tests/test_vectorized_ppo_loss_v15.py

python -u tools/profile_v15_real_minibatches.py \
    --corpus results/v15g-public-combat-ppo-corpus.json \
    --report results/v15i-topology-comparison.json \
    --devices cpu cuda \
    --batch-sizes 32 64 128 \
    --loss-modes batched \
    --topology-modes dynamic cached \
    --threads 1 --width 64 --warmups 2 --trials 5
~~~

Compare uninstrumented_total_seconds first (full PPO update), then
median_seconds.relation_setup (cache target) and backward (may remain
dominant), while checking CUDA peak allocated tensor memory. The
report separately supplies topology_preparation_seconds. That is a
one-time CPU corpus preprocessing cost excluded from PPO-update
throughput; amortize it over actual PPO epochs before judging
end-to-end efficiency.

The compatibility observation gap tracked in sts2-emulator#99 remains
explicit: missing generated ChoicePool cards receive unobserved-choice
markers; their unknown identities are never fabricated. Nothing in
this PR changes the emulator pin or the frozen v14 fidelity corpus.
