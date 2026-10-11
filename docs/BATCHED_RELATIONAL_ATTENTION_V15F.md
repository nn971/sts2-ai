# v15f — Batched relational attention and computational benchmark

## Scope

The v15e public-object representation and v15d two-layer, four-head
relation-aware network are retained *unchanged*. This patch improves the
**training execution** of the same network, not the model's semantics.

`RelationalObjectTorch.forward_batch_prepared` accepts a cohort of
public object sets, action-role bindings and frozen base estimates and
returns one value plus the original variable-length action-logit menu for
each decision.

- Calls `TypedObjectEncoder.encode_many` for all tokens in the minibatch.
- Pads to the largest *visible object count* in the batch, with attention
  scores masking all padded **keys** to prevent false information flow.
- Executes both relation-aware self-attention layers across the cohort.
  Directed ownership biases (self, owner/effect in either direction, sibling,
  unrelated) are preserved per observation.
- Uses valid-token masked mean pooling. The global-player token, the
  object-count feature and the value residual retain their old meaning.
- Flattens all **legal actions across the minibatch** and runs a single
  masked cross-attention using each action's role-aware referenced objects.
  Empty action bindings select the original `empty_action` parameter
  without invalid softmax contribution.
- Splits action logits back to the exact original legal-menu lengths and
  ordering. No alternative action spaces, reindexing or hidden emulator
  information are introduced.

All v15d module weights, version strings, portable serialization and
inference remain unchanged. The old single-decision method
`forward_prepared` is intentionally kept as a tested reference.

## PPO integration

For `--architecture relational`, the existing
`ppo_update_pooled` function now runs the new batched forward once
per minibatch. Each action's clipped PPO ratio, normalized advantage,
value regression and entropy term are computed as before. The frozen
combat/strategy baselines and public terminal-combat GAE are unchanged.
The legacy `--architecture pooled` control still uses its prior path.

This is an execution-only optimization. Existing relational checkpoints
remain resumable because there are **no changes to parameter names or
the configuration fingerprint**.

## Verification

`tests/test_batched_relational_v15.py` compares independent per-decision
and minibatched outputs and gradients on observations with:

- One to multiple enemies, owner-linked duplicate powers and status records.
- Different counts of cards, potions and enemy powers.
- Variable action menu sizes, mixed play/potion/selection actions, empty
  selections and end turn.
- No enemies, wholly empty action bindings, mixed padded batch lengths,
  and duplicate/reordered observations.

Tests compare policy logits and values within 2e-5 and selected neural
parameter gradients within 4e-4. Existing v15d portable parity,
permutation invariance, PPO checkpoint-resume and v15e encoder-gradient
tests are run in the same GitHub Actions workflow.

**This does not claim native gameplay performance or greater win rates.**

## Recommended next test: CPU/GPU throughput, not gameplay success

Use your WSL workstation (9700X and RTX 5070) for the throughput
measurements. This microbenchmark builds deterministic **synthetic public
object shapes** to isolate PyTorch cost; it does not access private game
state, train an agent, or invoke a .NET emulator.

```fish
git fetch origin
git switch agent/v15f-batched-relational-attention
git submodule update --init --recursive

python -m pytest -q tests/test_batched_relational_v15.py \
    tests/test_relational_ppo_v15.py \
    tests/test_vectorized_objects_v15.py

python -u tools/benchmark_v15_attention.py \
    --batch-sizes 1 8 32 128 \
    --devices cpu cuda \
    --threads 1 --warmups 1 --trials 3 \
    --report results/v15f-attention-cpu-gpu.json
```

The JSON report contains, for every device and batch size, the median
forward and forward+backward latency, decisions/s, and speedup relative
to the unbatched reference, plus CUDA peak allocated tensor memory.
CPU memory is deliberately reported as `null` because Python-level
allocation tracking would not measure all native Torch buffers.

No particular speedup is assumed in advance: single-decision batching
may be slower, and large padded batches can waste computation when
object counts vary widely. If that happens, the next optimization is
bucketing public decisions by visible object count (without changing
PPO sampling) and benchmarking packed/block-diagonal alternatives.

A small **CPU-only benchmark smoke** is also run in GitHub Actions.
The user's GPU is required to quantify RTX 5070 performance; it
is not simulated by CI.

## Passing criteria

- All batched/scalar value, policy and gradient parity tests pass.
- Zero-padding cannot influence outputs or gradients of another sample.
- End-to-end toy PPO/resume and old v15c checkpoints remain compatible.
- CPU/CUDA benchmarks run and report honest timing and memory metrics.
- Native selfplay throughput, if later desired, should be measured
  separately from the synthetic model-only benchmark.

The full 739-snapshot win-rate replay is intentionally not required.
