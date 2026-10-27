# v15e — Representation integrity and vectorized typed embeddings

v15d's instance-aware, relation-aware combat model is the chosen
architectural baseline. **The next test is a correctness/coverage/efficiency
audit, not another win-rate comparison.** v15e implements this test and
removes the most obvious per-token embedding dispatch bottleneck.

## What changed

1. The typed object encoder now has `encode_many(object_sets)`. It combines
   all public tokens across multiple decisions, performs embedding gathers
   and segmented categorical reductions once, and batches numerical features
   by token kind. The result is split back into each decision's distinct
   token rows. `encode(object_set)` uses the same vectorized implementation.
   Neither card/enemy instance identifiers nor the emulator's hidden draw
   order become neural features. v15c/v15d checkpoint tensor names and
   dimensions remain unchanged.
2. Vocabulary construction accepts the emulator's *already-public*
   `capability_manifest` (`cardIds`, `enemyIds`, `relicIds`,
   `potionIds`, `powerIds`). The JSONL backend has already validated this
   manifest at initialization. It is **not** derived from private state or
   from ad-hoc C# parsing. This removes needless identity OOV collisions
   for implemented but infrequently observed content.
3. `ObjectVocabulary.oov_report` reports identity, role, categorical and
   numeric-field unknown rates separately, with non-instance-ID examples.
   Dynamic attributes can still be OOV; we do not silently claim a static
   manifest contains every modifier.
4. The native audit CLI checks object reference uniqueness, every legal
   action's object bindings, vocabulary coverage and Torch/portable
   numerical parity using a nonzero-gated v15d head. It **does not train or
   score the agent's win rate**. It writes a machine-readable JSON report.

## Next test: small native public-state stress audit

In WSL, from your `sts2-ai` checkout:

```fish
git fetch origin
git switch agent/v15e-representation-integrity
git submodule update --init --recursive

python -m pytest -q \
    tests/test_object_instances_v15.py \
    tests/test_vectorized_objects_v15.py \
    tests/test_relational_objects_v15.py \
    tests/test_relational_ppo_v15.py

python -u tools/audit_v15_representation.py \
    --model results/ppo-v8-batched-40/models/stage-0120.json \
    --report results/v15e-representation-audit.json \
    --runs 4 \
    --max-decisions 512 \
    --parity-checks 4 \
    --build
```

Four short native run prefixes provide a diverse set of public combat
states and actions. All *recorded combat decisions*, not merely the four
parity-selected ones, receive structural extraction and action-binding
validation. Four evenly spaced decisions also receive a fresh, nonzero-gated
relational PyTorch/portable equivalence check with a tight numerical
tolerance. The manifest makes content identities available even when
those cards or enemies were not seen in the sample.

### What counts as passing?

- Every observed legal action has valid visible object references.
- No duplicate object-reference collisions, unobserved private fields,
  or non-finite Torch/portable results.
- The full manifest-plus-sample vocabulary has zero identity unknowns.
- The selected nonzero-gated PyTorch/portable scores agree within 5e-4.
- The CLI exits successfully and saves the audit report.

The `out_of_sample_oov` report separately records how much mutable
metadata or status identity is not predictable from just the **first half**
of the public observations plus the complete static catalog. This value
is diagnostic, not a hard zero-OOV requirement.

The test does **not** need fresh policy training, large GPU runs or
another 739-encounter win-rate evaluation. If it passes, the next
engineering task should be **end-to-end batched v15d relation attention
and action cross-attention**, using the same parity and invariance checks
to protect correctness while improving throughput.

## Existing checkpoint compatibility

The embedding vocabulary schema remains `v15b`: contents and array sizes
are versioned by the actual saved vocabulary. Old checkpoints retain
their original frozen entries. A new training run with a complete emulator
catalog creates a larger vocabulary and therefore a new checkpoint; the
training resume fingerprint must not allow silently interchanging them.

There is no claim that the new batched `encode_many` alone removes all
latency: the current v15d relational forward still assembles its attention
and action scoring **separately per decision**. This patch fixes one
high-dispatch area and installs the tests for the next optimization.
