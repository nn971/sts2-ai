# v9: cached, vectorized PPO backend (v7/v8 compatible)

## Motivation and scope

In the previous paired v7-v8 experiment the v8 PPO optimizer took ~568 s
over 40 rounds, compared with ~158 s collecting emulator episodes. With
3 PPO epochs, up to 4096 combat decisions per round, and many legal actions
per decision, the reference optimizer spends significant time in Python,
feature hashing and very small PyTorch computational graphs.

The new **opt-in** PPO execution backend changes no game mechanics, model
topology, reward definition, or training data. It implements the same
clipped PPO policy/value/entropy objective, per-phase advantage
normalization, combat-boundary GAE and per-phase optimizer updates.

In `phase_split_ppo_batch.py`:

1. Encode each public observation, legal menu, and every enemy instance
   once per cohort. Never access hidden future intents/RNG.
2. Evaluate frozen old policy values/logprobs in forward-only batches.
3. Cache encoded feature tensors across PPO epochs.
4. Pad variable legal-action menus and enemy sets within a minibatch.
   Mask padded action logits to zero probability; exclude padded enemies
   from pooled context. Unspecified action targets carry a -1 pointer
   and gather a zero vector.
5. Execute a single PyTorch batched forward/backward and optimizer step
   for each minibatch, retaining the reference update ordering and
   global gradient clipping. No independent cards/enemies are merged.
6. Report the same per-phase losses, entropy, KL and clip fraction.

v8 species and visible intent/status/position are still associated with
their exact `instance_id`. The ID selects a *slot* in the batch, not a
numeric network feature. Renumbering enemy IDs consistently gives the
same predictions. v7 public-resource and earlier tactical encodings
are also supported.

## Compatibility

- `train_phase_split(ppo_backend="reference")` is the default to keep old
  direct-call behavior and historical checkpoint fingerprints unchanged.
- `tools/train_phase_split.py --ppo-backend reference|batched` chooses
  execution mode; CLI default remains `reference`.
- `tools/train_longrun.py` now defaults to `--ppo-backend batched`.
- The backend is recorded in experiment JSON and staged report validation.
- Batched PPO uses an **additional checkpoint fingerprint key**, so an
  older reference PPO optimizer checkpoint cannot be resumed as batched
  PPO. Models can still be loaded as warm starts and retrained with
  either backend. This avoids accidentally mixing optimization paths.
- No change to model inference JSON or the pinned emulator commit.
- The reference PPO module is retained for regression comparisons.
- CPU-first: the new batch shapes enable a later controlled CUDA
  experiment, but this patch does not claim GPU acceleration.

## GitHub test coverage

`tests/test_phase_split_ppo_batch.py` covers reference/batch values,
individual action logits, gradients of model parameters (including the
v8 enemy encoder), variable enemy/target/action dimensions, padding
and masks, checkpoint/resume and backend mismatch. GitHub workflow
`batched-ppo-parity` runs the test alongside existing PPO and
enemy-instance suites.

## Benchmark (Fish shell)

```fish
python -u tools/benchmark_ppo_backends.py \
    --encoding enemy_instances --decisions 512 --epochs 2 --batch-size 64
python -u tools/benchmark_ppo_backends.py \
    --encoding public_resources --decisions 512 --epochs 2 --batch-size 64
```

This small reproducible synthetic workload measures CPU *feature encoding
and forward/backward*, excluding emulator rollouts. A microbenchmark
speedup is **not** a claim of whole-training speedup.

For an end-to-end comparison, launch two *fresh* runs with the same
emulator, warm start and seeds; do not reuse an incompatible checkpoint:

```fish
git fetch origin
git switch agent/batched-ppo-v9
git submodule update --init --recursive

for backend in reference batched
    python -u tools/train_longrun.py \
        --warm-start results/ppo-pr51-large/models/stage-0160.json \
        --tactical-state-encoding enemy_instances \
        --ppo-backend $backend \
        --rounds 4 --stage-size 4 --episodes 32 --workers 15 \
        --eval-seeds 16 --win-anneal-threshold 128 \
        --train-seed-prefix ppo-v9-speed-train \
        --eval-seed-prefix ppo-v9-speed-eval \
        --output-dir results/ppo-v9-speed-$backend
end
```

Compare summed `rounds[].optimizer_seconds`,
`rounds[].rollout_seconds`, and complete wall-clock time.
Check that the number of completed/censored episodes, phase sample
counts, update count and approximate PPO losses agree. Floating-point
summation order may cause small differences, and a long stochastic
training run need not follow exactly the same action trajectory.
Do not claim win-rate improvement without a separate held-out study.
