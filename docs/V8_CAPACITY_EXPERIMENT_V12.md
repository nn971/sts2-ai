# v12: Controlled v8 capacity experiment (32 vs 64 hidden units)

## Hypothesis

Before adding a Transformer, test whether the **same species-aware enemy-instance
architecture** benefits from twice the hidden width, independent of extra
parameters in attention. We test **hidden 32 vs hidden 64** in both strategy
and combat heads, keeping the feature dimension 128, the exact target-pointer
logic, and the three-epoch cached PPO backend unchanged.

The new **starting point** is the *independently confirmed* round-120
champion: 171/512 certified Act-1 clears, compared with 118/512 at round
40, on the earlier confirmation cohort. **Those 512 seeds must not be
reused to select width.**

## Critical warm-start consideration

The trainer intentionally refuses to load a hidden-32 model into a hidden-64
network. We preserve that rule. Instead:

1. Width 32 fine-tunes the exported round-120 model directly.
2. Width 64 is created by `tools/widen_phase_split.py`, which duplicates each
   hidden coordinate and its input parameters, including the enemy instance
   projection, context scale, and action-target contribution.
3. Each copied coordinate's policy/value **outgoing weight is split into
   unequal fractions that sum to one**. At initialization this preserves the
   complete state value and all legal-action logits (within floating-point
   rounding); the unequal splits allow clones to learn independently.
4. Both widths start with a **fresh AdamW optimizer state**, fresh 80-round
   training budget, and the *same game seeds and global RNG seed* within
   each paired training replicate. No optimizer state is transferred from
   round 120 or between widths.

This tests which width improves more after equal additional experience from
the same champion behavior. It does **not** independently replicate the
original 0-to-120-round training recipe. Three distinct continuation
replicates provide an initial robustness check, not a complete seed study.

## Experiment design

| Setting | Value |
|---|---|
| Training replicas | 3 paired seeds: 29, 53, 71 |
| Widths | 32, 64 |
| Feature dimension | 128 (identical hashing/features) |
| Additional PPO rounds | 80 |
| Episodes per round | 32, 2,560 per model |
| Worker count | 15 per training job |
| PPO | cached batched; epochs=3, batch=128, limit=4096 |
| Reward | hp_preservation, no HP monotonicity term |
| Victory annealing | 128 |
| Training seed prefix | ppo-capacity-v12-r1/r2/r3-train |
| Monitoring evaluation | 16 seeds, distinct from inference cohorts |
| Comparison cohorts | 256 exploratory + 512 fresh confirmation seeds |
| Emulator | pinned SHA 9117b4af09f0164a19bb1c41b70f688c8948e0e3 |
| Goal | certified native Act-1 boss clear |

Within pair `r1`, the two widths use **the same 80×32 episode seed
strings**, but the actions—and therefore emulator RNG consumption—may
diverge as policies learn. This is a *matched random-seed design*, not
guaranteed identical experience or exact paired replay trajectories.

The wall-clock costs may differ because the wider model changes inference
and optimizer expense. We record training-loop rollout and optimizer
seconds from all 80 rounds for both widths; judge skill at **equal
environment interactions** first, then report compute tradeoffs.

## Run with Fish on WSL

Restore `results/ppo-v8-batched-40/models/stage-0120.json` from your
completed 120-round archive. No `checkpoint.pt` is needed for the
capacity experiments because the new optimizer runs start fresh.

```fish
git fetch origin
git switch --track origin/agent/capacity-width-v12
git submodule update --init --recursive
python -m pytest -q tests/test_capacity_width_v12.py

# First run a matched 40-round pilot pair:
fish tools/train_capacity_v12.fish pilot r1

# If the pilot has no technical problems, run/extend all three pairs to 80:
fish tools/train_capacity_v12.fish full all

# Evaluate all 6 resulting policies on fresh paired seed cohorts:
fish tools/evaluate_capacity_v12.fish
```

Training results: `results/ppo-capacity-v12/r1/h32`, `r1/h64`, etc.
Each run has an independently resumable `checkpoint.pt`, stage JSON
model, reports, logs, and baseline evaluation cache. An interrupted
training stage can be repeated with the same command.

The initial widened model is saved to
`results/ppo-capacity-v12/warm-start-width64.json` and verified for
deterministic equality on repeated creation.

Evaluation results:
`results/ppo-capacity-v12/evaluation/comparison-256.json`,
`comparison-512.json`, plus `summary-256.md` and `summary-512.md`.

The comparison evaluator uses **non-selective comparison mode**: it
evaluates all six predefined models. It does not automatically choose
a lucky training replicate as a champion. Journal resumes are keyed by
all model SHA-256s, the pinned emulator, seed prefix, policy and goal.

### Interpretation rule (predeclared)

Treat evidence for higher capacity as *preliminary* if the 64-wide policy
outperforms 32-wide in at least 2/3 training pairs and the mean paired
advantage is positive on the 256-seed selection cohort. Consider the
finding supported if it persists on the distinct 512-seed cohort,
particularly if the improvement is not wiped out by much slower
inference. With only three training replicates we will not claim tight
confidence intervals on architecture-level effects.

If gains are small/inconsistent: do **not** assume a Transformer will
help; inspect optimization dynamics and compare more training seeds.
If gains are stable, the next comparison is a modest one-layer
enemy-self-attention model under exactly the same rollout and evaluation
budgets (and ideally parameter-count control).

The old 512-seed round120 confirmation set stays untouched throughout.
