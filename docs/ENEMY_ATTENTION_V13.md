# v13: one-layer enemy relational attention, matched v8/v12 controls

## Hypothesis and limits

Increasing hidden width from 32 to 64 **did not reliably improve** held-out
Act-1 clears across three v12 training pairs. Test a more specific representational
change: can one enemy's *visible* status, species and intent influence how the
agent scores another enemy as a target? v8 pools enemy embeddings before scoring
actions, but does not contextually modify each target according to the other
enemies. v13 introduces one self-attention-style relational layer over enemy
instances (with self-edges excluded).

This is **not** a full Transformer over history, cards or relics. Strategy
head, global combat features, targeted card features, action masking, exact
environment and PPO objective remain unchanged.

## Model

Let `e_i = ReLU(W_e x_i + b_e)` be the existing v8 embedding of the
**player-visible** enemy instance `i`, width 32. For each of 4 heads
(`d_head=8`), form learned query, key, value projections, and attend to
*other living enemy instances only*:

`alpha_ij = softmax_j (q_i dot k_j / sqrt(d_head)), j != i`.

Concatenate the weighted values; apply a learned output projection and
a per-coordinate residual gate:

`e'_i = e_i + gate * W_o concat_heads(sum_{j != i} alpha_ij v_j)`.

For 0 or 1 enemy, the relational update is **exactly zero**. The existing
global pooling and targeted-action scoring use `e'_i` instead of `e_i`.

At initial warm start `gate=0`; Q/K/V/O are small random matrices.
Thus **every initial action logit and the value prediction remain identical**
to those of the round-120 width-32 champion (within floating-point precision).
The gate can learn immediately, and Q/K/V/O receive gradients once it moves
off zero. Instance IDs remain lookup keys, not numeric model inputs.
Padding and diagonal/self edges are masked in batched Torch.

The portable Python inference and single-decision/batched PyTorch implement
the same computation. All existing v8 models remain valid and unmodified.
New checkpoints have explicit
`sts2-neural-policy-value-v13-enemy-relational-attention` format with
strictly validated attention tensors. The v8->v13 warm-start migration preserves
**all** existing state/action/value projections, not merely action weights;
optimizer checkpoints between formats remain deliberately incompatible.

## Predeclared comparison

**Primary control:** the v12 width-32 models, all trained for 80 rounds beyond
the same round-120 v8 champion. **Capacity control:** the v12 width-64 models.
**New candidate:** v13 width-32 attention, also trained for 80 rounds from
that champion.

Training replicas `r1/r2/r3` reuse the v12 RNG seeds `29/53/71` and the
exact same training seed strings `ppo-capacity-v12-rN-train`. Each 80-round
run uses 32 episodes/round (2,560 total), 15 workers, and the same batched PPO
hyperparameters. Attention trains **fresh optimizer state**, just like v12
width-32 and width-64.

The game RNG follows each policy's actions, so the same seed does **not**
mean identical episodes after the models diverge. The additional attention
parameters modestly increase training/inference work.

The paired heldout evaluation cohorts are **fresh**, disjoint prefixes and
are not the original 128, the earlier 512 confirmation, or the v12 256/512:

- `ppo-v13-attention-comparison-256`: exploratory, all nine fixed models.
- `ppo-v13-attention-comparison-512`: final, all nine fixed models.

No checkpoint is selected on the basis of these cohorts; this is an
**architecture-level** comparison of three preselected pairs. Report mean
attention-minus-width32 win-rate across **training replicas**, signs of
all three paired differences, boss-entry/conversion rates, and inference
and training cost. Paired McNemar tests within any one model pair
measure evaluation-seed uncertainty only; they do not make three training
seeds into hundreds of independent architecture replications.

Interpretation: positive mean attention-minus-h32 and at least 2/3 positive
training pairs on both cohorts are **preliminary** evidence for the relational
hypothesis. An attention benefit confined to Kin-like multi-enemy encounters
would be worth further targeted evaluation, but these reports alone do not
identify tactical causality. If outcomes are inconclusive, expand training
replicates rather than cherry-picking checkpoint winners.

## Run on WSL with Fish

Requirements: same pinned emulator git submodule revision
`9117b4af09f0164a19bb1c41b70f688c8948e0e3`, working .NET and Python
venv, the existing v8 round-120 champion at
`results/ppo-v8-batched-40/models/stage-0120.json`, and complete v12
`r1/r2/r3` h32/h64 checkpoints in
`results/ppo-capacity-v12/rN/hW/models/stage-0080.json`.

```fish
git fetch origin
git switch --track origin/agent/enemy-attention-v13
git submodule update --init --recursive
python -m pytest -q tests/test_enemy_attention_v13.py

# First validate training startup, time and zero-gate neutrality:
fish tools/train_enemy_attention_v13.fish pilot r1

# Then finish r1 and run r2/r3 to 80 rounds each:
fish tools/train_enemy_attention_v13.fish full all

# Compare nine frozen checkpoints on two fresh 256/512-seed cohorts:
fish tools/evaluate_enemy_attention_v13.fish
```

The training prints every round via the existing `train_longrun.py` pipeline.
Interrupted stage jobs resume their private `checkpoint.pt`.
No new in-game replay or HTML UI is involved.

Expected outputs:

- `results/ppo-enemy-attention-v13/r1/attn/` (likewise r2, r3):
  checkpoint, models/stage-0040.json, models/stage-0080.json, stage logs.
- `results/ppo-enemy-attention-v13/evaluation/comparison-256.json`,
  `comparison-512.json`, and corresponding resumable partial journals.
- `results/ppo-enemy-attention-v13/evaluation/summary-256.md`,
  `summary-512.md` and machine-readable JSON summaries.

The frozen v12 control checkpoints are **read**, never overwritten.
See focused GitHub CI `enemy-attention-v13` for masked-batch parity,
permutation equivariance, zero/singleton enemy behavior, gradients,
serialization, and v8 regression tests.

## Important scope limit

The 2,560-episode continuation evaluates attention *added to the existing
well-trained v8 model* versus further fine-tuning without attention. This
does not establish that attention improves a randomly initialized model,
nor that a Transformer with history/cards would help. If results are
promising, test a parameter-count-matched feedforward control and
multi-enemy diagnostic situations before expanding the architecture.
