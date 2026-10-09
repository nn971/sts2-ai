# Phase-balanced policy optimization (experimental v3)

## Why this option exists

The 20-round tactical v4 experiment (`tactical-v4-report.json`) completed
640/640 training runs and 64/64 heldout evaluations, but never cleared Act 1.
Relative to its preserved phase-split warm start, v4 reached a boss on
**5/64 versus 10/64** test seeds and had a mean paired frontier change of
**-0.334** (30 positive, 34 negative). Training combat victories declined
from 123 in round 1 to 75 in round 12 and recovered to 117 by round 20.
These are not evidence that the structured encoder improved gameplay.

A source-level check found another confound: v2 phase-split loss sums the
policy term across EVERY decision in an episode (often thousands per
cohort), but averages value and entropy terms within each episode before
dividing by number of episodes. About **83% of decisions** in that v4
experiment were tactical. Thus the effective policy/value/entropy tradeoff
changes with episode duration and the distribution of phase lengths.

This experiment adds an **opt-in phase-mean objective**:

```
For each phase p in {combat, strategy}:
    L_p = mean_over_phase_decisions[
        -(target - stop_gradient(V_p(s))) * log pi_p(a|s)
        + value_weight * (V_p(s) - target)^2
        - entropy_weight * H(pi_p(.|s))
    ]
L = L_strategy + L_combat + hp_monotonicity_loss
```

Batch counts are measured from **completed episodes** only. Each phase
receives equal aggregate weight even if the number of tactical card actions
substantially exceeds map and reward decisions. The implementation computes
per-episode gradients scaled by the final batch phase counts, avoiding
retaining a giant multi-episode Torch computation graph.

**Caution:** A trajectory-level *sum* of log-action probabilities is the
standard REINFORCE score function, so the original formulation is not
mathematically wrong. Normalizing by realized decision count changes the
optimization objective and can bias terminal-reward policy gradients. It is
a pragmatic stability/phase-balancing ablation, not guaranteed to maximize
true full-run victory probability. Keep the unmodified baseline and verify
on entirely untouched test seeds.

## Reproduce and compare

The Python API defaults to `legacy_episode_sum` to preserve old callers
and existing checkpoint fingerprint compatibility. The experimental
`tools/train_phase_split.py` CLI defaults to `phase_mean` for *new* runs.
Explicitly supply the flag in scripts:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
python -u tools/train_phase_split.py \
    --warm-start results/phase-split-20-model.json \
    --tactical-state-encoding structured \
    --loss-normalization phase_mean \
    --rounds 12 --episodes 32 --workers 15 --eval-seeds 64 \
    --learning-rate 0.001 \
    --checkpoint results/phase-mean-12-checkpoint.pt \
    --combat-samples-dir results/phase-mean-12-samples \
    --output results/phase-mean-12-model.json \
    --report results/phase-mean-12-report.json
```

This example selects a **lower experimental learning rate** and fewer
rounds than the v4 trial; neither modification has been empirically
validated as better. To isolate just normalization, repeat with
`--loss-normalization legacy_episode_sum` and **identical**
`--learning-rate`, initial model, seeds and rollout settings, but
different output paths. Use the `warm_start_heldout` paired results as
the common-reference control, not as evidence of significance by itself.

Every round prints `phase_diagnostics` with per-decision
`policy`, `value_mse` and `entropy`, alongside the prior metrics.
`mean_loss` is not directly comparable between loss modes and need not
be positive or monotone because the policy-gradient term is signed.

Existing `.pt` **optimizer checkpoints cannot change loss mode in
place**: the configuration fingerprint blocks mixing states. A portable
`.json` exported model may warm start either mode.

## Follow-up

1. First test whether normalized training avoids the v4 mid-run collapse.
2. Then compare policies on new paired runs (full-run clears, boss entry
   and boss damage, HP carried between combats and named potion inventory).
3. Train a strategic continuation value using actual outcomes at combat
   boundaries; do not assign a fixed HP-per-potion price.
4. Preserve full combat *joint* HP/potion distributions for risk-sensitive
   continuation values later, rather than just expectation.

No emulator mechanics, potion reward, action space, player observation
permissions or distributional critic models are changed.
