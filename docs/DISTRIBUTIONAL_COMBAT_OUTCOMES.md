# Distributional combat outcomes (experimental, October 2026)

This extends the combat-outcome instrumentation and phase-separated trainer
without changing the emulator or the deployable tactical policy.

## Why a distribution, not just expectation?

Mean exit HP does not describe whether a combat usually wins cleanly but
occasionally catastrophically loses, or nearly always ends at half HP.
The earlier public outcome predictor emits survival probability and mean HP.
This experimental predictor instead models:

- `Pr(combat survived | public combat-entry observation)`;
- ten HP-ratio bins **conditional on surviving**;
- a consistent joint distribution assigning one distinct category to
  defeat (HP=0) and ten to positive surviving HP fractions;
- derived mean, **normalized variance** and quantiles, without choosing
  a risk criterion.

Defeats must never train the conditional survivor-HP histogram with fictitious
zero-HP survival labels. Exact exit HP, named potions and other persistent
resource changes remain in the underlying per-combat JSONL records.

**This auxiliary model is not yet coupled to the strategy continuation
critic.** In particular, it makes no forecast of the next potion inventory
and cannot correctly price a potion against future challenges. A future
joint survival/HP/potion model can replace its output format. It does not
see hidden RNG state or use a teacher.

## Data and experiment commands

The old 8-combat diagnostic JSON report is **not a training dataset**. Older
JSONL combat exports also lack `entry_public_json`. Recollect combat samples
using the current recorder through `tools/train_phase_split.py` or
`tools/train_selfplay.py`, which export per-combat entry observations.

Example phase-separated run, using available pinned native Overgrowth and
recording independent samples (fish shell):

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
python -u tools/train_phase_split.py \
  --rounds 20 --episodes 32 --workers 15 \
  --warm-start results/stronger-pretrained-model.json \
  --checkpoint results/distribution-split-checkpoint.pt \
  --combat-samples-dir results/distribution-split-samples \
  --output results/distribution-split-model.json \
  --report results/distribution-split-report.json
```

Replace the warm-start path with a **locally available** compatible model
with identical dimension/hidden parameters (defaults 128 and 32). Do not
use the 16-unit single-round smoke model with the 32-unit default. If no
compatible checkpoint is available, omit `--warm-start`.

After collecting many runs:

```fish
python -u tools/train_combat_distributional.py \
  --samples results/distribution-split-samples \
  --epochs 30 \
  --output results/combat-distribution-model.json \
  --report results/combat-distribution-fit.json
```

The CLI prints each epoch. It uses a **run-seed-disjoint** holdout and
reports heldout survival Brier, exit-HP MAE/RMSE and joint HP-category NLL,
with a smoothed constant training-frequency baseline. Evaluate against
that baseline before calling the predicted probabilities informative.

## Limitations and next steps

1. Ten discrete positive-HP bins lose numerical resolution; reported means
   and variances are approximate. We keep exact samples for future studies.
2. Training data is **on-policy**. There is no basis for interpreting the
   predictor as a counterfactual evaluator of unchosen combat actions.
3. Both survival and positive HP predictions can be miscalibrated, especially
   with sparse encounter conditions. Model variance is outcome variability,
   **not epistemic model uncertainty**.
4. The model does not yet predict remaining potion identities. Future work:
   joint resource-outcome distribution, per-encounter calibration and
   stratification, risk-sensitive strategic continuation values, and heldout
   whole-run ablations. Do not impose a fixed HP-per-potion conversion.
