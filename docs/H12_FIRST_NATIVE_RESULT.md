# H12 first native A10 result and resource-boundary correction

## Immutable first result (29 October 2026)

The following values come from the locally generated `heldout-report.json`,
`train.log`, `heldout.log` and `fit.log`; the report's policy/emulator
fingerprints remain the canonical experiment identity. **Do not overwrite
the existing `outcome-model.pt` or `heldout-report.json`.**

| | Training | Heldout |
|---|---:|---:|
| Full native A10 source runs | 1,024 | 256 |
| Act 1 clears | 58 (5.66%) | 21 (8.20%) |
| Censored runs | 0 | 0 |
| Combat entries with complete exits | 5,445 | 1,400 |
| Public macro states | 43,135 | 11,093 |
| Combat survival Brier | 0.03924 | **0.04714** |
| Constant survival Brier | 0.14463 | 0.13825 |
| Exit-HP fraction mean absolute error | 0.06642 | 0.08103 |
| Named held-potion retention BCE/combat | 0.01410 | 0.02059 |
| Macro continuation mean squared error | 0.03371 | **0.10659** |
| Constant macro continuation MSE | 0.06832 | 0.09107 |

The survival forecast improves substantially over the constant-survival
baseline. Its high-survival bin (`p in [0.8, 1]`) is optimistic:
1,111 heldout fights, mean predicted survival 98.48%, observed survival
97.39%. Its mid-probability bins also have calibration discrepancies.

The continuation predictor **fails heldout**: it improves training MSE
but is worse than the heldout constant baseline. Even using the *training*
continuation target mean as a predeclared constant baseline would give
approximately 0.09210 heldout MSE, still below the model's 0.10659.
The model currently has weak public-map and action representations, and its
supervision assigns the same terminal outcome scalar to every macro state
within a run. Do not deploy it for macro action ranking or as a combat
boundary critic. A larger optimizer run on identical targets is
unmotivated.

**Training target coverage defect:** Original H12 v1 records the *first*
noncombat public state after combat. Rewards are usually selected in
subsequent noncombat decisions. Consequently the original labels show
**0 new named potions and 0 relic changes** across all 6,845 fights,
with only 108 training / 23 heldout deck changes. This is a property
of the observation boundary, not evidence that a real Overgrowth run has
no such rewards. v1 was learning a mostly-degenerate immediate-exit
resource distribution. The original v1 combat survival/exit-HP results
remain valid for that immediate boundary.

## H12.1 opt-in resource horizon correction

Existing `train.jsonl` and `heldout.jsonl` preserve all public
noncombat states with observation indices. We can reconstruct the
**last same-floor public resource state** after each survived combat,
stopping before the next combat or the next floor. This is a measured
post-reward result of the frozen H6-C policy, without querying hidden
RNG or inventing counterfactual outcomes.

`--resource-boundary postreward` trains **combat survival and immediate
exit HP from the original combat-exit observation**, while using this
post-reward snapshot for gold, deck, relic, held-potion retention, and
potion acquisition events. The original `immediate` mode remains the
default and retains its meaning. The new resource boundary is pinned
in the training identity and the checkpoint/report metadata.

**First audit the old corpora without fitting or writing new files**:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
python -u tools/train_h12_joint_outcomes.py \
  --train results/h12-joint/train.jsonl \
  --heldout results/h12-joint/heldout.jsonl \
  --checkpoint results/h12-joint/outcome-postreward.pt \
  --report results/h12-joint/heldout-postreward-report.json \
  --resource-boundary postreward
```

The `[h12-audit]` row prints training/heldout `new_named_potion`,
`changed_deck`, `changed_relics`, and `settlement_immediate_only`
coverage. Verify that the reconstructed labels are nondegenerate
before spending more GPU time. This audit uses the original 1,280
full-run seeds; **no emulator recollection is necessary**.

If resource labels are genuinely present and the settled-room
contract matches native trajectories, an optional outcome fit is:

```fish
python -u tools/train_h12_joint_outcomes.py \
  --train results/h12-joint/train.jsonl \
  --heldout results/h12-joint/heldout.jsonl \
  --checkpoint results/h12-joint/outcome-postreward.pt \
  --report results/h12-joint/heldout-postreward-report.json \
  --resource-boundary postreward \
  --epochs 30 --batch-size 256 --device cuda --execute \
  2>&1 | tee results/h12-joint/postreward-fit.log
```

Do not interpret an improvement on the already-inspected H12 heldout
cohort as a fresh unbiased model-selection result. After deciding on
the boundary and architecture, collect a new independent validation
cohort for the next final comparison. Do not deploy the continuation
head until its target and representation are redesigned.


## Reward-settlement audit and next fit

The user's `--resource-boundary postreward` dry run, with the original
1,024/256 source runs, produced:

| Resource target | Training (5,445) | Heldout (1,400) |
|---|---:|---:|
| `settlement_settled` | 4,490 | 1,168 |
| `settlement_defeat_immediate` | 955 | 232 |
| Deck changed | 4,487 | 1,166 |
| Relics changed | 465 | 125 |
| New named potion | 1,887 | 518 |

There are no `settlement_immediate_only` surviving encounters. This
confirms the original public journals are sufficient for postreward
resource labels. The near-universal deck changes **conditional on
survival** (4,487/4,490 and 1,166/1,168) require caution:
predicting whether a card was added is nearly constant under the
frozen H6-C reward-choice policy. The current heads model event
indicators, not new card/relic/potion identities.

The `--outcomes-only` flag skips the previously unsuccessful
continuation critic and its 43,135 public macro-state tensors. The
new outcome evaluator reports for deck/relic/potion gain both
unconditional and survival-conditioned Brier scores, each versus
the best constant-rate baseline. Conditional results help distinguish
genuinely learned resource predictions from the trivial fact that
a dead player receives no rewards. Heldout baselines are optimistic,
computed using heldout prevalence, so beating them is meaningful;
treat this previously inspected heldout cohort as *development data*.

After `git pull --ff-only`, run:

```fish
python -u tools/train_h12_joint_outcomes.py \
  --train results/h12-joint/train.jsonl \
  --heldout results/h12-joint/heldout.jsonl \
  --checkpoint results/h12-joint/outcome-postreward.pt \
  --report results/h12-joint/heldout-postreward-report.json \
  --resource-boundary postreward --outcomes-only \
  --epochs 30 --batch-size 256 --device cuda --execute \
  2>&1 | tee results/h12-joint/postreward-fit.log
```

Preserve the original immediate-exit H12 model and its report. A
new independent final evaluation cohort is required before accepting
new model-selection claims. Neither outcome-only training nor the
postreward supervision permits counterfactual reward inference.
