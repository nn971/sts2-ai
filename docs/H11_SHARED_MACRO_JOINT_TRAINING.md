# H11: shared Transformer macro head, frozen H9 combat (M2)

This is the next stage of `docs/H10_ACT1_JOINT_TRAINING_PLAN.md`,
following the completed 40-round H10 native-combat PPO experiment.

## Why proceed

H10 R40 and H9 R250 are statistically indistinguishable on the 173
held-out native combat entries. A small fine-tuning experiment is not a
requirement for a large performance gain. Preserve H9 R250 as the combat
reference and move to native full-run macro learning now.

## Implemented on this branch

- The H7 14.66M-parameter Transformer already has distinct combat and macro
  policy/value heads. No checkpoint migration or distillation is performed.
- The new `shared_macro_objects.py` codec encodes public named options for
  native shop, event, reward, deck-selection and candidate map actions.
  Raw offer/node/card-instance IDs are used only for joining to public objects.
  Each chosen map node carries a candidate-owned, depth-labelled public
  reachable-room multiset. Ambiguous same-token alternatives fail loudly.
- `tools/train_h11_shared_macro.py` is a first native A10 *full-run* PPO
  implementation. The frozen H9 combat head, object embeddings, encoder,
  attention and shared parameters remain fixed. Only
  `policy_heads[1]` and `value_heads[1]` are trained. The actor samples
  macro actions while combat is greedy. One source seed per fresh complete
  game; 15 ordinary emulator workers; per-round SHA-pinned actor; strict
  native PPO provenance audits, optimizer checkpoint and deterministic
  seed schedule. Censored runs do not enter updates.
- Macro training optimizes real certified Act-1 clears, with small defeat-floor
  progress shaping (at most 0.1 on the critic scale). No fake Act-clear
  claims. This is initial sparse-return learning, not yet M3 outcome modeling
  or an unlocked shared-encoder joint update.
- `tools/evaluate_h11_full_runs.py` performs a source-run-disjoint paired
  native Act-clear evaluation: H6-C macro + frozen H9 R250 combat versus
  trained H11 macro + *identical frozen H9 R250 combat*. It checks that
  every non-macro model tensor remains bitwise identical to H9. Individual
  source-run results have a fingerprinted resumable journal.

## Local acceptance (fish)

Use the same compatible emulator used for the H10 held-out paired comparison
(`2c46c703a30875d6ef84fcbe0a0511fec9c74dcd`). No emulator update or
native corpus recollection is required. Preserve existing H10 results.

```fish
set emu ~/projects/sts2-emulator
set rev 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd
test (git -C $emu rev-parse HEAD) = $rev; or exit 1

cd ~/projects/sts2-ai
git fetch origin
git switch --track origin/research/h11-joint-macro-head-20261029
source .venv/bin/activate.fish

python -m compileall -q tools/train_h11_shared_macro.py tools/evaluate_h11_full_runs.py
pytest -q tests/test_h11_shared_macro_objects.py tests/test_shared_transformer_h7.py

set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set combat results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt

# One end-to-end functional check; not a performance-selection experiment.
python -u tools/train_h11_shared_macro.py \
    --initialize-from-model $combat \
    --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
    --output-root results/h11-shared-macro-smoke \
    --rounds 1 --episodes-per-round 8 --workers 4 \
    --device cuda --execute
```

This new macro codec is intentionally strict: encountering an unsupported
same-kind menu, unresolved public target or legitimate new native action
semantic should generate a source-specific failure and a narrow codec fix.
Do not suppress or skip failing game seeds.

## Full M2 cohort

```fish
cd ~/projects/sts2-ai
python -u tools/train_h11_shared_macro.py \
    --initialize-from-model $combat \
    --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
    --output-root results/h11-shared-macro-r40 \
    --rounds 40 --episodes-per-round 64 --workers 15 \
    --temperature 0.85 --lr 0.00003 \
    --checkpoint-every 5 --device cuda --execute \
    2>&1 | tee results/h11-shared-macro-r40.log
```

Training can be continued after interruption with the *identical* command
plus `--resume`. A changed parameter, emulator binary or model checkpoint
changes the run fingerprint and requires a new experiment.

## Independent full-run confirmation

```fish
python -u tools/evaluate_h11_full_runs.py \
    --baseline-macro-model results/nature/h6c-macro-attention-r80.json \
    --combat-checkpoint $combat \
    --macro-checkpoint results/h11-shared-macro-r40/models/player-r0040.pt \
    --emulator-root $emu --emulator-revision $rev \
    --seed-prefix h11-shared-macro-a10-heldout-v1 \
    --runs 128 --workers 15 \
    --report results/h11-shared-macro-r40/paired-heldout.json
```

The comparator reports certified full-run Act-1 clear and censoring, not just
one-shot combat scores. Do not judge this initial macro model solely by its
first small training update.

## Subsequent stages

M3: collect policy-versioned *joint* combat outcomes (victory, exit HP, named
potions, deck/relic/gold changes) from H11 macro-generated native encounters;
train/calibrate a distributional outcome predictor and a macro continuation
critic. Preserve correlation and source-run-disjoint splits. Do not pretend
to evaluate counterfactual tactics from a single sampled outcome.

M4: alternate frozen-combat macro PPO, combat-entry PPO with the pinned
macro continuation target, and outcome predictor refresh; only then
selectively unfreeze shared encoder weights with anti-regression gates.
Maintain independent held-out Act-clear checks and record decisions
throughout. Neither M3 nor M4 is implemented by this first H11 branch.

## Validation notes

GitHub `parent-ci` currently includes pre-existing Ruff violations outside
these new files. Local focused tests and an end-to-end native smoke are
required before a long run. The source code has been committed, but the
author has not executed it against the user's local CUDA/.NET environment.
