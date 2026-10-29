# H12: policy-conditioned native joint combat outcomes

## Status

Implementation on `research/h12-native-joint-outcomes-20261029`, **based on
H11's unmerged shared-Transformer branch**. This is the first usable M3 data and
supervision stage, NOT an M4 combat-policy update and NOT yet a deployed macro
lookahead policy. The collection and training tools require a local native
emulator acceptance run before quantitative claims can be made.

### Baseline decision

H11 R40 learned only its macro policy/value heads from a combat-initialized
shared model; its common encoder stayed frozen. On 2,560 native A10 training
runs (40 × 64), it recorded 3 Act 1 clears. The independent 128-seed paired
heldout showed:

| Frozen H9 combat, macro actor | Act 1 clears | Mean frontier |
|---|---:|---:|
| H6-C (reference) | 10 / 128 | 11.0955 |
| H11 R40 | 0 / 128 | 7.4178 |

H11 remains a reproducible negative result; do not resume identical R40 PPO.
For H12, **H6-C + H9** is the behavior policy. The new H11 shared macro and
H9 combat checkpoint formats are untouched.

## M3 components implemented

1. **Public-frame hook:** optional `public_frame_observer` in `play_run`.
   It observes every published fair-player state, including terminal states.
   All other users of `play_run` retain the existing behavior.
2. **Native joint-outcome journal:** `tools/collect_h12_joint_outcomes.py`.
   Uses 15 long-lived CPU-worker backends, a pinned H6-C macro checkpoint,
   and frozen H9 combat. Every full-run source seed owns all its entries.
   Captures first **public** combat entry and first public exit or terminal
   defeat. Records joint HP, named potion multiset, gold, deck, and relic
   changes; macro decision history and subsequent full-run continuation target.
   Censored combat exits have **no** fabricated transition labels.
   Existing journal records are durable after an interrupted collection;
   `--resume` refuses changed policy/emulator fingerprints.
3. **Shared-latent probabilistic model:** 
   `src/sts2_ai/models/h12_joint_outcome.py`.
   An 8-component conditional mixture jointly models survival, exit HP,
   named *held* potion retention, gold delta, and deck/relic change events.
   The shared component induces statistical correlation across resource
   outcomes. A separate public-state continuation critic is fitted to
   native full-run target outcomes.
4. **Source-seed-disjoint fit/evaluation:**
   `tools/train_h12_joint_outcomes.py` checks train/heldout seed prefixes,
   source-run sets, both frozen model SHA-256 hashes, and emulator/DLL
   fingerprint. It trains without using heldout examples for updates or
   early stopping. Its report contains survival Brier score and reliability
   bins, HP fraction MAE, joint negative log likelihood, named potion
   retention loss, and continuation MSE versus a constant baseline.

The published-state feature projection is an allowlist. It excludes emulator
handles, hidden RNG state, deck/draw order, map-node identifiers, run seed
and exact-state hashes. The full-run seed stays only in provenance metadata.

**Current scope boundary:** The model records exact newly acquired
potion/card/relic *names* in its corpus, but its generative distribution only
models retention of named potions already held at combat entry; for new
potions it predicts the gain-event probability. It cannot yet sample an exact
new item identity or perform credible combat counterfactual planning.
The continuation critic evaluates states visited under H6-C + H9; it is not
automatically a value for every alternative action.

## First local execution (fish / WSL)

Work on the **AI branch**; do not switch or modify the emulator feature branch.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch --track origin/research/h12-native-joint-outcomes-20261029
source .venv/bin/activate.fish

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set combat results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt

# Supply the same EXISTING H6-C macro JSON model used in your
# H11 paired evaluation; no incompatible checkpoint conversion.
read -P "Path to H6-C macro model: " macro

python -m compileall -q \
    src/sts2_ai/models/h12_joint_outcome.py \
    src/sts2_ai/training/h12_transition_data.py \
    tools/collect_h12_joint_outcomes.py \
    tools/train_h12_joint_outcomes.py
pytest -q tests/test_h12_joint_outcome.py \
    tests/test_shared_transformer_h7.py

# TRAIN: 1,024 independent complete native A10 source seeds
python -u tools/collect_h12_joint_outcomes.py \
    --baseline-macro-model $macro \
    --combat-checkpoint $combat \
    --emulator-root $emu --emulator-revision $rev \
    --emulator-dll $dll --output results/h12-joint/train.jsonl \
    --seed-prefix h12-a10-joint-train-v1 \
    --runs 1024 --workers 15 --execute \
    2>&1 | tee results/h12-joint/train.log

# HELDOUT: independent 256 full-run seeds
python -u tools/collect_h12_joint_outcomes.py \
    --baseline-macro-model $macro \
    --combat-checkpoint $combat \
    --emulator-root $emu --emulator-revision $rev \
    --emulator-dll $dll --output results/h12-joint/heldout.jsonl \
    --seed-prefix h12-a10-joint-heldout-v1 \
    --runs 256 --workers 15 --execute \
    2>&1 | tee results/h12-joint/heldout.log

python -u tools/train_h12_joint_outcomes.py \
    --train results/h12-joint/train.jsonl \
    --heldout results/h12-joint/heldout.jsonl \
    --checkpoint results/h12-joint/outcome-model.pt \
    --report results/h12-joint/heldout-report.json \
    --epochs 30 --batch-size 256 --device cuda --execute \
    2>&1 | tee results/h12-joint/fit.log
```

Resume a partially completed collection with the **identical arguments** plus
`--resume` and append the log via `tee -a`. A completed fit refuses to
overwrite an existing model/report.

## Checkpoints for next stage

1. Inspect actual H12 heldout Brier/calibration/HP/potion outcomes, and
   whether continuation value improves on the constant baseline.
2. Implement exact *publicly deterministic* macro transitions for candidate
   scoring. Do not query hidden game-state branch outcomes as if observed.
3. Calibrate joint outcome sampling (including new named resource identities)
   and integrate the frozen continuation critic into macro lookahead.
4. Only then begin M4 alternating combat updates using a pinned target macro
   critic at the post-combat **public** continuation boundary. The H9 combat
   checkpoints remain frozen throughout H12 collection/fitting.
