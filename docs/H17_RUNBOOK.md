# H17 execution: adaptive full-combat PUCT and joint macro learning

Implementation in draft PR #157, stacked on H16 diagnostic PR #156.
Plan: \`docs/H17_TRAINING_PLAN.md\`.

## What is implemented

- Initial R79 joint H16 checkpoint; architecture unchanged.
- 0/16/64 **full-combat PUCT**, chosen per eligible root from already
  computed logits, HP, floor and public card-type exposure. Random audit
  supplies search even where policy is confident. At zero H6C weight
  the frozen teacher is not evaluated.
- 64 **fresh** full-Act runs per round by default, with 15 worker
  emulators, fused steps and one shared CUDA inference actor.
- H6C macro prior decays from 0.5 to 0 over 120 H17 rounds. H16's
  joint macro policy remains trainable with same macro policy-gradient loss.
- Geometric future-HP λ starts at 0.411395 and grows linearly to 0.77
  over 200 H17 rounds. Search labels preserve combat-victory dominance.
- Card exposure counts by type (deck, playable, played, searched) persisted
  in every round. Coverage-informed adaptive search.
- Up to 8192 public-state replay examples from completed runs, <=32 rounds
  old, gzipped. Each new round samples <=256 examples from **prior rounds**,
  balanced between macro/combat, with rare cards and late floors preferred.
  Replay updates only the value/Act heads and previously measured PUCT
  visit-count cross-entropy. **No stale REINFORCE.**
- One replay archive snapshot per completed round; resume loads the archive
  with the same progress fingerprint and matching round. Old archives are
  pruned after progress is durably committed.
- Every full-combat simulation must terminate before the step watchdog,
  otherwise the run aborts; no fabricated terminal reward.
- Existing H16 CLI and default fingerprint remain unchanged when H17 is off.

## Commands (fish shell)

Update the stacked branch:

\`\`\`fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h17-adaptive-fullcombat-training-20261102
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set macro results/nature/h6c-macro-attention-r80.json
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
\`\`\`

Do not switch the emulator away from ongoing fidelity work merely for this
experiment; pass the exact SHA of the emulator checkout used for this run
and ensure the experimental fixed-route Act1 profile still works.

Check source and CLI:

\`\`\`fish
python -m pytest -q tests/test_h17_adaptive_training.py \
    tests/test_h16_joint_act_pilot.py tests/test_h16_full_combat.py

python tools/train_h17_adaptive.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --output-root results/h17-preflight
\`\`\`

The final command prints config/fingerprint only (no \`--execute\`).

**Smoke:** use separate seed and output root. Uses 2 rounds x 16 fresh Act
runs per round and 4 policy-only heldout runs; it exercises actual PUCT,
the frozen teacher, horizon, replay write/load and metrics.

\`\`\`fish
set smoke results/h17-adaptive-fullcombat-smoke
python tools/train_h17_adaptive.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --output-root $smoke \
    --rounds 2 --runs-per-round 16 --heldout-runs 4 \
    --study-seed 20261118 --execute
\`\`\`

**Long run:** begin from R79 again in a different output directory;
64 new native Act1 runs x 40 rounds. The option \`--initial-h16\` defaults
to \`$initial\` in the H17 entrypoint and can be overridden explicitly.

\`\`\`fish
set run results/h17-adaptive-fullcombat-r40
python tools/train_h17_adaptive.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --output-root $run \
    --rounds 40 --execute
\`\`\`

Resume the *same* run with **identical** options and \`--resume\`:

\`\`\`fish
python tools/train_h17_adaptive.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --output-root $run \
    --rounds 40 --resume --execute
\`\`\`

The progress report \`$run/progress.json\` has per-round:
\`train.search_budget_roots\`, \`train.search_transitions\`,
\`timing.collection_seconds\`, \`timing.update_seconds\`,
\`timing.approx_emulator_transitions_per_second\`,
\`card_coverage_summary\`, \`macro_prior_weight\`,
\`horizon_lambda\`, \`replay\`, \`replay_buffer_rows\`, and
policy-only \`heldout\` floor/Act-clear metrics.

## Acceptance gates

1. Smoke terminates without censored episodes or incomplete full-combat
   watchdogs; H16/H17 focused CI passes.
2. Search mix has both 16 and (over enough roots) 64 budgets. Observe
   actual ratios, rather than assuming exact 83/15/2%.
3. No duplicated/stale replay samples; replay samples in round 2 derive
   from completed round-1 runs; replay and coverage resume identically.
4. Confirm throughput is affordable. Prioritize \`fresh runs/hour\`,
   \`PUCT-labeled states/hour\` and \`new card types encountered/hour\`.
   If costly, reduce \`--h17-search-p16\` and
   \`--h17-search-p64\` **in a new output directory** (fingerprint changes).
5. Larger independent policy-only evaluation at the first meaningful
   checkpoint (e.g. R10 or R20) before committing to 150–200 rounds.

## Current limitations

- Fixed-route Overgrowth only. Variable map routes are gated on native
  emulator fidelity.
- Replay resamples **observed** card types; it cannot manufacture exposure
  to unseen cards. Synthetic seeded decks/continuation states should only
  be introduced via a separately verified, distributionally controlled
  native-state sampler.
- Search is root-conditioned exact-emulator forking and may expose hidden
  information to the teacher. The student encoder uses public observations.
- If a smoke is interrupted *inside* a round, H16 resumes from the last
  completed round; incomplete-round experience is discarded. Progress,
  optimizer and replay state are checkpointed at completed rounds.
