# H13 M1 — public encounter law and combat/reward boundary

**Base:** H12 PR #140, with the H6-C macro policy and H9 combat policy
unchanged. This is a data and decision-interface milestone, **not yet a
deployed macro improvement**. The standalone H12 survival/HP predictor remains
a reference checkpoint; the failed H12 continuation critic remains disabled.

## Verified structural issue

The H12 model is conditioned on a *realized* combat-entry public observation,
including enemy identities, turn-one intents and opening hand. At an earlier
map choice, only the visible node type, destination floor, current deck and
persistent resources are available. A call to H12 with the eventual enemy
state would leak future information, and copying a different run's hand would
create an invalid counterfactual.

H12's "postreward" resource target also includes choices made by frozen H6-C.
It measures an on-policy observed continuation, rather than a policy-independent
combat effect or set of offered rewards.

## What the H13 code implements

- `h13_public_transitions.public_choice_context` records the *selected*
  visible map room type, act, and destination floor, omitting the ephemeral
  node ID. For reward decisions it records an explicit allowlist of published
  card, potion, relic and extra reward offers. It reads only the already
  published fair observation.
- `h13_public_transitions.observed_combat_boundary` exposes survival,
  immediate HP/gold deltas, named potion consumption and any immediate
  card/relic changes, separately from later public reward-choice records.
  The separate last-same-floor `postcombat_snapshot` retains its explicitly
  H6-C-conditioned interpretation.
- `observed_room_encounters` joins each **chosen** room with combats
  subsequently observed before the next map choice, retaining no-fight
  outcomes. It sees actual enemies only *offline* after the run.
- `EmpiricalEncounterLaw` groups enemy-composition outcomes by
  (act, public room type, 5-floor band), and abstains when the observed
  count is below `min_count`. This captures empirical encounter uncertainty
  without inventing an opening hand.
- `tools/build_h13_encounter_law.py` reads a complete native full-run
  journal, requires policy/emulator identity consistency, distinct run seeds,
  and H13 choice metadata; writes a new immutable, source-hashed catalog.
  The catalog carries no source seeds, map node IDs, RNG state, or deck order.

**Important distinction:** the catalog is an observational H6-C/H9 encounter
law. Different paths are selected in different player states; empirical room
frequencies may be confounded by route choice. It is a candidate prior for a
future planner and must be validated, not treated as the true independent
game-mechanics encounter law.

## Producing an H13 catalog (fish; native emulator)

The H12 1,024/256 journals lack selected map destination type and cannot be
used to infer it reliably from the saved fields. Preserve those original
artifacts; collect a new H13-enriched journal with a **new source prefix**.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch --track origin/research/h13-public-encounter-boundary-20261029
source .venv/bin/activate.fish

set emu ~/projects/sts2-emulator
set rev 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set macro results/nature/h6c-macro-attention-r80.json
set combat results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt

# Confirm the pinned emulator source and DLL from the original H12 run.
test (git -C $emu rev-parse HEAD) = $rev; or exit 1

python -m pytest -q tests/test_h13_public_transitions.py tests/test_h12_joint_outcome.py

# Initial native cohort, 256 complete full runs; all public frames preserved.
mkdir -p results/h13-encounters
python -u tools/collect_h12_joint_outcomes.py \
  --baseline-macro-model $macro --combat-checkpoint $combat \
  --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
  --seed-prefix h13-a10-public-encounters-train-v1 \
  --runs 256 --workers 15 --max-decisions 4096 \
  --output results/h13-encounters/train.jsonl --execute \
  2>&1 | tee results/h13-encounters/collect.log

python -u tools/build_h13_encounter_law.py \
  --journal results/h13-encounters/train.jsonl \
  --output results/h13-encounters/catalog.json \
  --min-count 30 \
  2>&1 | tee results/h13-encounters/catalog.log
```

The 256-run cohort is an **implementation/coverage pilot**, not a fresh
model-selection or win-rate evaluation. If room/floor-band support is too
sparse, use a new catalog filename and extend collection with the same
fingerprinted `--resume` arguments and `--start-index 256` for additional
runs; do not merge test runs with the inspected old H12 heldout cohort.

## Next integrated policy milestone

1. Implement an independent **public-state combat-entry sampler** using
   `sts2-emulator`'s interchangeable stochastic provider or an explicitly
   verified ordinary encounter/draw law. Given a currently visible map
   alternative and player resources, it must generate **complete**
   prospective combat entries, including enemies, intents, powers and opening
   hand. Hidden native encounter RNG and the eventual realized hand are
   excluded. This requires checking which initial enemy properties and
   draw effects can be produced from public state.
2. Generate scenarios for every *legal* candidate, estimate policy-conditioned
   survival and **survival-conditional** exit HP using frozen H12, and handle
   route rewards via the **visible** offered-card/relic/potion mechanics after
   arrival. Unknown events/offer contents stay probabilistic until revealed.
3. Build a constrained H6-C wrapper that overrides a map choice only when
   the scenario law has support, its estimate is stable, and a predeclared
   risk/benefit score justifies the change. Otherwise fall back to H6-C.
   Never substitute H12's nearly constant deck-change head or the failed
   continuation critic for actual offered identities.
4. Evaluate baseline vs wrapper on a *new*, independent, identical A10
   source-seed cohort (15 emulator workers). Record Act 1 clears, progress,
   HP, intervention counts, model inferences, failure modes and paired
   uncertainty intervals; reject regressions. A valid catalog or successful
   unit tests alone do not establish policy improvement.

## Constraints and provenance

Use the existing H12 H6-C/H9 checkpoint files and pinned emulator revision,
with no extra worktrees or changes to the emulator feature branch. H13 adds
context to newly collected H12-style journals and marks those records with
`public_choice_context_version`; the original H12 source journals and
frozen checkpoint/report are untouched. The H13 catalog builder intentionally
rejects the old records.

No actual native H13 collection or paired evaluation has been performed in
this PR; native execution requires the local DLL and checkpoints.
