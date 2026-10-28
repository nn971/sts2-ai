# H9 rich Overgrowth v2 — continuation curriculum

Status: **prepared, native-playability validation pending**. This is a new
training distribution, not a revision of frozen H9 R80 or its original
seven-encounter zero-shot benchmark.

## Goals

The original H9 training distribution has seven encounters and six additional
cards. H10's 74 native entries exposed weaker generalization, especially
on multi-enemy fights and familiar starting decks.

The opt-in `--curriculum h9-rich` distribution provides:

- 21 explicitly listed Overgrowth encounters: 4 weak, 11 normal, 3 elite,
  3 boss. Includes Slimes, Inklets, Ruby Raiders, and The Kin.
- Deterministic tier mixture: 30% weak, 40% normal, 15% elite, 15% boss.
  These are research weights, **not** actual native encounter frequencies.
- 12–24 card decks varying starter retention and six broad card archetypes;
  27 possible additions, with zero/one upgrade per card.
- One to three relics including Silent starter relic, zero to two named potions,
  and tier-aware initial HP.
- 1,024 training configurations and 256 disjoint structural zero-shot
  configurations; independent source seeds determine chance-law worlds.

The original `silent-a10-structured-combat-v1` remains unchanged.
The new fixture ID is `silent-a10-overgrowth-rich-v2`, with source
identifiers `h9-rich-<split>-v2-...`. Metadata and model hashes must be
pinned for reproducibility. The frozen H9 R80 model can initialize new PPO
training; the optimizer is reset when the curriculum changes.

## Branches

- AI: `research/h9-rich-overgrowth-curriculum-20261028` based on H10 #134.
- Emulator: `research/h9-rich-overgrowth-fixtures-20261028` based on emulator
  PR #221. **Both** branches must be checked out together.
- Emulator expands only opt-in v2 fixture validation. It preserves the
  original C# v1 generator and source RNG providers.

## Native opening unblock (October 28)

The first local C# run reported five `UNSUPPORTED_SEMANTIC_OPENING
site=act1.fixed_single_formation` errors for SlimesWeak, InkletsNormal,
RubyRaidersNormal, PhrogParasiteElite and TheKinBoss. This was an omitted
routing integration: the original isolated research opening recognizes only
the seven original H9 encounters.

Emulator PR #222 now sends **only rich v2** into source-shaped native
formation resolution already used by full-run fidelity: weak Slimes uses its
three encounter-local RNG calls; normal formations use the native resolver;
elite and boss fixtures reuse registered fixed source formations including
named slots and leader references. Original v1 entry behavior remains
separate.

The next local run (13/16 passing) exposed two more integration problems:
the prototype six-floor map has guaranteed Elite nodes on floor 3 rather than
floor 2, and enemy starting powers had duplicated subscriber application orders
across multiple enemies. The emulator now uses Elite room identity and
floor 3 (row 2) for rich fixtures, plus global application-order assignment
for rich initial enemy powers; the Python source generator has the matching
elite floor metadata. Opening tests now assert room identity, floor and
global order uniqueness.

These fixes are pushed, but local rerun and complete-combat acceptance
remain pending. Do not start PPO until the native test passes.

## Python curriculum corrections (October 28)

Initial `tests/test_h9_rich_curriculum.py` run exposed one duplicate
structural specification (indices 564 and 722), and the existing H2
`CombatScenario` rejected rich elite floor 3 / row 2. The rich generator
now deterministically retries duplicate structural draws within their
original encounter tier, preserving index-stable 1,024/256 structural
splits. Rich scenarios use an isolated subclass with an explicit 21-encounter
allowlist; legacy H2 source validation remains unchanged. Additional
regression checks cover random-access stability and the old H2 floor gate.

The source corpus is versioned v2 and the R80 benchmark is unaffected.
Python tests and full native-combat acceptance should still be run locally
before initiating PPO.

## Full-combat acceptance blockers (October 28)

The first 64-source native H8 pilot completed four episodes before workers
reported three distinct execution gaps:

1. Newly offered legal research potions (Swift, Weak, Dexterity, Energy,
   Strength, Explosive) were outside the original three-potion research
   semantic allowlist.
2. The richer card corpus includes Calculated Gamble (controlled draws) and
   Bouncing Flask (random target), absent from the player action gate.
3. Multi-enemy next-turn intents committed into the continuation frame
   without synchronizing the surrounding combat snapshot, violating the
   intentionally strict stochastic continuation invariant.

Emulator PR #222 now admits the explicitly supported source potion effects,
reuses the existing controlled draw and target paths, and synchronizes
committed intent mechanics with the pipeline state. Unsupported-action errors
now include the actual card/potion identifier. The source Chance Law, hidden
deck owner, automatic combat transitions, and journal replay checks remain
enabled. These fixes are committed but **not yet verified by the 64-source
acceptance run**.

Do not discard unsupported actions, count failures as defeats, or begin the
40-round PPO run until the source-complete pilot succeeds. If another legal
card/potion fails, use the reported model ID to address its engine semantics.

## Multi-enemy partial-intent validation follow-up

The native `RichMultiEnemyTurnAndSwiftPotionAdvance(inklets_normal)`
regression reached the second enemy intent, then failed the end-turn
`Run == Origin` check. Partial intent commitment legitimately changes the
active enemy mechanics before the next pending chance ticket. Emulator
commit `955a07a` replaces that obsolete equality check with comparisons
of the original pre-intent checkpoint, replayed intent combat and unchanged
noncombat run state. The stochastic continuation invariant is retained.

Focused emulator tests and the 64-combat acceptance rerun remain pending.

## Native acceptance second wave (October 28)

The next H8 native pilot reached 34/64 completed source workers before
failing on three gameplay events. These are source execution coverage,
not bad curriculum indices:

- **Elite victory:** `reward.elite_relic.model` was omitted from the
  narrowly scoped engine-owned **postcombat** reward-site list. It now
  uses the existing reward stream, with the complete combat result
  preserved and no reward choice exposed to the combat policy.
- **Slimes:** the source-generated `proto.status.slimed` is playable,
  exhausts, and draws one card. It is admitted to the controlled
  card-effect draw path; it remains a genuine player action.
- **Phrog Parasite:** death summons roll HP for new Wrigglers. The
  reference-law, Nature-owned-deck H8 fixture has no native combat
  RNG owner; this site now consumes exactly one scoped and journaled
  `combat.initial_unique_hp` selection from the existing prototype
  combat stream per death summon. Unrelated guarded chance still
  fails closed. The recorded provider mode is `legacy-prototype-v0`,
  **not** native RNG parity. An eventual stochastic-provider migration
  should give the newly spawned creatures a full typed RNG owner.

Latest emulator feature branch also includes a focused scope-isolation
test. The 64-combat complete native acceptance and replay are **still
pending** on a locally rebuilt DLL. Do not silently drop winning
elite combats or Slimed actions from the curriculum.

## Rich native acceptance: 59/64 terminal

The full 64-source pilot no longer crashes during action execution. It
reported `terminal=59, unsupported=5` in about 19 seconds, then failed
`--require-all-terminal` *before saving the cohort JSON*. This is an H8
unclassified-boundary issue; the existing log does not identify which five
sources or which pending mechanics the runtime could not classify.

AI PR #135 now retains all source trajectories in the requested report
file, prints the failed source/encounter/boundary and last six recorded
actions, then raises the same strict acceptance exception. This diagnostic
change does not alter player policy, chance sampling, emulator mechanics,
or the all-terminal acceptance criterion.

Rerun exactly the same pilot command on the updated AI branch and read
`[h9-rich-incomplete]` lines, or inspect the saved cohort's
`trajectories` entries with `status != "terminal"`. Do not assume all
five share one cause or narrow the curriculum before identifying them.

## Five unsupported boundaries diagnosed from preserved 64-source report

Analysis of the preserved `h9-rich-v2-acceptance.json` reveals that all five
incomplete sources stopped **immediately after playing Bouncing Flask**:
configuration indices 7 (Inklets), 23 (Slimes), 38 (Vantom, upgraded Flask),
44 (Shrinker Beetle), and 62 (Nibbits). The previous report preserved
replay-verified journals and showed all five as `unsupported`, with no
worker exception and no detected replay divergence.

Bouncing Flask already has source-native controlled **random enemy target**
mechanics in `SemanticTargetChoiceV111`. H8 formerly exposed only physical
draws and enemy intents as Nature sites, so these target continuations
fell through to `unsupported`. The paired emulator and AI branches now
expose a versioned `random_target` Chance Law and explicit resolve action,
and teach the local reference provider and external H8 pilot to consume this
site. Every selected target is retained in the H8 decision journal and
checked during deterministic replay; no scripted card-move label or combat
solver was added.

A focused C# regression plays Bouncing Flask against Vantom and Inklets
and checks target choice and complete journal replay. The old H9 source
distribution is unchanged. **Build and 64-source acceptance are still
pending**; do not interpret the previous 59/64 as the result of this fix.

## Verified 64-combat native acceptance (2026-10-28)

The user-supplied `h9-rich-v2-acceptance(1).json` confirms the
Bouncing Flask random-target fix on emulator revision
`7a69ce40cb6c34bf2082092d2cbe8bfe970a4378`, using the frozen
14,659,972-parameter H9 R80 policy, 15 workers, packed CUDA actor,
and H8 local reference Nature. This supersedes the earlier 59/64
unsupported-boundary result.

- **64/64 terminal combats, 64/64 replay verified, zero unsupported**.
- **51 victories, 13 defeats**: 79.69% on this particular synthetic cohort.
- **16 random-target Nature selections** across five encounters; all five
  original previously unsupported source indices (7, 23, 38, 44, 62)
  reached terminal outcomes and replayed successfully.
- Weak: 22/22 victories; normal: 23/24; elite: 4/9; boss: 2/9.
- Wall time 19.596 s, 2,086 player decisions (106.45/s),
  3,490 total boundaries (178.09/s), 3.27 terminal fights/s.
- Twenty of 21 allowed encounters are represented in these 64 samples;
  `slithering_strangler_normal` is absent by chance.
- This acceptance used `capture_player_training=false`; it validates
  mechanics/replay/inference but does not independently validate the
  PPO action-capture and optimizer step. The H8 training backend uses
  the same cohort function with capture enabled.
- Evaluation using the `selection` split changes seeds but shares
  structural indices with the 1,024 training fixtures. True structural
  zero-shot evaluation must use the separate 256 `zero-shot` fixtures.

**Decision:** start a new H9 rich PPO run from the immutable H9 R80
checkpoint without changing the curriculum or adding further pilot
benchmarks. Keep the original structured-v1 distribution as a separate
regression. Inspect training progress and independent native Act 1
results after the run; no claim of improved generalization yet.

## Required acceptance on local checkout

1. Run Python structural tests and focused C# research-fixture tests.
2. Build emulator Release, verify C# fixture admission and native opening of
   multi-enemy weak/normal, elite and boss encounters.
3. Run complete terminal/replay acceptance using the H8 native pilot with
   `--curriculum h9-rich --require-all-terminal`. Treat unsupported mechanics,
   missing legal actions, or replay drift as failures; do not relabel them
   as defeats or silently omit configurations.
4. Only after native acceptance, initialize a **new run directory** from
   H9 R80 and execute richer 15-worker PPO training. Retain the H9 old
   distribution as an independent regression benchmark.
5. Evaluate fresh policies on held-out rich v2 configurations and independent
   full native Act 1 H10 combat entries. Neither is a substitute for complete
   Act-clear evaluations.

## Smoke command (fish)

```fish
cd ~/projects/sts2-emulator
git fetch origin
git switch research/h9-rich-overgrowth-fixtures-20261028
git pull --ff-only
dotnet test tests/Sts2Emulator.Core.Tests -c Release --filter FullyQualifiedName~ResearchCombatFixtureV111Tests
dotnet build src/Sts2Emulator.Cli -c Release

cd ~/projects/sts2-ai
git fetch origin
git switch research/h9-rich-overgrowth-curriculum-20261028
git pull --ff-only
source .venv/bin/activate.fish
python -m pytest -q tests/test_h9_rich_curriculum.py tests/test_h8_structured_curriculum.py

set emu ~/projects/sts2-emulator
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emu rev-parse HEAD)

python -u tools/benchmark_h8_h7_policy_pilot.py \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev --curriculum h9-rich \
  --structured-curriculum-seed 20261028 \
  --player-checkpoint results/h9-structured-a10-r80/models/player-r0080.pt \
  --order native --episodes 64 --workers 15 \
  --actor-device cuda --actor-collation packed \
  --local-reference-nature --inline-player-frame \
  --require-all-terminal --replay-every 1 \
  --output results/h9-rich-v2-acceptance.json
```

Following successful acceptance, start a new training run, for example:

```fish
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev \
  --output-root results/h9-overgrowth-rich-v2-r40 \
  --initialize-from-model results/h9-structured-a10-r80/models/player-r0080.pt \
  --curriculum h9-rich --structured-curriculum-seed 20261028 \
  --rollout-backend h8-native --actor-backend shared \
  --actor-device cuda --actor-collation packed --local-reference-nature \
  --device cuda --lr 0.00003 --rounds 40 --episodes-per-round 256 \
  --workers 15 --eval-every 5 --eval-episodes 256 \
  --train-terminal-replay-every 16 --max-boundaries 1200 --execute
```

The 40-round choice is a proposed initial training allocation, not a claim
that this configuration has been validated or that H9 improves over H6.

## Limitations

This is a synthetic, declared mixture, not a corpus of naturally reachable
deck–encounter correlations. Native research fixtures expose configured
opening states without granting hidden future chance outcomes to the actor.
All multi-enemy gameplay and broadened card/relic mechanics still require
real emulator acceptance. The H9 action codec has an object cap of 256 and
action cap of 128; a cap violation is a coverage failure.
