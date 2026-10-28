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
