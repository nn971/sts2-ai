# H8: massive, structured A10 Silent reference-law combat training

**Status:** Isolated implementation proposal (2026-10-27), based on H7's
14.7M-parameter shared-object Transformer. No distillation; reference-only
Chance Law PPO; 15 existing process workers. Changes are additive and the
H5/H7 fixed 16-cell experiment remains the default.

## Data and evidence

The emulator's canonical contract is
`ResearchCombatFixtureV111` (emulator #181, isolated research branch).
`research_combat_fixtures.py` mirrors the C# generator exactly at the
structural recipe level; the semantic JSONL bridge validates the full spec and
constructs persistent cards, relics, potions, upgrades and starting HP before
the first shuffle and hook. Normal player actions and the reference Chance
Law are unchanged.

The v1 research distribution has seven supported single-enemy Act 1
weak/normal/boss encounters; every configuration includes a 20-card Silent
deck, two additional compatible relics, optional supported potions and
bounded initial HP. It is **not** a native game encounter distribution.
A separate configuration ID (not source RNG seed) determines structural
diversity. Data are reproducible by `distribution_id`, `curriculum_seed`,
configuration index and `source_seed`.

- Train corpus: **256 structurally distinct** configurations, indices 0–255.
- Zero-shot corpus: **64 never-trained configurations**, indices 256–319.
- Each round of 256 combats covers every train configuration once with an
  independent source seed; smaller smoke batches rotate through the corpus.
- Selection and confirmation use separate source namespaces on the same
  train configurations. Zero-shot is a disjoint configuration-level holdout.
- Run the 1,000-spec structural census separately from full combat acceptance.
  An audit of specs alone does not prove 1,000 terminal combats are playable.
- Existing H7 PPO optimizer, public object encoder, model checkpoints,
  frozen-actor data collection, rollout logging and 15 workers are reused.
  H5 checkpoints and baseline semantics remain unchanged by default.

## Local launch after emulator branch is built

Both repositories must be checked out on their respective **isolated** H8
research branches. Keep the H6/H7 integration branch and active emulator
Nature/Inspector branches unchanged. In fish shell:

```fish
cd ~/projects/sts2-emulator
dotnet test tests/Sts2Emulator.Core.Tests --filter FullyQualifiedName~ResearchCombatFixtureV111Tests
dotnet build src/Sts2Emulator.Cli -c Release
set dll "$PWD/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll"
set bindings "$PWD/bindings/python"
set rev (git rev-parse HEAD)

cd ~/projects/sts2-ai
python -m pytest -q tests/test_h8_structured_curriculum.py
python -u tools/train_h7_reference_combat.py \
    --curriculum h8 --curriculum-seed 20261027 \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev \
    --output-root results/nature/h8-structured-a10 \
    --device cuda --rounds 80 --episodes-per-round 256 \
    --workers 15 --eval-every 5 --eval-episodes 160 --execute
```

For a *smoke*, use a distinct output directory and override:
`--rounds 1 --episodes-per-round 8 --eval-episodes 8`; additionally use
a small H7 model configuration only for inexpensive integration tests.

To evaluate held-out structural configurations with independent RNG worlds:

```fish
python -u tools/evaluate_h7_reference_combat.py \
    --curriculum h8 --curriculum-seed 20261027 \
    --player h8=results/nature/h8-structured-a10/final-player.pt \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev \
    --split zero-shot --runs 128 --workers 15 \
    --output results/nature/h8-structured-zero-shot.json --execute
```

## Acceptance before large training

1. Compile emulator/bridge with no warnings; exercise invalid specs, replay,
   fixture health and initial state, 100 then 1,000 **complete** semantic
   combats with terminal snapshot replay and reference-only Chance Law.
2. Confirm 15 independent emulator processes and that training receives only
   the fair public observation; check runtime throughput.
3. Review train/held-out configuration sets for silent mechanic failures and
   mark unsupported spec ingredients explicitly. Preserve seeds and build SHA.
4. Run a small 8-episode smoke on H8, then an 80-round run; compare held-out
   win rates, preserved HP and learning curves against the historical H7/H5
   reference baseline. Avoid declaring model improvement on seed-only effects.
5. Once reliable policy performance is achieved, use the *same configuration
   IDs* to train a combat outcome predictor for H6 macro planning.

**Training independence:** original-game Inspector, native-game fidelity and
exact RNG parity are **not** acceptance gates. They remain separate audits.

**Limitations:** v1 is single-enemy and source-supported. The public
formation-position bridge for multi-enemy tactical encoding remains a separate
engineering milestone. This branch has not yet established 1,000 complete
terminal-combat execution results.
