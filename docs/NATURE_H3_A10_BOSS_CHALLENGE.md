# H3 A10 boss stress-test: 20-card Silent decks, seeded combat relics

This is a **research combat fixture**, not a source-native map or reward
distribution. Paired evaluation (on frozen checkpoints) comes **before**
another long joint-training run.

## Implemented setting

- Single-player Silent, Ascension 10, prototype Act 1's final boss
  map node (floor **6**). Native StS2's Act 1 boss node is floor 16;
  this deliberate test-only route shortening is NOT native fidelity.
- Two supported fixed single-enemy bosses: Ceremonial Beast and Vantom;
  multi-enemy Kin is excluded until fair positional target encoding is fixed.
- 20-card archetypes use 12 starter cards + 8 engine-owned extras:
  - `silent-boss-cycle-20-v1`: 2 Acrobatics, 2 Backflip,
    2 Slice, Poisoned Stab, Deadly Poison.
  - `silent-boss-poison-20-v1`: 2 Deadly Poison,
    2 Poisoned Stab, 2 Backflip, Bouncing Flask, Acrobatics.
- Starting HP 70/70, starter Ring of the Snake, 3 base energy,
  2 empty potion slots; add **two distinct seeded relics**
  from Lantern, Ink Bottle, Ornamental Fan, Orichalcum.
  Relics are deterministic for a fixture source seed and its deck
  archetype, stable on replay and **not** claimed to match native
  relic acquisition probabilities.
- The engine constructs persistent physical cards, initializes its
  own opening shuffle and combat/relic hooks, and owns all legal player
  actions. Nature remains draw-only and enemy intents use Chance Law.
  Full checkpoint/replay comparisons verify stable source openings.

## Primary scientific test (evaluate *before* training)

1. Load the **same frozen v8 player** and frozen trained H2 PPO Nature
   checkpoint for each arm, use greedy player decisions (not MCTS).
2. For each source world, score (a) reference physical draw law `p`
   and (b) Nature draw law `q`; use matched encounters, decks,
   selected relics, initial opening and keyed enemy-intent sampler.
3. On `selection` source worlds, check that reference win rate is
   neither close to zero nor one (a target band around 30–80% is
   desirable, not a formal exclusion rule). Look at turns/draws,
   HP scores, player defeats, Nature effect by boss and by deck.
   Do NOT change models or sample selection based on confirmation outcomes.
4. Once the configuration is fixed, run **1,000 disjoint
   `confirmation` worlds**, 250 in each boss/deck cell.
   Report paired source-bootstrap CI for `score(q)-score(p)`.
   Examine the distribution of p/q trajectory ratios and ESS only
   after the combat is nontrivial; *more draws can worsen IS ESS*.
5. Only then run a long H3 joint player–Nature training on this same
   support, comparing reference-law PPO, unweighted H3, and IS H3
   at equal emulator transitions. Include frozen initial-v8
   benchmark and, separately later, MCTS-guided evaluations.

## Fish commands (reuse existing H2 checkout / worktree)

```fish
cd ~/projects/sts2-ai-h2
git status --short
git fetch origin
git switch --detach origin/research/h3-boss-challenge-20261022
git submodule update --init --recursive emulator

# The AI parent submodule may point at the older emulator. Use the
# isolated draft PR #183 code, NOT the active comparator branch.
git -C emulator fetch origin feature/nature-h3-boss-decks-relics-20261022
git -C emulator switch --detach 083fe8de8dc35002f001c01b1de5d8164465c010

source .venv/bin/activate.fish
python -m pip install -e '.[neural]'
dotnet build emulator/Sts2Emulator.sln -c Release

set model ~/projects/sts2-ai/results/ppo-v8-batched-40/models/stage-0120.json
set nature (pwd)/results/nature/h2-a10-ppo-v8-seed29/checkpoint.json
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev 083fe8de8dc35002f001c01b1de5d8164465c010

# Diagnose difficulty with a selection set first.
python -u tools/evaluate_nature_h3_boss_challenge.py \
  --player-model $model --nature-checkpoint $nature \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 \
  --split selection --eval-episodes 120 --max-boundaries 1200 \
  --output-root results/nature/h3-boss-selection-v8-h2q

# Reserved disjoint confirmation set: run only once the fixture is viable.
python -u tools/evaluate_nature_h3_boss_challenge.py \
  --player-model $model --nature-checkpoint $nature \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 \
  --split confirmation --eval-episodes 1000 --max-boundaries 1200 \
  --output-root results/nature/h3-boss-confirmation-v8-h2q
```

Each run creates immutable `report.json` and `paired-worlds.jsonl`.
The report separates source-paired reference and Nature performance
by boss, deck, and all 4 configurations and describes total nontrivial
draw counts. It does not itself train the player.

Once the paired boss fixture passes the terminal/replay smoke, train
using the same optional H3 `--scenario-pool` and `--deck-pool`
flags with `--player-target reference-is`. Use a **different**
output root from older 12-cell experiments. Do not treat 20 rounds
or a short smoke as evidence of improved learning efficiency.

Native RNG parity/fidelity inspection remains a separate, explicitly
nonblocking research track.
