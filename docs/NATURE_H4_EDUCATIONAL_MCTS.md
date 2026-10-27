# H4: Educational Nature and greedy/UCT boss evaluation

**Status:** implementation candidate. Preserve all H3 results. The H4
Python branch is `research/h4-educational-nature-mcts-20261024` (PR #97);
its isolated emulator companion is
`feature/h4-terminal-boss-hp-20261024` (PR #184).

## Question and controlled comparison

Can Nature learn to **teach** rather than harm the Silent player, improving
performance under the original draw law? Compare under equal emulator-
transition budgets:

1. **Reference-only PPO** (H4 reward, 256 reference draws per round).
2. **Educational joint PPO** (128 reference and 128 educational draws, with
   auxiliary weight 0.25 for the player and a reference-gradient-alignment
   teacher update for Nature).
3. Optionally replay archived **H3 adversarial IS** and frozen v8 checkpoints
   as external baselines. H3's old reward must be recomputed with H4's v2
   score before comparing scores.

Use two A10 Act 1 prototype bosses × two 20-card archetypes, 15 workers,
40 rounds × 256 combats, repeated holdout checks. Training player actions
remain sampled, while evaluation player actions are greedy or searched.

## Reward contract

`boss_progress_v2` uses `1 + exit_player_hp/max_player_hp` on victories,
and `max(0, 1-terminal_enemy_hp/opening_enemy_hp)` on defeats.
Every victory outranks every defeat. `terminalEnemyHp` must be sampled
from the authoritative game state *before combat teardown*. On H4's current
single-boss fixtures the first visible enemy HP is the denominator. A
multi-phase boss that heals beyond its opening HP can receive a clamped zero
progress signal upon defeat; log this separately before extending H4 to
other bosses.

The old H3 score and entrypoint default remain untouched. Reward version,
opening enemy HP and final enemy HP are included in H4 episode records.

## Teacher and student optimization

Before updating either policy, derive an estimated reference policy gradient
from the reference-arm sampled-combat trajectories. Each educational
trajectory gets a signed cosine alignment with that gradient. This is a
**first-order proxy** for actual held-out reference-law learning progress;
it must be validated against eventual improvement.

The player uses ordinary reference PPO plus a weighted auxiliary PPO loss
on educational experience. No full-combat importance ratios: these were
negligible in H3. A degenerate reference gradient produces a reference-only
student step without advancing the teacher. Nature's PPO remains on-policy
with its captured draw decisions.

Track `reference_score`, `reference_victory`,
`teacher_mean_alignment`, `teacher_positive_fraction`, player/Nature
KL, wall clock, emulator transitions and deck/boss strata.

## Matched evaluation

`tools/evaluate_nature_h4_greedy_mcts.py` compares the same exported
player model using greedy actions versus reference-Chance-Law UCT for each
real player decision. The tool accepts multiple `--player name=path`
arguments and both reference/Nature actual draw distributions.

Its UCT search branches on legal *public player* actions, and stochastic
simulation samples unresolved physical draws and enemy intents from the
declared reference Chance Law. Child decisions are keyed by public
observations. **The root simulator still uses the current exact semantic
continuation**: this is a root-conditioned simulator search baseline.
It may retain hidden information via the starting continuation and therefore
must be labeled separately from a belief-state-fair MCTS baseline and the
legacy exact-state oracle MCTS. For a genuinely fair comparison, add
sampling of hidden states from a public-history posterior before each
search rollout; that extension is future work.

Use identical source-world sets for greedy and UCT, measure matched
win-rate/score differences, simulations, transition counts and wall-clock
cost. Search planners use reference draws even when the actual environment
uses educational or adversarial Nature.

## Local runbook (Fish)

Use an existing checkout, with a clean tree, and keep emulator development
on its separate branch/submodule. Do not create new project worktrees.
Resolve and pin the SHA from PR #184 before launching.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h4-educational-nature-mcts-20261024
git submodule update --init emulator
git -C emulator fetch origin feature/h4-terminal-boss-hp-20261024
git -C emulator switch --detach origin/feature/h4-terminal-boss-hp-20261024
source .venv/bin/activate.fish
python -m pip install -e '.[neural]'
dotnet build emulator/Sts2Emulator.sln -c Release
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev (git -C emulator rev-parse HEAD)
set model (pwd)/results/ppo-v8-batched-40/models/stage-0120.json
```

Reference-only H4 control:

```fish
python -u tools/train_educational_nature_h4.py \
  --player-model $model --emulator-dll $dll \
  --emulator-bindings $bindings --emulator-revision $rev \
  --workers 15 --rounds 40 --episodes-per-round 256 \
  --training-mode reference-only --max-boundaries 1200 \
  --eval-every 5 --eval-episodes 120 \
  --output-root results/nature/h4-reference-only-a10 \
  --execute 2>&1 | tee h4-reference-only.log
```

Educational H4:

```fish
python -u tools/train_educational_nature_h4.py \
  --player-model $model --emulator-dll $dll \
  --emulator-bindings $bindings --emulator-revision $rev \
  --workers 15 --rounds 40 --episodes-per-round 256 \
  --reference-fraction 0.5 --educational-weight 0.25 \
  --max-boundaries 1200 --eval-every 5 --eval-episodes 120 \
  --output-root results/nature/h4-educational-a10 \
  --execute 2>&1 | tee h4-educational.log
```

After training, compare greedy versus UCT at budget 16 on 120 identical
source worlds. Replace the H3 path with an actual exported checkpoint
if desired:

```fish
python -u tools/evaluate_nature_h4_greedy_mcts.py \
  --player v8=$model \
  --player reference=results/nature/h4-reference-only-a10/player_models/player-r0040.json \
  --player educational=results/nature/h4-educational-a10/player_models/player-r0040.json \
  --nature-checkpoint results/nature/h4-educational-a10/joint-checkpoint.pt \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 \
  --eval-episodes 120 --mcts-simulations 16 \
  --output results/nature/h4-greedy-uct-selection.json \
  --execute 2>&1 | tee h4-greedy-uct.log
```

The 40-round experiments and MCTS evaluation are substantial local
computations. The 1-round CI is a mechanical smoke test, **not evidence of
learning effectiveness**. Use independent confirmation worlds after
checkpoint/model-selection choices are frozen.

**Constraints:** Keep Silent A10 single-player scope; preserve H1–H3
artifacts; no new folders under `~/projects`; 15 workers in real runs;
native fidelity and RNG parity are independent research tasks, never a
training gate.

## H4 UCT runtime correction

The original UCT evaluator mistakenly permitted 16 full-combat simulations
of up to 1200 boundaries **at every single player decision**. On a boss
fight this multiplies the already-expensive semantic JSONL calls enormously:
a silent period after the greedy arms was a compute explosion, not evidence
of failed reference-law training.

The corrected branch defaults to a **24-boundary rollout horizon** and a
**4000-simulated-boundary cap per real combat**. Search continues until the
cap is depleted, then the player chooses greedily for the remainder of
the fight. This policy is explicitly a **budgeted UCT/greedy hybrid**, not
unlimited/full-tree MCTS. Each completed comparison arm is now saved
atomically in the output JSON, with per-world scores, and can be resumed
using `--resume`. Worker-level UCT progress is printed while searching;
parallel-batch progress is printed after each completed UCT combat.

Recommended initial mechanical check (reuse the `$dll`, `$bindings`,
`$rev`, and `$model` variables above):

```fish
python -u tools/evaluate_nature_h4_greedy_mcts.py \
  --player v8=$model --emulator-dll $dll \
  --emulator-bindings $bindings --emulator-revision $rev \
  --eval-split confirmation --eval-episodes 8 \
  --mcts-simulations 8 --mcts-max-boundaries 16 \
  --mcts-combat-budget 1500 --workers 15 \
  --output results/nature/h4-uct-smoke.json --execute
```

For the substantial, independent confirmation comparison, use 120 worlds
and 16 simulations with the default horizon and combat budget.
Results report `mcts_search_boundaries`, `mcts_searched_decisions` and
`mcts_greedy_fallback_decisions` so the actual search effort and budget
exhaustion can be assessed. The `--mcts-max-boundaries` argument is
**per rollout**; `--max-boundaries` still caps *real combat* length.
Expensive horizons over 128 are rejected unless
`--allow-expensive-search` is explicitly specified.

Do not treat the previous greedy log lines as recoverable benchmark
checkpoints; the original evaluator saved its JSON only at the very end.
Future runs preserve each completed arm.
