# H6-A: Frozen Combat, On-Policy Macro PPO

**Status:** implementation and mechanical CI in progress; substantive full-run
training **not yet performed**. Stacked on H5 PR #98. H1–H5 research runs
remain untouched.

## Research question

Does updating only the macro policy (routes, rewards, rest, shops, events
and deck choices) improve native-structure Silent Ascension 10 **Act 1
boss-clear probability**, given an identical competent tactical policy?

The portable `PhaseSplitNeuralModel` already has disjoint strategy and
combat weights. H6-A explicitly freezes the serialized combat model and
its learned value head. Its strategy optimizer sees *only public macro
decisions* from genuine full emulator runs. There is no cross-phase
importance sampling, no learned Nature and no semantic-state UCT.

## Experiment and architectural contract

- Environment: `native-overgrowth`, explicit `ascension=10`,
  `native-act1-boss-v1` completion goal. The certified Act-1 transition
  counts as a success; capped runs remain censored and are never invented
  defeats or awarded Monte Carlo returns.
- Initial backbone: frozen H4 **reference-only** player, round 40.
  Candidate H5 can replace it only after independent evaluation.
- Combat chooses **greedy** actions from the frozen exported player,
  separate from strategy exploration temperature. All uncertainty in
  encounter/chance events continues through the emulator.
- Strategy is sampled on-policy using temperature 0.85 during training.
  At evaluation both phases are greedy. State and legal-action encoding
  is exactly the public semantic-action format already in the model.
- Each completed run contributes its certified Act-1 terminal return:
  1 for boss clear; for completed defeats a small floor/HP auxiliary signal
  based on `_bounded_return` with coefficient initially 0.35, linearly
  annealed to zero after 80 observed victories.
- PPO (3 epochs, minibatch 128, max 4096 strategy decisions, LR 3e-4).
  No combat samples are fed to the PPO batches or combat optimizer.
  Save strategy optimizer state, Torch RNG, source digests, per-round
  training and selection telemetry, and frozen combat SHA-256.
- Training source IDs: distinct namespace per round; fixed **selection**
  seeds for checkpoint monitoring; independent **confirmation** seeds
  reserved for final model comparison. Policy evaluation is same
  run-source pairing, not necessarily identical stochastic continuations.
- Censoring and emulator errors are separate: supported max-decision caps
  are marked truncated, not losses; unsupported mechanics cause explicit
  failure instead of silently generating a surrogate world.
- **No fidelity/native RNG parity gate**. No emulator code is modified.

The existing macro encoder is general semantic hashing; it does *not*
properly distinguish all map futures or card-reward synergies. Its
representation is a known limitation: H6-B will introduce candidate-aware
map path/deck features after we establish this exact-freeze baseline.

## Validation criteria

1. Native A10 reset and completion goal used, verified end-to-end with
   an actual .NET semantic run (not the older A0 legacy prototype).
2. Across training/checkpoint/resume, `combat` JSON remains **identical**
   to the warm-start checkpoint; `combat_optimizer_steps == 0`.
3. Strategy can learn on completed run trajectories; all truncations
   excluded from gradient returns. Count nonterminal censoring explicitly.
4. Evaluate **the original strategy head versus the learned strategy**
   with the *same combat backbone*, 15 workers, source-matched confirmation,
   independent of selection. Report completion coverage and paired
   victory difference on *both-completed* runs with bootstrap interval.
5. Include per-run macro/combat decision counts and combat outcomes.
   Do not call a 1-round mechanical test evidence for strategic learning.

## Run locally (Fish; existing directories only)

Only use `~/projects/sts2-ai`, do not create extra worktrees or new
folders under `~/projects`.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch --track origin/research/h6-macro-frozen-combat-20261026
git submodule update --init emulator
git -C emulator fetch origin fix/h6-event-gold-bounded-20261026
git -C emulator switch --detach a580b1d9f1c961472c4189dc15421bd803afeec3
source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
dotnet build emulator/Sts2Emulator.sln -c Release

set rev (git -C emulator rev-parse HEAD)
set h4 results/nature/h4-reference-only-a10/player_models/player-r0040.json
```

Dry-run configuration check:

```fish
python -u tools/train_frozen_macro_h6.py \\
  --player-model $h4 --emulator-revision $rev \\
  --output-root results/macro/h6-frozen-h4-a10
```

Mechanical smoke (**not** learning evidence):

```fish
python -u tools/train_frozen_macro_h6.py \\
  --player-model $h4 --emulator-revision $rev \\
  --workers 15 --rounds 1 --episodes-per-round 4 \\
  --eval-every 1 --eval-episodes 4 --max-decisions 1800 \\
  --output-root results/macro/h6-smoke --execute
```

Full experiment, 40 rounds × 32 complete-run attempts,
15 emulator workers (actual computation may be substantial):

```fish
python -u tools/train_frozen_macro_h6.py \\
  --player-model $h4 --emulator-revision $rev \\
  --workers 15 --rounds 40 --episodes-per-round 32 \\
  --eval-every 5 --eval-episodes 32 --max-decisions 4096 \\
  --output-root results/macro/h6-frozen-h4-a10 --execute \\
  2>&1 | tee h6-frozen-h4-a10.log
```

Interrupted runs may be resumed by repeating the *identical options*
and adding `--resume`; previous macro/combat checkpoints are retained.

After strategy checkpoint selection has finished, freeze the candidate and
evaluate against the **unchanged H4 strategy baseline**, both with
identical combat weights:

```fish
set macro results/macro/h6-frozen-h4-a10/player_models/player-r0040.json

python -u tools/evaluate_frozen_macro_h6.py \\
  --player initial=$h4 --player trained=$macro \\
  --emulator-revision $rev --workers 15 \\
  --split confirmation --eval-episodes 120 \\
  --max-decisions 4096 \\
  --output results/macro/h6-confirmation.json --execute \\
  2>&1 | tee h6-confirmation.log
```

Evaluation supports `--resume` per completed player arm. Check whether
all-source completion coverage is sufficiently high before interpreting
victory deltas. Do not treat truncated runs as binary losses.

## Native full-run compatibility

H6-A uses the isolated emulator branch `fix/h6-event-gold-bounded-20261026`
([emulator PR #186](https://github.com/nn971/sts2-emulator/pull/186))
based on the H4 pin. It repairs a real event-generated-gold bounded integer
RNG operation crash reached during native A10 run rollout at decision 51.
This is a gameplay transition repair, **not** a fidelity/inspector test gate.
The H6 targeted CI checks native opening, actual full-run stepping,
strategy-only checkpoint and source-paired evaluation before a long run.

## Next stage

H6-B upgrades public-only strategy features for map connections,
prospective elite/rest opportunities, card/relic/deck interactions and
reward candidate valuation. H6-C explores combat-to-macro resource
valuations and alternating, **not immediately fully joint**, optimization.
Potential combat reward mismatch (e.g. potion use vs later HP) is a
limitation of the frozen-backbone experiment, not a reason to delay H6-A.

Future broader combat builds depend on independent emulator issue #185;
native inspector fidelity remains explicitly non-blocking.
