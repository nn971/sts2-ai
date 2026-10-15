# First Nature combat-training pilot — intents and deferred draws only

**Scope:** Silent combat. Nature controls only `enemy_intent` and unresolved
`draw_card` **physical card instances**. The emulator controls mechanics,
source eligibility, draw-pile constraints, intent publication, all other
randomness and player-visible observations.

**AI branch:** `feature/nature-intent-draw-combat-pilot`, stacked on
[N3 playable fixture diagnostics PR #77](https://github.com/nn971/sts2-ai/pull/77).
The desired RNG/Nature documents (updated 2026-10-15) remain authoritative.

## Experiment question and success criterion

Does training the player against a small, learned Nature proposal over only
enemy intents and card draws produce a stronger player under **ordinary
native RNG** at a fixed simulator budget?

The first experiment has two arms from *identical* initial player weights,
candidate events, supported encounter set and stage-aligned combat snapshots:

1. **Control:** player trained under native game RNG, including genuine source
   intent weights and the committed hidden draw order.
2. **Nature arm:** alternate frozen-player Nature-optimization episodes and
   player-optimization episodes. Nature minimizes the player's combat return
   with a strictly positive legal-proposal floor; the player maximizes its
   return against that controlled proposal.

Evaluate **both** player checkpoints under independent **native RNG** on a
common, source-run-disjoint combat holdout. Compare combat survival, win
rate, exit HP, native evaluation return and confidence intervals clustered
by source run. Evaluate after fixed total emulator combats / elapsed compute.
Run at least three training seeds before any promotion decision.

This first learned proposal is a **stress-testing adversary**, *not*
variance-minimizing importance sampling. Its `q` need not equal source
`p`, and losses against Nature alone do not demonstrate better normal play.
A later calibrated `p/q` experiment can test variance-minimizing Nature
separately after the N5 reference-law gate.

## Implemented AI components

- `nature/two_site_opponent.py`: small event/context/candidate-feature
  logits; full-support mixture of source-provided legal proposal and a
  learned softmax. Intent proposal uses current positive source weights
  as a **base q**, never assuming the game's conditional law has been
  audited. Draw proposal samples distinct **eligible card instances**
  uniformly at baseline. Source weights of zero are ineligible. Uses
  complete-terminal REINFORCE with a leave-one-episode-out baseline.
  Stores actual chosen `log q`, behavioral revision, and checkpoint
  provenance; stale/censored episodes cannot update.
- `nature/two_site_semantic_runner.py`: future real-emulator binding of
  `SemanticNatureBridge` with explicit `describe_event` (source-backed
  card-model labels/context), public-only `player_choose`, and an
  externally validated terminal-return decoder. A bare terminal Boolean
  and unclassified outcomes are insufficient training labels.
- `training/two_site_combat_toy.py`: **explicitly synthetic** three-turn
  finite combat: visible light/heavy enemy intents, draw-on-demand from
  physical card instances without replacement, and public card/intent/HP
  player choices. Its declared reference law is 0.7/0.3 for intents and
  uniform legal card draw. Runs equal-simulator-episode-budget training:
  native control gets two player batches per round; Nature arm uses one
  frozen-player Nature batch + one player batch. Evaluates on fixed
  independent synthetic-native seeds.
- `tools/try_two_site_nature_toy.py`: prints checkpoint-by-checkpoint
  native held-out performance and writes the report. Useful to detect
  wrong gradient direction, support, batch accounting, determinism,
  or a proposal that cannot find poor outcomes; **not** a StS2 result.

## Run immediate synthetic learning test (Fish)

```fish
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_nature_two_site_pilot.py
python -u tools/try_two_site_nature_toy.py \
    --rounds 40 --episodes-per-batch 32 \
    --evaluation-episodes 1024 --seed 29 --nature-mix 0.75 \
    --output results/nature/two-site-toy-seed29.json
```

Run several seeds and inspect the paired native-eval deltas. The testbed's
arbitrary toy card/HP rewards cannot establish actual StS2 performance.

## Live emulator integration gate (issue #143)

[Emulator PR #162](https://github.com/nn971/sts2-emulator/pull/162)
already implements a **restricted genuine end-turn action**:
real enemy-intent commitments, source-timed discard recycle, deferred
physical draw identities, per-card hooks, and the next legal player
decision. It is limited to a checkpoint with **empty draw pile and
nonempty discard pile** and remains internal to the C# emulator.
Complete semantic combat JSONL and persistent per-card unknown-order
state across arbitrary player actions are still pending.

To connect the above training loop to actual combat, the emulator must
supply a **new, opt-in, versioned two-site playable semantic capability**:

- **Start/step/resolve/terminal:** return a fair public player
  observation, legal player action IDs, or pending semantic `enemy_intent`
  / `draw_card` choice; resume to the next boundary/terminal through
  original engine hooks, persisting the semantic deck across steps.
- **Current eligible intent metadata:** source AI move-history/phase
  filtering, stable intent identity, raw weights/total, and the already
  committed public intent on the next player observation.
- **Eligible physical card metadata:** stable instance IDs **plus**
  card-model descriptor for generalizable Nature learning; the hidden
  position constraints remain authoritative inside the emulator.
  Known-top draws should not create a redundant Nature event.
- **RNG outside the two sites:** continue under a separately specified,
  mechanically source-correct native/default provider. Never silently
  substitute legacy Fisher–Yates swaps for semantic deferred draws.
- **Terminal outcome:** explicitly identify combat victory/defeat and
  player exit HP, with source run/snapshot partition and reward provenance.
  Preserve fork/save/load, deterministic replay, and fail-closed
  unsupported-event classification.
- **15-process rollout support:** independent handles, event context,
  q-version and experiment manifest per worker. Verify identical
  transcripts across 1 and 15 workers.

The current AI semantic bridge `sts2-v111-semantic-chance-v1` expects
its negotiated capability; it deliberately rejects today's legacy
`controlled_*` shuffle interface. The toy game is a learning test while
the emulator supplies the missing semantic gameplay boundary. Real
StS2 combat is therefore **not yet a reported experimental result**.

## Pilot rollout design after the bridge lands

Start with a small set of source-supported Silent combat snapshots from
Overgrowth/Hive, at appropriate deck progression, excluding unsupported
cost/peek/scry mechanics in the first run. A frozen v15 player and a tiny
Nature policy establish supported multi-turn terminal replays. Then:

- Run a 15-worker paired **native versus learned-Nature** training
  comparison with the *same* total simulator episodes, same snapshots,
  player model and training hyperparameters, and logged opponent
  optimization overhead. Keep adversarial rollouts and native holdout
  mutually independent.
- Preserve model versions, complete Nature `q` for every choice, event
  support/constraints, player public observations, terminal rewards,
  all RNG outside the controlled sites, and source-run groups.
- Measure adversarial difficulty against a frozen player and **native
  held-out performance after player updates** as separate metrics.
  Track policy KL/cross-entropy, q-entropy, proposal floor, defeat
  enrichment, wall time and confidence intervals. More difficult
  Nature rollouts are only an intermediate diagnostic.

### Explicit constraints

No emulator source code, main branch, emulator gitlink, frozen v15
checkpoints, or existing PPO trainer changed. This branch introduces a
standalone Nature learning pilot and toy experiment. Actual native
chance-law calibration and calibrated importance-weighted PPO are later
work packages. The first experiment uses no map, rewards, shops,
formations, or other discretionary Nature-controlled RNG.
