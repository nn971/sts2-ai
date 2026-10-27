# H7 stratified terminal replay: faster training without false verification claims

**Research branch:** `research/h7-sampled-terminal-replay-20261027`
**Base:** H7 shared-actor PR #116.

## Motivation — actual 15-worker profile

The 128-combat H7 A10 profile from sts2-emulator PR #194 shows that
`semantic_load` (terminal snapshot restoration by re-executing the entire
combat transcript) consumed **739.4 cumulative worker-seconds** across
128 episodes, compared with **2,195.2 worker-seconds** for all episode wall
durations, and 159 seconds of overlapping cohort elapsed time.
Four thousand draw resolutions consumed another 999.1 worker-seconds.
State-store forks alone were only 9.2 worker-seconds.

We preserve exact replay in the emulator. The performance experiment
changes the **training consumer's policy** for when to invoke it.

## Design and guarantees

- `play_draw_only_episode(verify_terminal_replay=True)` remains the default.
  When opted out, no private `semantic_save`/`semantic_load` is invoked
  at the terminal boundary. The returned immutable episode has
  `terminal_replay_verified=False`. A selected audit that fails raises;
  it is **never** converted to an unchecked success.
- `draw_only_batch(..., terminal_replay_indices=...)` selects indices
  before the workers are launched. Selection is independent of completion
  order, player decisions, chance outcomes or whether an episode won.
  Existing callers omitting this argument still replay **every episode**.
- H7 training's `--train-terminal-replay-every 8` audits the first
  episode and every eighth episode **within each curated combat cell**.
  This guarantees at least one audited episode per cell per training cohort.
  With 128 sources balanced over 16 cells, exactly 16 episodes are
  replayed (112 unchecked). With 256 sources: 32 replayed.
- H7 selection/heldout evaluation always replays every episode.
  The existing strict H7 readiness benchmark also replays every episode.
- H7 PPO's default is still strict. Sampled replay requires the explicit
  `allow_sampled_terminal_replay=True` opt-in used by H7 training **after**
  the collector has checked all source IDs, expected cell contexts,
  exact verification flags and the audit selection.
- The source-conditioned reference Chance Law, seeded action sampling,
  public policy observation, legal action menu, terminal outcome, score,
  and on-policy behavior-logprob checks are unchanged.
- H7 training progress contains `train.terminal_replay_verified`,
  `train.terminal_replay_not_checked`, the same counts for each cell
  and evaluation, plus the `terminal_replay_audit` manifest with
  schedule and selected seed digest.
- Existing `--train-terminal-replay-every 1` runs retain their
  original configuration fingerprint and resume behavior. A sampled-run
  fingerprint contains the new audit contract and cannot silently
  resume a full-audit checkpoint. Use a **fresh output-root**.

## Minimal CI gates

Unit tests cover deterministic source-cell coverage, invalid schedules,
wrong source/cell/audit statuses, a skipped episode retaining the exact
decision transcript but marking replay false, a corrupted selected
snapshot failing, and PPO strict-default/source-provenance rules.
A separate real-emulator H7 PPO smoke uses 32 train combat episodes
with 16 audited and 16 unchecked; the 16 evaluation combats are fully
audited. Both a checkpoint and policy weights must be produced.

## Suggested full-size controlled benchmark (fish)

The pinned H7 source benchmark uses the full 14,659,972-parameter actor.
Use your existing build and paths (from the PR #194 runbook).
Run one-round training twice from the same source seed and model
initialization; the only experimental variable is the replay rate.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h7-sampled-terminal-replay-20261027
source .venv/bin/activate.fish

set rev e56779ccd86725582af580925ca8242826ba2229
set h5 /tmp/sts2-h5-emulator.zIOUjm
set dll $h5/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings $h5/bindings/python

for rate in 1 8
    python -u tools/train_h7_reference_combat.py \
      --emulator-dll $dll --emulator-bindings $bindings \
      --emulator-revision $rev \
      --output-root /tmp/h7-terminal-audit-$rate \
      --rounds 1 --episodes-per-round 128 \
      --eval-every 1 --eval-episodes 16 \
      --workers 15 --device cuda \
      --actor-backend shared --actor-device cuda \
      --width 384 --layers 6 --heads 8 --feedforward 1536 \
      --train-terminal-replay-every $rate --execute
end
```

Inspect `progress.json` in both output roots. For a fair speed
comparison, the source_digest and behavior_sha256 must agree and
the reward/score/decision histories should match; the checkpoint
fingerprints **must differ** because the replay policy differs. Compare
`rollout_seconds` alone for collection throughput. PPO learning updates
and the selection evaluation are separate, and both runs will perform
identical optimizer computations. The GPU rollout speedup still needs a
matched real-hardware benchmark before it can be claimed.

## Next: draw resolution

The emulator's stable-ID `ResolveDrawCard` repeated invariant
validation and a canonical SHA-256 already performed by its caller;
PR #194 removes the duplicate. Its follow-up must quantify a matched
real H7 benchmark and measure allocations in the remaining transition
before touching structural sharing, pending-event logic, or caching.
Fidelity Inspector and native RNG cursor parity are *not* training gates.
