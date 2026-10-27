# H7 combined throughput baseline — exact pinned experiment

This branch intentionally **combines** the two measured optimizations without
merging or modifying unrelated work:

- **Emulator:** `nn971/sts2-emulator` PR #194 commit
  `bfbcc394a2d6f09927d8935b52a3518e1613d934`. Its stable-ID draw and intent resolutions omit duplicated
  full-state validation/hashing; the emulator still owns explicit exact
  semantic snapshot/replay.
- **AI:** stacked on H7 replay-audit PR #117. In training,
  `--train-terminal-replay-every 8` checks one terminal per combat cell;
  every selection/evaluation terminal is still fully verified. Unchecked
  episodes remain `terminal_replay_verified=False`.
- **Submodule:** `emulator` gitlink pinned exactly to the emulator SHA above.
  Never silently float to the active emulator development branch. The
  smoke workflow asserts this gitlink, compiles it, performs 32 genuine
  reference-law A10 combats with the audit enabled, and evaluates 16
  fully replayed combats.

This integration branch does not alter H7's Python event-key-seeded
reference Chance Law, the frozen behavior model, or fidelity Inspector policy.

## Existing controlled measurements (different individual experiments)

- Sampled terminal replay: 128 combats, 15 workers, H7 14,659,972-param
  CUDA shared actor, H5-compatible emulator `e56779cd`: 166.50s with
  128/128 checked versus 123.30s with 16/128 checked, identical 34/128
  train wins and 13/16 heldout wins.
- Draw-resolution optimization: same model/seed/world with *full* replay,
  `ca458eda7` vs `bfbcc394a`:
  161.22s vs 151.72s; source and model digests agree;
  internal draw-transition allocation 352.81GB vs 258.99GB
  across 4,054 draws. Native profiler measured cumulative
  `sub.semantic_resolve_draw_card_transition` of 657.18s vs 523.19s.
- **The speed of both optimizations together is not yet measured.**
  The two isolated measured gains must not be multiplied and presented
  as an observed result.

## Run combined 128-combat GPU experiment, fish

The gitlink and docs make the exact research revision persistent. This
benchmark builds it from a temporary export so your active checkouts are
left untouched. Run from the combined H7 AI branch, with environment
already prepared:

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

set rev bfbcc394a2d6f09927d8935b52a3518e1613d934
set root (mktemp -d /tmp/sts2-h7-combined.XXXXXX)
set src $root/emulator
mkdir -p $src
git -C ~/projects/sts2-emulator fetch origin perf/h7-semantic-rollout-hotpath-20261027
git -C ~/projects/sts2-emulator archive $rev | tar -xf - -C $src
dotnet restore $src/Sts2Emulator.sln
and dotnet build $src/Sts2Emulator.sln -c Release --no-restore -m:1

set dll $src/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings $src/bindings/python
set -gx STS2_SEMANTIC_PROFILE_DIR $root/profiles
mkdir -p $STS2_SEMANTIC_PROFILE_DIR

python -u tools/train_h7_reference_combat.py \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev \
    --output-root $root/run \
    --rounds 1 --episodes-per-round 128 \
    --eval-every 1 --eval-episodes 16 --workers 15 \
    --device cuda --actor-backend shared --actor-device cuda \
    --width 384 --layers 6 --heads 8 --feedforward 1536 \
    --train-terminal-replay-every 8 --execute

python $src/tools/summarize_semantic_perf.py \
    $STS2_SEMANTIC_PROFILE_DIR --output $root/native.json
set -e STS2_SEMANTIC_PROFILE_DIR
echo $root
```

The `run/progress.json` report must contain training
`terminal_replay_verified=16`, `terminal_replay_not_checked=112`,
and selection `terminal_replay_verified=16`. Compare training
`rollout_seconds`, outcomes, source digest, actor-model SHA, decision
counts and backend before claiming a speedup. Avoid comparing full PPO
wall time to inference-only benchmark time.

For a high-quality comparison against the old replay-only run, use
identical fresh directories and seed/model configuration; inspect
`run/progress.json` and `native.json`. The accumulated native
`sub.*` categories are nested inside `op.*`, and overlapping
15-worker durations are **not** wall-clock durations.

## Next emulator research

The remaining `semantic_resolve_draw_card` costs roughly 64MB of
managed allocation per inner draw transition on the measured H7
fixture. Inspect the nested cost in
`SemanticPilotCombatEngineV111.ResolveDrawCard` →
`SemanticPilotCombatEngineV111.Resolve` →
`SemanticEndTurnEngineV111.Resolve` →
`SemanticCombatDrawEngineV111.Resolve`.

Collect per-stage allocations **before** eliminating another validation
or fork: some checks guard nested/stale-source invariants and changing
ownership of run snapshots may invalidate replay. See the emulator's
separate diagnostic research branch; the pinned commit here remains
unchanged until a new change passes genuine-combat replay equivalence.
