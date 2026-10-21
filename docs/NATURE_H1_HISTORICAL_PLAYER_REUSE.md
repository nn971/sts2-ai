# Reusing historical v8/v13 players for H1 draw-only Nature

**Status:** model-loader integration and reference-draw selection benchmark implemented on isolated AI branch. **Actual historical weight files are not in GitHub PR #82/#84, so no old-policy performance result has been measured yet.** In particular, CI's synthetic `v8` / `v13` shape fixtures **are not trained checkpoints** and cannot certify competence.

**Emulator dependency:** `nn971/sts2-emulator@3636d274cb5c0160d0b0e358b768dec83705201f` from isolated repair PR #176. External two-draw-mode and 1-vs-15 real-combat replay previously passed on AI PR #84. This comparison branch starts at PR #84's exact `522531c4f54e6f1fe3ddba1ee47d474584580d0e`; the original comparator, emulator, and AI PR #82/84 are untouched.

## Candidate models and compatibility

| Candidate label | Historical local output | Combat architecture |
|---|---|---|
| v8 | `results/ppo-v8-batched-40/models/stage-0120.json` | v8 enemy-instance tactical |
| v13 | `results/ppo-enemy-attention-v13/r1/attn/models/stage-0080.json` | v13 enemy-attention tactical |
| v15-relational, optional | `results/ppo-v15j-relational-cpu64/models/stage-0200.json` | portable relational |

These are **recorded historical local file paths**, not files verified to exist on this GitHub feature branch. Supply your actual local JSON files. The frozen adapter reads versioned **phase-split portable** `sts2-phase-split-policy-value-v1` models; a raw combat-only, PyTorch `.pt`, or an unsupported serialization needs a separately verified conversion. It SHA256-pins the whole file and rejects mutation between model selection and each worker's load.

**P0 public-observation compatibility:** the emulator's fair `PrototypeAiEnemy` exposes `instance_id`, `enemy_id`, public HP/powers, and intent, but **does not include `formation_position`**, which v8/v13 instance encoders require. The AI frozen-player bridge now derives only the unambiguous slot **0 for a single visible enemy**, through a dedicated audited `project_public_enemy_slot_for_legacy_instances` function (without adding an extra hashed categorical feature). It refuses ambiguous multiple-enemy observations unless all required formation positions are actually present. This uses only fair player-visible data, never provider-private stochastic metadata, and leaves the pooled/relational v15 adapter unchanged. Extend the emulator's public formation-position field before using legacy v8/v13 policies on multi-enemy encounters.

The historical model's `combat_value_objective` is descriptive metadata, not an inference compatibility gate. The H1 **player return** remains computed independently from genuine emulator terminal HP and victory. Therefore previous `continuation`-target players may be used unchanged; no value-head transplantation or forced objective overwrite.

The comparison uses the **same actual emulator** source seeds, fair public observations, legal action menus, physical-card constraints and ordinary Chance Law in both candidate arms. The frozen old policy receives no Nature-private support. The first H1 benchmark uses the explicitly restricted `silent-basic-v1` action scope and a real fixed Shrinker Beetle encounter. The scope must be reported alongside competence. Both ordinary-draw replicates use different independent reference seeds, allowing us to measure whether draw randomness materially changes the player's returns.

## Local evaluation and training (fish; `~/projects/sts2-ai`)

The H1 branch **pins its own emulator submodule** at `3636d274cb5c0160d0b0e358b768dec83705201f` (PR #85 commit `966a6893`), so there is no need to change your separate, active `sts2-emulator` working tree. Verify any uncommitted local AI changes before switching branches.

```fish
cd ~/projects/sts2-ai
git status --short
git fetch origin
git switch -c nature-h1-local --track origin/research/nature-h1-reuse-frozen-v8-v13-20261021
git submodule update --init --recursive emulator
git -C emulator rev-parse HEAD # must be 3636d274cb5c0160d0b0e358b768dec83705201f

python3 --version # Python 3.12+
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
dotnet build emulator/Sts2Emulator.sln -c Release

set revision 3636d274cb5c0160d0b0e358b768dec83705201f
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set v8 results/ppo-v8-batched-40/models/stage-0120.json
set v13 results/ppo-enemy-attention-v13/r1/attn/models/stage-0080.json
test -f $v8; and echo "v8 checkpoint present"; or echo "v8 checkpoint missing"
test -f $v13; and echo "v13 checkpoint present"; or echo "v13 checkpoint missing"
python -m pytest -q tests/test_nature_h1_frozen_model_reuse.py
```

If a different local `nature-h1-local` branch already exists, switch to it after fetching and verify its SHA manually. The checkpoint files above are locally generated experiment outputs and may be absent. Move or copy **actual** portable phase-split JSON checkpoints into these paths, or substitute their exact absolute paths. Examine a candidate's `format` key: it must be `sts2-phase-split-policy-value-v1`.

Run a short **technical** preflight (not research evidence) first:

```fish
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate "v8=$v8" --candidate "v13=$v13" \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $revision --workers 15 \
  --action-scope all-legal-v1 \
  --split selection --episodes 16 --reference-replicates 2 \
  --output-dir results/nature/h1-player-preflight
```

Then evaluate on enough independent sources to compare performance and draw sensitivity. Use the same action scope in player evaluation and Nature training:

```fish
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate "v8=$v8" --candidate "v13=$v13" \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $revision --workers 15 \
  --action-scope all-legal-v1 \
  --split selection --episodes 256 --reference-replicates 2 \
  --output-dir results/nature/h1-player-selection

python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate "v8=$v8" --candidate "v13=$v13" \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $revision --workers 15 \
  --action-scope all-legal-v1 \
  --split confirmation --episodes 512 --reference-replicates 2 \
  --output-dir results/nature/h1-player-confirmation
```

If `all-legal-v1` reveals a genuinely unsupported mechanical action, diagnose and document it; `silent-basic-v1` is a separately labeled restricted-policy alternative. Do not mix scopes in matched comparisons.

After selecting the genuine checkpoint, set `model` to its path (example: `set model $v8`). A single full H1 Nature run:

```fish
set model $v8
python -u tools/train_draw_only_nature_h1.py \
  --player-model $model \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $revision --workers 15 \
  --player-action-scope all-legal-v1 \
  --training-seed 29 --rounds 40 --episodes-per-round 256 \
  --eval-episodes 1000 \
  --output-root results/nature/h1-draw-v8-seed29 \
  --execute
```

The script requires `--execute`; without it, it only prints its immutable plan. It produces `checkpoint.json`, `holdout-report.json` and two full held-out JSONL files. To resume an interrupted run, repeat the **identical command** with `--resume` appended and unchanged model/args. For five separate Nature initializations, run the same command with `--training-seed` in `29 101 211 307 401` and a different `--output-root` each time. Each full seed uses 40 × 256 actual C# combats plus 1,000 held-out source combats **per arm**.

The H1 player is always frozen. This experiment optimizes the small `TwoSiteNature` REINFORCE policy, not the old neural player's weights. The 15 independent C# emulator workers primarily use CPU/RAM; Nature H1 does not benefit materially from the RTX GPU. Log output remains unbuffered via `python -u`. If the local system cannot sustain 15 concurrent emulator processes, rerun a fresh experiment with a smaller fixed `--workers` value and record it in the configuration.


**Results** (immutable outputs per directory): `report.json` with each model's format, SHA, original value objective, win rate, exit HP, return, bootstrap interval, average deferred physical draw count, fraction of source-paired reference runs that select different draws, and fraction where the **player reward** changes; plus full real episode `NAME-reference-00.jsonl` / `NAME-reference-01.jsonl`. The report includes a paired candidate return difference clustered by source seed, preserving matching opening IDs and reference seeds.

A clear player-quality result requires a meaningful source-replicated score, sufficiently frequent nontrivial card draws, and a measurable effect of reference draw outcomes on player reward. This selected fixed encounter may prove too easy; if scores saturate, choose a more discriminative supported combat before large-scale H1. Record that change as a distinct environment. Old all-Act win rates are context only.

## All-legal player-choice continuations

A real local v8 benchmark with `--action-scope all-legal-v1` reached
`state.boundary == "player_choice"`, which the first H1 replay loop did not
handle. The current branch now treats these as **genuine player-owned
decisions**, calling `semantic_observe`, restricting to the pending source
choice's action IDs, selecting through the frozen player's normal fair
`choose` interface, and committing via `semantic_step`. Exact replay
records `player_choice` as a separate phase, the source pending
event/support together with the fair action menu, and the resulting state.

Supported kinds are currently `player_card_selection` and
`draw_hook_action_choice`. Unsupported player-choice kinds, missing menus,
or mismatched candidate IDs remain explicit errors rather than silent
fallbacks; Nature still controls **only** physical draws. Targeted
`tests/test_nature_draw_only_player_choice.py` tests replay/menu integrity.
`tools/check_nature_draw_only_player_choice.py` provides a real
C# Survivor choice/replay regression case, included in H1 compatibility CI.

For a previously failed benchmark, update the local `nature-h1-local`
branch by fast-forwarding to the latest H1 PR branch and rerun with a
**fresh** `--output-dir` (for example `h1-player-selection-choice-v2`);
the benchmark deliberately does not overwrite partial results.

## H1 with the selected actual player

The existing `tools/train_draw_only_nature_h1.py` now permits historical phase-split combat **policy** heads irrespective of their original value-training objective. It still pins the model, owns a frozen player during Nature REINFORCE optimization, validates supported emulator chance boundaries and writes disjoint holdout evaluations.

After the frozen-player reference benchmark, supply the selected real checkpoint via `--player-model` and run the read-only plan first. Explicit `--execute` enables training. The current default budget is 40 x 256 real combats per Nature seed with 15 workers and 1,000 genuinely held-out combats per draw arm, replicated over at least five independent Nature seeds; this is **not** a claim that the study was executed or that an effect exists.

**No fidelity gate:** Source-law calibration N5, inspector comparison and game-native RNG parity remain independent. Declared reference probabilities retain the emulator's `UNVERIFIED` provenance without preventing legitimate emulator training.
