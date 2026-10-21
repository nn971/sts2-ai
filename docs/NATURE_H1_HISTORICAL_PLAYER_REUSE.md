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

The historical model's `combat_value_objective` is descriptive metadata, not an inference compatibility gate. The H1 **player return** remains computed independently from genuine emulator terminal HP and victory. Therefore previous `continuation`-target players may be used unchanged; no value-head transplantation or forced objective overwrite.

The comparison uses the **same actual emulator** source seeds, fair public observations, legal action menus, physical-card constraints and ordinary Chance Law in both candidate arms. The frozen old policy receives no Nature-private support. The first H1 benchmark uses the explicitly restricted `silent-basic-v1` action scope and a real fixed Shrinker Beetle encounter. The scope must be reported alongside competence. Both ordinary-draw replicates use different independent reference seeds, allowing us to measure whether draw randomness materially changes the player's returns.

## Local evaluation commands (fish)

Run in the **isolated AI branch checkout**, with the frozen model paths adjusted to your own files. The source splits do **not** overlap H1 Nature training or held-out H1 evaluation.

```fish
set emulator /path/to/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings $emulator/bindings/python
set revision 3636d274cb5c0160d0b0e358b768dec83705201f
set v8 results/ppo-v8-batched-40/models/stage-0120.json
set v13 results/ppo-enemy-attention-v13/r1/attn/models/stage-0080.json

# Build exactly the revision that passed the focused real-combat gate.
git -C $emulator checkout $revision
dotnet build $emulator/Sts2Emulator.sln -c Release

# Same real fixtures, reference draws, two independent reference RNG replicas,
# source-level bootstrap, player policy frozen throughout.
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate v8=$v8 --candidate v13=$v13 \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $revision --workers 15 \
  --split selection --episodes 256 --reference-replicates 2 \
  --output-dir results/nature/h1-player-selection

# Confirm the choice independently. Changing the split changes source seeds.
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate v8=$v8 --candidate v13=$v13 \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $revision --workers 15 \
  --split confirmation --episodes 512 --reference-replicates 2 \
  --output-dir results/nature/h1-player-confirmation
```

**Results** (immutable outputs per directory): `report.json` with each model's format, SHA, original value objective, win rate, exit HP, return, bootstrap interval, average deferred physical draw count, fraction of source-paired reference runs that select different draws, and fraction where the **player reward** changes; plus full real episode `NAME-reference-00.jsonl` / `NAME-reference-01.jsonl`. The report includes a paired candidate return difference clustered by source seed, preserving matching opening IDs and reference seeds.

A clear player-quality result requires a meaningful source-replicated score, sufficiently frequent nontrivial card draws, and a measurable effect of reference draw outcomes on player reward. This selected fixed encounter may prove too easy; if scores saturate, choose a more discriminative supported combat before large-scale H1. Record that change as a distinct environment. Old all-Act win rates are context only.

## H1 with the selected actual player

The existing `tools/train_draw_only_nature_h1.py` now permits historical phase-split combat **policy** heads irrespective of their original value-training objective. It still pins the model, owns a frozen player during Nature REINFORCE optimization, validates supported emulator chance boundaries and writes disjoint holdout evaluations.

After the frozen-player reference benchmark, supply the selected real checkpoint via `--player-model` and run the read-only plan first. Explicit `--execute` enables training. The current default budget is 40 x 256 real combats per Nature seed with 15 workers and 1,000 genuinely held-out combats per draw arm, replicated over at least five independent Nature seeds; this is **not** a claim that the study was executed or that an effect exists.

**No fidelity gate:** Source-law calibration N5, inspector comparison and game-native RNG parity remain independent. Declared reference probabilities retain the emulator's `UNVERIFIED` provenance without preventing legitimate emulator training.
