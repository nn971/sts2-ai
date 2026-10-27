# H8 / H7 shared Transformer integration pilot

Date: 2026-10-28. Branch `research/h8-h7-shared-transformer-pilot-20261028`
(draft PR #120), stacked on the H7 source sampled-replay branch.
Depends on emulator H8 public bridge [PR #211](https://github.com/nn971/sts2-emulator/pull/211)
(stacked on H8 consolidated #210).

## Implemented in this slice

`tools/benchmark_h8_h7_policy_pilot.py` uses the **unchanged 14,659,972
weight H7 SharedObjectTransformer** (width=384, layers=6, heads=8,
FFN=1536) through H7's existing single shared inference process
(`SharedH7Service`, `RemoteH7Policy`), existing frozen-source-fair
`FrozenP0Player` action sampling, and the original H7 balanced 16-cell
Ascension 10 source bank. The agent receives *exactly* the source
`PrototypeAiEnvironment.Observe` public policy view and source-visible
`ActionWire` menu from H8's new version-checked `h8_observe`.
The session still owns all mechanics and card/intent support.

- It samples **actual player actions** from the frozen H7 model, including
  legal card plays, rather than always ending its turn.
- The default `--reference-law-mode event-keyed-v111` requests the
  exact declared v111 enemy-intent/draw Chance Law through H8's explicit
  read-only `h8_reference_law` endpoint. The unchanged H7
  `DeclaredLaw.from_pending` and `SeededLawSampler` implement
  event-keyed reference sampling, preserving the old outcome IDs,
  raw or source-verified weights and per-source RNG seed.
- Optional `--reference-law-mode raw-weight-pilot` uses the compact
  weighted-intent and eligible physical-card support with a sequential
  seeded independent Nature RNG. This is a *faster preliminary mode*,
  not an H7 Chance Law equivalence claim. The exact mode currently
  incurs extra v111 event projection/hashes and is an explicit M3
  typed-Chance-Law optimization target.
- `h8_begin_seeded` now takes the existing H7 research deck preset
  exactly as the semantic source engine did. The H7 16 source cells are
  grouped and resolved before contact with the engine.
- A/B or A/B/B/A cohorts are **checked H8** vs **native H8**, with
  identical player checkpoint, actor seeds and Nature sample stream.
  Acceptance compares each source's chosen physical IDs, player actions,
  decision journal and **complete mechanics SHA**; 1-in-8 replay audits
  are selected by *global source index*.
- Existing nested `player_card_selection` and
  `draw_hook_action_choice` boundaries are resumed by H8's checked
  v111 continuation, with the exact H7 fair-action menu restriction.
  Unknown other subchoice types and unsupported stochastic boundaries
  **stop** the pilot with explicit per-type counts; they never count as
  terminal combats or PPO trajectories.
- A dedicated paired-repository CI builds emulator source, tests
  public observation/weighted-sampling invariants, then attempts
  H7 public inference for 16 A10 cells on shared CPU inference.
  The worker IPC uses explicit spawn-time queue inheritance; the
  shared actor has exactly one outstanding request per worker.
  Production benchmarking uses shared CUDA.

## Run on target workstation (fish)

First build the emulator branch including the H7 public bridge:

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-emulator switch research/h8-h7-public-policy-bridge-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h8-h7-shared-transformer-pilot-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emulator rev-parse HEAD)
python -m pytest -q tests/test_h8_h7_policy_pilot.py

python -u tools/benchmark_h8_h7_policy_pilot.py \
  --emulator-dll $dll \
  --emulator-bindings $emulator/bindings/python \
  --emulator-revision $rev \
  --workers 15 --episodes 128 --max-boundaries 1200 \
  --replay-every 8 --actor-device cuda \
  --actor-max-batch 16 --actor-delay-ms 3 \
  --reference-law-mode event-keyed-v111 \
  --order abba \
  --output results/h8-h7-transformer-pilot-a10-128-abba.json
```

Use `--player-checkpoint /absolute/path/to/h7-actor.pt` to select
the **same actual H7 model** rather than creating deterministic fresh
random weights; in either case the whole A/B/B/A run has an unchanged
frozen model and source cohort.

The output's `statuses.terminal` is the number of complete fights,
whereas `player_choice`, `unsupported`, and `max_boundaries` describe
nonterminal work; never substitute all 128 source count for completion.
`terminal_combats_per_second` measures real terminal completions only.

**Note:** The initial passing 32-boundary CPU CI measurement below
used the earlier compact sequential raw-weight sampler. The new
default event-keyed declared-law mode is the stronger reference
compatibility gate and must be validated independently; do not equate
the throughput of these two modes.

## Initial real-model CI acceptance

The first paired-source CPU run using all **16 curated A10 source cells**,
two H8 .NET emulator processes and the **14,659,972-parameter H7 model**
passed on [GitHub Actions run 40575127367](https://github.com/nn971/sts2-ai/actions/runs/40575127367).

- **All 16** checked/native H8 source trajectories matched the complete
  decision/action transcripts, immutable journals and full mechanics SHA.
- Checked: **19.75 s** for 16 seeded sessions; native: **15.23 s**
  (1.30x faster at that short capped workload on shared **CPU** inference).
  Exact actor and source RNG are held fixed between the two H8 arms.
- At the deliberately short **32 boundary cap**, **one** combat reached
  terminal; 15 were capped; no unsupported player/decision boundary was
  reported. Thus **this is not 16 completed combats** and is in no way
  a comparison against the user's H7 shared-CUDA 128-combat baseline.
- A larger **200-boundary** paired CPU integration workflow is the next
  acceptance checkpoint. Target workstation benchmark remains
  15 workers / 128 A10 source cells / shared CUDA at 1200 max boundaries.
- Prior failed CI attempts exposed an incorrect
  `multiprocessing.Queue` task pickle pattern. The live model pilot
  now uses explicit spawn-time shared queues and one persistent emulator
  worker per shard; this is validated by the passing real-model run.

## Remaining blockers for valid H7 versus H8 training comparison

1. The current default sampling is now **the exact H7 event-keyed
   `DeclaredLaw` + `SeededLawSampler`** through a temporary v111
   read-only law bridge. Replace that expensive compatibility projection
   with an interchangeable **typed H8 Chance Law provider** returning
   byte-for-byte equivalent event identities, conditioning, eligible
   outcomes and weights. The cheaper sequential `raw-weight-pilot`
   mode remains research-only and must never silently replace the
   training reference law. A full legacy H7-vs-H8 law identity/trajectory
   comparison remains to be completed.
2. The existing `player_card_selection` and
   `draw_hook_action_choice` cases now resume through a checked
   v111 continuation with exact H7 fair-action filtering. Continue
   migrating any remaining nested player choices and stochastic sites
   (random targets, costs, generated cards) through typed H8 cursors
   or an exact checked fallback.
3. Promote this pilot to `draw_only_workers` once full source-valid
   H7 16-cell terminal combats and audited on-policy source logprobs
   can be gathered. Build a matched H7-legacy vs H8-native 15-worker,
   128-combat A/B/B/A under a *single frozen checkpoint SHA* and the
   exact same reference law and source cells.
4. Remove output-only v111 continuation projection, design one native
   H8 effect cursor/owner, and transport typed batched observations
   without repeated JSON for additional throughput after profiling.
5. **No emulator submodule pin update or long PPO training** until
   complete mechanics, replay and on-policy capture pass.

## Reference baseline

User's legacy H7 local full 128 source-complete A10 combats with
15 processes and shared CUDA inference: **53.131 s**, **2.409 complete
combats/s**, **64.331 player decisions/s**, **144.153 boundaries/s**,
source digest `f7e599e1154c99c59c52d85174daa7b329f456396d2f779bbd7d14304905c3b5`
and emulator rev `ec948b5b5abbb94a4ef63d362f8eefe71656987b`.
The H8-scripted 128-combat end-turn-only benchmark on 15 workers
measured a 2.01x speedup, but omitted model inference and all real
player card decisions. Do not extrapolate that result to PPO.
