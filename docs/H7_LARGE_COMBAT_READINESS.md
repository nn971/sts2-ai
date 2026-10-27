# H7 large combat — throughput readiness (2026-10-27)

## Snapshot

The merged-in-history H7 encoder and Transformer already implement a **14,659,972 parameter**
384-wide / 6-layer / 8-head / 1536-feedforward public-only combat model, starting
from random weights. The small-model PPO smoke and checkpoint creation pass on the
PR #112 integration branch (`01ba5959b4a7`). This is *mechanical integration* evidence.

The existing H7 trainer still has exactly **16 curated Silent A10 training
scenario/deck combinations**, plus three reserved zero-shot combinations.
More independent RNG seeds or more PPO episodes cannot increase that
configuration diversity. Emulator issue [#185](https://github.com/nn971/sts2-emulator/issues/185)
tracks engine-owned card/relic/potion/HP/encounter combat fixture support.

## Immediate executable gate — actual actor throughput

`tools/benchmark_h7_rollouts.py` uses **the same H7 P0 worker collection path**
as the PPO trainer, including replay-verified terminal outcomes, source-public
legal actions and the reference Chance Law. It does no optimizer updates.

For a comparable benchmark, use the **H5-compatible semantic emulator** pinned
at `e56779ccd86725582af580925ca8242826ba2229`, which can be built outside
the active emulator development branch. For example, in fish:

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

# Use an existing built H5-compatible emulator (not the H6 macro submodule pin).
set rev e56779ccd86725582af580925ca8242826ba2229
set dll /absolute/path/to/H5-emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings /absolute/path/to/H5-emulator/bindings/python

python -m pytest -q tests/test_h7_readiness.py tests/test_shared_transformer_h7.py tests/test_h7_combat_ppo.py

# Four versus fifteen workers, identical fixed curriculum/source worlds and actor RNG.
python -u tools/benchmark_h7_rollouts.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 4 --episodes 32 \
  --output /tmp/h7-readiness-w4.json

python -u tools/benchmark_h7_rollouts.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 --episodes 32 \
  --output /tmp/h7-readiness-w15.json
```

The default is the actual full-size randomly initialized model and samples its
legal actions at temperature 0.85. To profile a frozen *trained* H7 actor, pass
`--player-checkpoint path/to/h7-player.pt` in both commands. Each worker loads
its own pinned CPU model. The learner's `--device cuda` option **does not move
worker inference to the GPU**.

The versioned JSON report includes a SHA-256 of actor weights, model dimensions,
requested/observed process count, replay-verified source coverage, per-cell
sample counts, total combats, public player decisions, total boundaries, wall
time and measured combats/decisions per second. A missing seed, duplicate
seed, unverified terminal, unexpected source or absent training cell aborts
report creation. This tool compares actual **rollout** throughput; it does
not claim GPU train-step throughput, steady-state memory use, or gameplay strength.

## Development order

1. **Throughput:** run 4- and 15-worker full-size H7 collection. Check CPU
   utilization, resident memory, model load overhead and per-player decision
   latency. Repeat with more than 32 episodes if startup dominates the run.
2. **Fix bottleneck before expensive training:** if 15 independent CPU
   inference copies do not scale, implement a bounded actor request queue
   and central **batched inference** owner (CPU or CUDA as benchmarked).
   Preserve deterministic per-source action sampling and frozen on-policy
   actor version per cohort. Keep the H6-F macro pathway untouched.
3. **Fixture expansion:** wait for a validated opt-in engine-owned A10
   fixture contract. Audit at least 200 distinct, *mechanically supported*
   configurations with coherent deck, encounter, relic, potion, HP and
   upgrade data. Split by actual configurations and independent worlds.
   Never fabricate 200 cells from 16 configurations by reusing seeds.
4. **Checkpoint equivalence:** a synthetic H7 contract test now compares
   resumed versus uninterrupted next-round model weights and AdamW buffers
   using the same freshly recaptured source actor. Next validate the same
   equivalence end-to-end on a pinned emulator, including Torch RNG, actor
   and source seeds. Retain old H7 and H4/H5/H6 formats.
5. **Full-size learning curves:** 15 rollout workers where feasible,
   independent scratch initializations, long runs with persistent
   per-round logs and checkpoint restoration. Compare 256x4 and
   384x6 models under matched source worlds, matched env-steps and
   matched compute. Evaluate confirmation and zero-shot separately.
6. **Later:** train a calibrated combat outcome model (win, exit HP,
   enemy progress, potion consumption) for long-horizon macro planning.

## Non-goals of this preparation PR

No training-quality claim, H6 retraining, Nature or MCTS experiments, checkpoint
format change, emulator feature-branch mutation, distillation, or fidelity
Inspector/native RNG prerequisite.
