# v15p — combat policy diagnostic suite

Stacked on v15o (#67). **Do not merge to main before review and CI.**
The aim is to understand why 200 rounds of importance-sampled combat PPO
changed policy behavior without convincingly beating frozen v8 on the
source-disjoint greedy combat snapshot benchmark.

This patch adds three independent, intentionally opt-in diagnostics:

1. **Offline greedy checkpoint comparison**:
   `tools/analyze_combat_checkpoints_v15p.py` consumes existing complete paired
   JSON evaluation reports, checks that corpus SHA256, baseline model SHA256,
   emulator revision, row ordering, baseline outcomes and decision cap match,
   and compares survival, exit HP and potion retention by tier/progress.
   Source-run cluster bootstrap handles snapshots sharing run-level decks/RNG.
   It writes JSON and human-readable Markdown. Enemy composition rankings are
   descriptive; many tiny groups and multiple checkpoint selection do not
   justify strong significance claims.

2. **Paired stochastic combat evaluation**:
   `tools/evaluate_stochastic_snapshots_v15p.py` replays genuine source groups
   against two public-observation policies under softmax selection at a fixed
   evaluation temperature. Each combat entry / replicate gets its own seeded
   actor; the two agents use matching uniform random streams with the same
   initial exact emulator RNG, but game outcomes are not forced to match
   *after actions diverge*. The benchmark runs independent per-source worker
   processes and returns the standard paired JSON/report format. Replicates
   are clustered by originating run, **never counted as new independent
   source runs**. Stochastic reports are separate from greedy reports; the
   replay tool does not modify the original snapshot recipe corpus.
   Pilot for a small number of runs before increasing replicates.

3. **PPO gradient diagnostics**:
   Training with `--diagnose-ppo` records mean and maximum pre-clip total
   gradient norm, proportion of minibatches clipped by norm threshold 1.0,
   raw weighted advantage standard deviation, and decision-level importance
   weight ESS fraction (after the existing sample-limit subsampling). Results
   are in each training round's report and progress logs. No extra gradient
   evaluations or hidden-game state are introduced; total gradient norm does
   **not** isolate policy loss from critic/entropy contributions. A fresh
   diagnostic run gets a distinct checkpoint fingerprint. Do **not** attempt
   `--resume --diagnose-ppo` on an existing v15o checkpoint created without
   this option. Resume works only within one matching experiment.

## Analyze existing v15o round 20–200 evaluations

Run in the repo root using Fish; the 6 evaluation JSONs already exist in
`results/ppo-v15o-importance-pilot/`:

~~~fish
python -u tools/analyze_combat_checkpoints_v15p.py \
  --evaluation 20=results/ppo-v15o-importance-pilot/evaluation-0020.json \
  --evaluation 40=results/ppo-v15o-importance-pilot/evaluation-0040.json \
  --evaluation 80=results/ppo-v15o-importance-pilot/evaluation-0080.json \
  --evaluation 120=results/ppo-v15o-importance-pilot/evaluation-0120.json \
  --evaluation 160=results/ppo-v15o-importance-pilot/evaluation-0160.json \
  --evaluation 200=results/ppo-v15o-importance-pilot/evaluation-0200.json \
  --bootstrap-samples 2000 \
  --output results/ppo-v15o-importance-pilot/v15p-greedy-diagnostics.json
~~~

Inspect `v15p-greedy-diagnostics.md` beside the JSON output.

## Stochastic-vs-greedy mismatch

Run a small pilot with 2 stochastic replicates on 10 **source runs** and
8 workers. Increase to 5 replicates and all source runs only after the
pilot verifies encounter completion/throughput:

~~~fish
python -u tools/evaluate_stochastic_snapshots_v15p.py \
  --corpus results/v15o-snapshots/partition/development.jsonl \
  --baseline-model results/ppo-v8-batched-40/models/stage-0120.json \
  --candidate-model results/ppo-v15o-importance-pilot/models/stage-0200.json \
  --temperature 0.7 --replicates 2 --workers 8 \
  --limit-source-runs 10 \
  --max-combat-decisions 512 \
  --report results/ppo-v15o-importance-pilot/stochastic-0200-pilot.json
~~~

The standard greedy comparison remains:
`tools/combat_snapshot_benchmark.py evaluate`.

The stochastic report has repeated rows per recipe and makes no assumption
that training-time temperature annealing is the best evaluation temperature.
For a comparable learning-curve check use the *same* evaluation temperature
and replicates/seeds for all checkpoint rounds. Do not compare a greedy win
rate against a stochastic win rate as though sampling uncertainty vanished.

## Controlled target-vs-proposal ablation

Lambda=0 and lambda=0.7 must share the *same beta=0.5 target*, base,
vocabulary, corpus, seed and optimization budget. Their objectives are
nominally equal; their simulation allocations differ. This is preferable
to comparing original `--snapshot-sampling balanced` with prioritized
importance, because their targets do not necessarily agree.

The launcher validates SHA256 of the input files and writes an immutable
`ablation-manifest.json` before running both arms, one after the other.
It streams each arm's log to the terminal and its own file. `--plan-only`
checks commands and provenance without invoking training.

~~~fish
python -u tools/run_importance_ablation_v15p.py \
  --base-model results/ppo-v8-batched-40/models/stage-0120.json \
  --vocabulary results/ppo-v15j-relational-cpu64/vocabulary.json \
  --snapshot-corpus results/v15o-snapshots/partition/train.jsonl \
  --output-root results/v15p-ablation \
  --rounds 40 --episodes 32 --workers 8 --plan-only

# Remove --plan-only to actually run 40-round target and priority arms.
~~~

For later continuation, use the same settings plus `--resume` and a
larger `--rounds` value, but note: the manifest intentionally fingerprints
the **round target**; changing that target currently requires a new ablation
manifest/output root. Prefer specifying the final round budget from the
beginning, or extend the runner's manifest to distinguish configuration from
mutable target. Merely copying a checkpoint to a new path is not a reliable
resume strategy.

Evaluate both arms **against v8 on the same held-out development corpus**
using the standard greedy snapshot evaluator, then compare their paired
outcome/HP curves. If prioritization helps, the lambda=0.7 arm should win
at equal combats. For statistical robustness, repeat with multiple seeds
and preserve source disjointness. Confirm on newly collected final source
runs only after choosing the experiment.

## Interpret diagnostics

- **Greedy worsens, stochastic improves**: exploration/evaluation mismatch,
  likely poor calibration of logits' argmax.
- **Both worsen**: objective/reward signal, credit assignment, or data coverage
  problem; inspect hard encounters and clipped gradient rate.
- **Gradient norms frequently clipped**: investigate batch size, learning
  rate and the simultaneous actor/critic objective before training longer.
- **Importance ESS stays high, neither arm wins**: the difficulty proxy
  does not allocate simulation more effectively, or the PPO signal is weak.
- **Hard-encounter setbacks**: look at elite/boss-only HP/survival and paired
  per-enemy outputs; do not infer significance from small encounter families.

Known limitations: exact simulator fidelity to native StS2 remains unverified,
stochastic evaluation currently has no resumable source-group journal
(unlike the original greedy evaluator), and no token-level action trace
is saved for diagnosing specific card-target choices. PPO gradient telemetry
is training-time only and cannot be reconstructed from the old round-200 JSON.


## v15q: run the two-arm ablation and automatic paired comparison

After fetching and switching to `agent/v15q-importance-ablation`, use the
same source-disjoint development partition employed in the 200-round v15o
greedy and stochastic evaluations. The runner checks that no
`(seed, ascension, source_policy)` appears in both partitions.

In Fish from the repository root:

~~~fish
python -u tools/run_importance_ablation_v15p.py \
  --base-model results/ppo-v8-batched-40/models/stage-0120.json \
  --vocabulary results/ppo-v15j-relational-cpu64/vocabulary.json \
  --snapshot-corpus results/v15o-snapshots/partition/train.jsonl \
  --eval-corpus results/v15o-snapshots/partition/development.jsonl \
  --output-root results/v15q-ablation-seed29 \
  --seed 29 --rounds 40 --episodes 32 --workers 8 \
  --eval-workers 8 --ppo-epochs 3 --ppo-batch-size 64 \
  --ppo-sample-limit 4096 --max-decisions 512 \
  --eval-max-decisions 512 --snapshot-every 10 --plan-only
~~~

Remove `--plan-only` and rerun the **same command** to execute sequentially:
40 rounds × 32 sampled authentic combats for lambda=0,
40 rounds × 32 for lambda=0.7, paired held-out greedy evaluation of the
final models against v8, and strict source-clustered A/B comparison.
This is **2,560 new combat rollouts** in total, plus replayed held-out
evaluations (which are not PPO training samples). Incomplete runs can
resume with the *same flags* and `--resume`.

At completion the runner writes:

- `results/v15q-ablation-seed29/ablation-manifest.json` — exact file hashes and
  experimental settings.
- `target/report.json` and `priority/report.json` — per-round PPO
  telemetry with `--diagnose-ppo` enabled.
- `target/evaluation-greedy.json` and
  `priority/evaluation-greedy.json` — full paired combat results
  using v8 as baseline and the **same** held-out corpus.
- `ablation-comparison.json` and `ablation-comparison.md` — direct
  prioritized-minus-target survival and exit-HP contrasts, clustered
  bootstrap intervals, tier breakdown and gradient/ESS diagnostic comparisons.
- `target.log`, `priority.log`, `target-evaluation.log`,
  `priority-evaluation.log`, and `comparison.log` — streaming logs.

A completed held-out report is reused only when its baseline, candidate,
corpus SHA256 and decision cap match the current inputs. Failed
evaluations retain their original resumable source-run journal. Existing
ablation manifests cannot silently be relabeled with different models,
source splits, or training budgets.

For manual re-analysis without rerunning either simulation:

~~~fish
python -u tools/analyze_importance_ablation_v15q.py \
  --target-train results/v15q-ablation-seed29/target/report.json \
  --priority-train results/v15q-ablation-seed29/priority/report.json \
  --target-eval results/v15q-ablation-seed29/target/evaluation-greedy.json \
  --priority-eval results/v15q-ablation-seed29/priority/evaluation-greedy.json \
  --manifest results/v15q-ablation-seed29/ablation-manifest.json \
  --output results/v15q-ablation-seed29/ablation-comparison.json
~~~

A single training seed is a **pilot**, not proof of the superiority of
either proposal. If the effect is promising, repeat both arms with several
independent training seeds and, after selection, collect a fresh final
source-disjoint evaluation set. Full native-StS2 fidelity remains unverified.
