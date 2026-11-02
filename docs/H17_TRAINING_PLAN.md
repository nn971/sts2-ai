# H17: adaptive full-combat search and joint Act-1 learning

Status: **implementation branch**, following H16's R79 checkpoint and the depth-8 vs full-combat evaluation. The initial training run is contingent on CI and a small local smoke; no gameplay results are claimed here.

## Motivation and constraints

- Current H16 training uses 16 native full-Act runs/round, default 16 PUCT simulations and an 8-action depth cutoff. R79 full-combat PUCT16/64 outperformed depth-8 search on the paired evaluation.
- The H16 shared macro head **is trainable**. A frozen H6C model still provides a fixed logit prior (weight 0.5), and macro decisions receive no direct search policy targets. H17 anneals that prior, preserving the learned macro head.
- Current runs use the experimental fixed-route Act1 Overgrowth profile. Broader route selection requires separately verified emulator support. The present H17 run should not be described as broad map/path-policy training.
- The model architecture and R79 checkpoint remain unchanged; no external solver, action labels, distillation or neural Nature provider is introduced.
- Every deep PUCT simulation stops at the actual combat boundary; a 512-action watchdog raises an error rather than fabricating value/reward on nontermination.

## H17 pilot design (phases)

1. **Uncap combat PUCT depth:** use `search_depth=None`, full-depth tree selection, 16 or 64 simulations at selected roots. Simulations are **per decision root**, not per full-Act run. `search_depth=8` remains available for regression comparison.
2. **Adaptive root allocation:** target a 0/16/64 mix around 83/15/2% using the existing policy logits' normalized entropy, low HP and later floors, and a random audit component (to discover confident mistakes). A persistent card-type exposure histogram may increase sampling of underrepresented playable cards. Enforce reproducible RNG selection; track search roots by budget, actual transitions and overrides. Scheduler targets are approximate; cap cost/round.
3. **Macro teacher annealing:** begin from H6C weight 0.5 and decrease smoothly to zero over about 120 H17 rounds. H16's learned macro head remains trainable throughout. Log the actual weight each round; do not change H7/H16 network structure.
4. **Future HP:** preserve the geometric return and the strictly separated combat-victory/progress credit. H17 lambda starts at R79's evaluation value (~0.4114) and increases gently toward 0.77 over ~200 rounds. Continue training the separate Act-clear head and auxiliary random-horizon regression.
5. **Coverage:** increase fresh trajectories from 16 to 64/round once a smoke is healthy. Persist per-card exposure counters (in deck / playable / chosen / searched), plus phase/floor, and report the number of unique cards and action contexts. Keep enough unprioritized native runs to monitor generalization.
6. **Prioritized supervised replay (bounded):** retain a capped reservoir of public-state search-policy targets and value targets from completed episodes, weighted toward rare card-action contexts, difficult fights, and stale coverage gaps. Replay uses supervised visit-count CE and critic regression **only**, not off-policy REINFORCE; cap replay-age and per-round fraction; checkpoint for exact resume.
7. **Throughput guardrails:** report actual collection/train/evaluation seconds, emulator transitions/sec, actor service batching, unique card exposures/round and search transitions/round. Default to CPU emulator workers (15) and shared CUDA inference. If the smoke blows the cost budget, lower the searched-root fraction before sacrificing full-combat depth.

## Rollout and acceptance

- **Smoke:** 2 rounds x 16 fresh Act-1 runs; no PUCT search in fixed held-out evaluation; explicit option/fingerprint, checkpoint, and resumable replay/coverage assertions.
- **Initial training:** 40 rounds x 64 fresh runs with periodic larger policy-only held-out evaluations. Audit completion/censoring and sample diversity. Only increase runs/round if acceptable throughput.
- **Extended:** 150–200 rounds if policy-only results improve; do not assume each small task must improve performance.
- Evaluate policy-only floor7 survival, mean/max floor, Act-clear, per-floor deaths, combat enemy progress on defeat, coverage, and throughput. PUCT64 full-combat checkpoints are expensive and should be checked infrequently.
- Keep a frozen R79 baseline and fresh held-out seed sets. Original diagnostic seeds are *not* held-out for choosing H17.
- **Primary metric:** policy-only Act 1 outcomes on independent held-out seeds, not search-assisted wins on training seeds.

## Implementation / sequencing notes

This branch is based on `research/h16-paired-checkpoint-diagnosis-20261102` (draft PR #156). Use a stacked draft PR until the dependency merges. H17's first increment is the adaptive deep-search scheduler, macro-prior/horizon schedules, increased fresh-run budget, logging and resume; card-aware supervised replay can be staged after the core smoke. The full-route macro experiment is gated on native fidelity.

For fish shell use `set macro results/nature/h6c-macro-attention-r80.json`; retain the checkpoint `results/h16-v5-combat-credit-r150/models/joint-r0079.pt` and make a new H17 output directory.
