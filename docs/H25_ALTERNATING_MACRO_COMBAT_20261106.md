# H25 — alternating macro reinforcement / full-depth combat distillation

**Stage:** isolated experimental pilot on H24. Draft PR #175, stacked on
H24 PR #174. Source checkpoint: trained H22/H21 R95
`results/h22-h21-packed-a0-long-v1/models/joint-r0095.pt`.

## Intended learning process

1. Initialize independent macro and combat H24 Act-only networks from R95.
   They have separate `state_dict` instances, optimizers, and actor services.
   There are no shared mutable trainable parameters.
2. Collect *complete native Overgrowth A0 trajectories* using a frozen macro
   policy and frozen combat policy. **No PUCT** is run in this phase.
   Train the macro network with terminal Act-clear REINFORCE and Act-clear
   binary cross-entropy on *macro decisions only*. Begin with a low macro
   update frequency (every fourth round).
3. Sample the start of one or more encountered combats per full-Act run.
   Reconstruct their **exact native-state handles** by resetting the same
   seed and replaying every recorded action. Compare the complete public
   state and the exact legal action menu at each replayed prefix step;
   fail closed if any discrepancy occurs. A public observation is never
   treated as a resumable simulator snapshot.
4. From each sampled fight, run a new combat trajectory. Apply
   fixed-budget PUCT **at every decision with 2+ legal actions**, with
   `max_depth=None` (full fight), and continue until combat ends. Distill
   visit distributions into the **combat policy only**. A 512-step watchdog
   is a safety guard, not an early future-HP value target.
5. Repeat using the resulting combat policy to collect better full-Act
   experience. Increase macro-update frequency when measured combat
   completion and heldout performance support the transition.

There are no `p16` or `p64` probabilities in H25. Two independent
parameters determine search expense:
`--combat-fights-per-run` (how many fights to revisit per full Act) and
`--combat-simulations` (simulations at **each** eligible root). Search depth
is fixed to complete-fight; `--max-combat-decisions` is a watchdog for
the subsequently explored fight.

## Continuation value and reward

The macro network predicts binary Act-clear continuation `V_M(s)` with
hard HP+gold monotonicity from H24. Search uses the combat-only value

`V_C(s) = (1-eta)*V_M(s) + eta*clamp(HP/maxHP, 0, 1)`,

default `eta = 0.15`. The macro critic itself is unchanged by this
value adjustment. The existing combat search boundary reward remains:

- Win current fight: `alpha + (1-alpha)*V_C(post-combat state)`,
  default `alpha=0.25`.
- Lose current fight: `alpha * loss_progress_cap * enemy_damage_fraction`.

This HP bias is a learning prior, **not proof that the imperfect macro
critic ranks all successor states correctly**.

The combat actor learns only from PUCT visit-target cross-entropy.
The macro actor learns only from terminal Act-clear policy gradient and
BCE; no future-HP regression objective. Full-Act samples use no combat
PUCT. This is a phase-alternating experiment, not per-round end-to-end
joint backpropagation. Any PUCT actions are root-conditioned experimental
targets, potentially privileged under native chance.

## Very small local test (fish)

    cd ~/projects/sts2-ai
    git fetch origin
    git switch research/h25-alternating-macro-combat-20261106
    set emu ~/projects/sts2-emulator
    set rev (git -C $emu rev-parse HEAD)
    set h22 results/h22-h21-packed-a0-long-v1/models/joint-r0095.pt
    set out results/h25-a0-exact-replay-smoke-v1

    python tools/train_h25_alternating_a0.py \\
      --initial-h22 $h22 --emulator-root $emu \\
      --emulator-revision $rev --output-root $out \\
      --rounds 1 --runs-per-round 1 --heldout-runs 1 \\
      --workers 1 --combat-fights-per-run 1 \\
      --combat-simulations 8 --max-combat-decisions 128 \\
      --actor-max-batch 2 --actor-fixed-shape-batch-size 2

The first command prints the fingerprint without executing. Add
`--execute` once the parameters are confirmed. A fully searched fight
can be significantly more expensive than the H24 smoke; this pilot
tests replay fidelity and training integration, not high throughput.

For a bounded 15-worker first experiment, use a distinct `--output-root`,
`--workers 15 --runs-per-round 8 --heldout-runs 4 --rounds 4`,
`--combat-fights-per-run 1 --combat-simulations 16` and fixed-shape
batch 4. It contains up to 8 searched fights per round, but *unlimited
eligible roots within each sampled fight*.

### Live monitoring

    watch -n 10 'jq . results/h25-a0-exact-replay-smoke-v1/live.json'

The atomic `live.json` heartbeat updates every 20 seconds or after
a worker finishes. Aggregate search totals count completed worker tasks.
Additionally, `workers/worker-PID.json` is updated **after each completed
PUCT root**, recording the number of roots and transitions in the current
sampled fight. One individual full-depth root may still run for a long time.
Round-level outcomes and model checkpoints are committed to
`progress.json` and `models/` after heldout evaluation.


## Eight-hour combat-first experiment (fish)

This uses a **new study seed** instead of the H22 R95 selection/evaluation
cohort, 15 workers, a fixed 16-simulation budget at **every** eligible
decision in one sampled fight from each full-Act run, and no tree search
during full-Act rollout. The macro network updates every fourth round.

    cd ~/projects/sts2-ai
    git pull --ff-only
    set emu ~/projects/sts2-emulator
    set rev (git -C $emu rev-parse HEAD)
    set h22 results/h22-h21-packed-a0-long-v1/models/joint-r0095.pt
    set long_out results/h25-a0-combat-first-8h-v1
    set long_args \\
        --initial-h22 $h22 \\
        --emulator-root $emu --emulator-revision $rev \\
        --output-root $long_out \\
        --rounds 4096 --max-wall-seconds 28800 \\
        --runs-per-round 16 --heldout-runs 8 --workers 15 \\
        --study-seed 20261209 \\
        --combat-fights-per-run 1 --combat-simulations 16 \\
        --combat-hp-bias 0.15 --macro-update-every 4 \\
        --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \\
        --heartbeat-seconds 20 \\
        --keep-last-rounds 3 --archive-every-rounds 25 \\
        --device cuda

    python tools/train_h25_alternating_a0.py $long_args
    python -u tools/train_h25_alternating_a0.py \\
        $long_args --execute 2>&1 | tee results/h25-a0-combat-first-8h-v1.log

The dry run prints the fingerprint. The executable call should run in tmux
(or an equivalently persistent terminal). For interruption recovery, repeat
the exact parameters with `--resume --execute`; operational wall/heartbeat
controls are excluded from experiment identity. It resumes from the most
recent committed macro, combat, and optimizer state. The eight-hour target
is **total committed training wall time** across resumes.

A dashboard in another terminal:

    python -u tools/monitor_h25.py \\
        --root results/h25-a0-combat-first-8h-v1 --interval 30

One-shot output:

    python tools/monitor_h25.py \\
        --root results/h25-a0-combat-first-8h-v1 --once

The dashboard shows round clear counts, heldout floor reach, PUCT roots,
transitions, update steps, live worker completion, and the most recent
per-worker root counts. It is read-only and can be stopped independently.

## Tests and outstanding gates

- Focused CPU unit tests: policy routing, independent macro/combat
  parameters, combat search distillation, deterministic fight selection,
  and fail-closed exact prefix replays with a stub backend.
- Native replay fidelity must be verified against the pinned
  `sts2-emulator` revision; action IDs, public observations, and RNG
  continuation are checked on actual encounters.
- Real full-depth worker smoke, dual CUDA-actor throughput and search
  cost profiles are still pending.
- The initial design **does not auto-switch training focus**. Set
  `--macro-update-every 4` initially; promote to every round only
  after analyzing heldout combat metrics and Act-clear performance.
- Every committed round contains separate immutable macro, combat and
  optimizer snapshots, each SHA-256 verified on resume; the atomic
  `progress.json` write is the final round-commit marker.
- `--max-wall-seconds` uses cumulative active wall time across resumes
  and stops at the **next complete round boundary**. A single unusually
  long round can exceed the target time; stopping mid-root would discard
  that round's uncommitted work.
- To bound disk usage, retain the latest three rounds and checkpoints
  every 25 rounds by default. Historical metrics are retained in
  `progress.json`, even for checkpoints pruned from `models/`.
- Native full-depth scaling is still experimental. Validate the first
  15-worker cohort's throughput, memory and root progress before leaving
  the experiment running unattended.
