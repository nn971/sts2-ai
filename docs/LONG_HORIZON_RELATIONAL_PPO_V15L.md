# v15l — Long-horizon relational combat PPO

**Purpose:** commit to substantial training before judging architecture quality.
The 5-round v15j smoke run did not establish a significant improvement over
v8 in its paired 739-snapshot comparison. That short pilot is evidence the
pipeline runs, **not** evidence against the architecture.

This plan deliberately does **not** gate continuation on each short-window
win-rate change. The baseline is the frozen v8 policy; only the v15 relation-
aware residual combat head trains. All inputs remain fair public observations
from the pinned native emulator (`9117b4af09f0164a19bb1c41b70f688c8948e0e3`).

## Phase A: complete 200 rounds, continuing the existing 5

**200 is the total round target, not 200 additional rounds.** With the
existing 5-round checkpoint, this adds 195 rounds. Each round uses 32 native
single-player episodes with 15 rollout workers: 6,400 runs over the complete
200-round history. Native combat PPO uses one optimizer thread, 64-wide
relational object features, batch size 64, three epochs, and a cap of 2,048
combat decisions per round. The frozen strategy head, base combat logits,
HP-preservation objective, and original vocabulary remain unchanged.

In WSL **Fish**, from the repository root (keep output and checkpoint files):

~~~
git fetch origin
git switch agent/v15l-long-run-training
git submodule update --init --recursive

python -m pytest -q tests/test_long_run_training_v15.py \
    tests/test_relational_ppo_v15.py

python -u tools/train_pooled_objects_v15.py \
    --base-model results/ppo-v8-batched-40/models/stage-0120.json \
    --architecture relational \
    --output-dir results/ppo-v15j-relational-cpu64 \
    --width 64 \
    --rounds 200 \
    --episodes 32 \
    --workers 15 \
    --vocabulary-runs 16 \
    --ppo-batch-size 64 \
    --ppo-epochs 3 \
    --ppo-sample-limit 2048 \
    --snapshot-every 20 \
    --resume 2>&1 | tee results/v15l-long-run.log
~~~

The original 5 rounds used these values and defaults (including seed 29,
temperature decay 200 rounds and the original training seed prefix). Changing
`--workers`, episode count, batch size, sample limit, learning rate,
temperature schedule, model width, train seed/prefix, base model, vocabulary
or emulator revision is **not a legal resume**. The full-checkpoint SHA-based
fingerprint rejects mismatches. Never edit the existing `checkpoint.pt`
or copy another run's checkpoint into this directory. If a training setting
must change, begin a **new experiment directory with no `--resume`**.

`--build` is optional for a first native emulator build; omit it if the
pinned .NET backend is already built. This command does not require CUDA.

## Durable artifacts and interruption handling

- `checkpoint.pt`: full weights, AdamW optimizer, Torch RNG and history;
  **atomic after each completed round**; the last fully completed round
  can be resumed even if killed later.
- `report.json`: atomically updated after every completed round; includes
  requested/completed round counts, per-round metrics and full training
  configuration. Prior 5-round rows from older checkpoint versions have
  `null` for diagnostics newly added in v15l.
- `models/stage-0020.json`, `stage-0040.json`, ...,
  `stage-0200.json`: portable standalone weights exported only after
  the round's checkpoint was committed. Each name means the cumulative
  number of completed rounds, **not** the number of training steps.
- `model.json`: latest published portable milestone; updated at each
  milestone and at a normal final exit. Between milestones, this can be
  older than the optimizer checkpoint. Use `rounds_completed` from the
  report, or the numbered model file, when interpreting evaluations.
- `results/v15l-long-run.log`: continuous terminal progress including
  current round, wins, all resolved combat decisions, number actually
  sampled for PPO, policy/value/entropy losses, and rollout/optimizer
  times. `python -u` and the trainer's `flush=True` show progress while
  the job is active. Some rounds may have long rollouts; don't interpret
  missing *within-round* output as a crash.

The first start with `--resume` loads the original frozen vocabulary
and checkpoint. Press Ctrl-C to stop. Re-run the identical command with
`--resume` to pick up from the last completed round. A non-recoverable
worker failure should not corrupt the last finished checkpoint. If all
requested rounds were already completed but `model.json` is missing,
running the same `--resume` invocation regenerates the final model/report
without collecting episodes.

## Monitoring without prematurely selecting checkpoints

Use *windowed* statistics, e.g. 20-round groups of 640 runs; do not infer
progress from a single 32-run count or react to momentary decreases.
The recorded `ppo_samples_used`, separate policy and value losses,
entropy, resolved-combat count, throughput, and any non-finite errors
provide training-health diagnostics. PPO loss magnitude alone is
**not** an agent-skill metric. Raw training wins are on changing on-policy
seeds and exploration temperatures, not a held-out comparison.

For long-term analysis, retain every 20th-round exported model regardless
of intermediate win rate. Read/report the actual model's training round
when comparing snapshots. Do not stop training on a small in-training
win-rate dip or select the best of 10–20 checkpoints using a tiny sample
without correction.

At the end, perform a paired assessment on the v14 authentic combat
snapshot corpus for **diagnostic comparison only** (it was already used
to inspect v15j and is therefore not a fresh holdout). Also evaluate full
runs on newly reserved native seeds to measure end-to-end agent strength.
Treat the frozen v8 baseline as a single matched comparator. If using
multiple milestones for selection, reserve a *separate unseen final*
cohort and evaluate just the selected model on that cohort.

## Future experiments (separate runs, not changes to this continuation)

The existing 5-round profile collected ~26,873 combat decisions but
used up to 10,240 in PPO (38%). Raising `--ppo-sample-limit` to 4,096
is plausible for better data use, but it changes the optimizer trajectory
and invalidates this run's checkpoint fingerprint. Evaluate this as a
separate 200-round control with frozen shared vocabulary and matched
seed/architecture settings rather than silently changing the continuation.

Keep GPU profiling independent of this experiment. v15j's best measured
CPU PPO update was 1,783 decisions/s, versus 1,611/s for CUDA, but
**rollout collection dominated complete-round wall time** and the
current training code runs on CPU. Another GPU optimization is not a
prerequisite for completing the long-horizon study.
