# 9700X / RTX 5070 / 24 GB RAM — first parallel training

> **Important after the map audit:** The historical commands in this
> guide ran the six-floor `legacy-prototype` environment. To reproduce
> those experiments, explicitly add `--environment legacy-prototype`
> to each command. The current CLI defaults to native-structure Overgrowth,
> but it will deliberately **stop with an unsupported-interface error**
> until the emulator JSONL bridge implements the requested reset. Do not
> resume the legacy checkpoint in native mode. See
> [native Overgrowth gate](NATIVE_OVERGROWTH_TRAINING_GATE.md).


The v3 neural learner uses **one GPU-free PyTorch learner** and `--workers`
**independent emulator worker processes**. Each worker starts a separate
pinned .NET JSONL bridge *without recompiling*, and receives the **same frozen
network** for an entire round. It returns only public observations, legal
action menus, sampled action indices and observed terminal outcomes.

This separation is intentional: actor-generated trajectories are valid
on-policy samples under the same network regardless of which worker finishes
first. Worker ordering cannot change the target corpus: the collector uses a
stable per-run SHA-256 actor RNG seed and preserves request order. No native
seed, exact hidden game state or oracle search result reaches the policy.

## Recommended starting configuration

For a Ryzen 7 9700X (8 cores / 16 threads), 24 GB system RAM and an
RTX 5070 with 12 GB VRAM, start with **four workers**. Run the tiny
current 128-dimensional/32-hidden actor on CPU, because the present
bottleneck is sequential JSONL gameplay and Python per-decision feature
extraction, not large GPU matrix multiplication. Each emulator bridge
is its own process, so increasing workers will consume more RAM. Use
four until throughput/memory measurements justify six; do not spawn
16 independent emulator processes simply because SMT exposes 16
logical CPUs.

### CachyOS / fish commands

From the `sts2-ai` repository checkout:

```fish
git pull --ff-only
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -e '.[neural,dev]'
dotnet --version
```

If PyTorch or your CUDA driver requires a different wheel, consult
the installed torch/CUDA version rather than assuming a wheel built
for the RTX 5070. This small baseline runs Torch on CPU; CUDA is not
required for it.

Start with a modest **four-worker 60-episode pilot**, including a
16-seed held-out diagnostic:

```fish
set -x OMP_NUM_THREADS 1
set -x MKL_NUM_THREADS 1
python tools/train_selfplay.py \
  --build --workers 4 --rounds 5 --episodes 12 \
  --max-decisions 2048 --dimension 128 --hidden 32 \
  --evaluate-seeds 16 --seed 19 \
  --checkpoint results/9700x-train.pt \
  --output results/9700x-model.json \
  --report results/9700x-report.json
```

If those runs complete at acceptable CPU utilization and memory
consumption, extend the **same** training process to 20 rounds (total
240 episodes in this example) without losing the optimizer state:

```fish
python tools/train_selfplay.py \
  --workers 4 --resume --rounds 20 --episodes 12 \
  --max-decisions 2048 --dimension 128 --hidden 32 \
  --evaluate-seeds 64 --seed 19 \
  --checkpoint results/9700x-train.pt \
  --output results/9700x-model.json \
  --report results/9700x-report.json
```

For the subsequent **640-episode** experiment, start a **new**
checkpoint with 20 rounds × 32 episodes; changing
`--episodes` for an existing checkpoint is prohibited because it
changes the policy-gradient batch objective.

## Checkpoint and resume semantics

- The checkpoint is atomically replaced **after each completed
  on-policy round** and contains the complete neural weights,
  AdamW momentum/state, Torch CPU RNG, initial model and all
  completed-round metrics.
- Interrupted work **within the current round is not recovered**.
  The next invocation restarts only that unfinished round, with
  the same episode seeds and the same policy snapshot.
- Exact resumption requires matching learning hyperparameters,
  seed, seed prefix, network shape, batch size, and pinned emulator
  revision. The *total number of rounds* may be increased.
  Worker count may also change, with stable per-episode actor seeds.
- Checkpoints use Torch `weights_only=True` loading and should be
  treated as local research artifacts. Do not load unrelated files.
- The training report contains model ID, emulator revision, worker
  count, round metrics, completed/censored/victory counts and wall
  time for the **current invocation**. Holdout seed sets are disjoint
  from training and comparable across the four agents.

## Validation and limitations

CI covers deterministic toy-policy learning, exact equality between
uninterrupted and checkpoint-resumed training (including optimizer
state), fail-closed incompatible configuration, public-only inputs,
and an actual **two-worker pinned Silent emulator** smoke with
checkpoint artifact.

The existing neural value model is deliberately tiny; this work is
primarily about scaling the number of *real trajectories* first. It
does not implement asynchronous stale-policy actors, PPO, learned
chance distributions, GPU-batched inference, distributed nodes or
native STS2 calibration.

To assess a throughput improvement, compare
`training_wall_seconds_current_invocation` on identical
machine/seed/training settings with `--workers 1`, `4` and `6`.
Wall times from GitHub-hosted runners are not reliable estimates for
your workstation. A higher worker count can lose performance if
Python serialization, bridge startup, RAM pressure or the learner
become limiting.

The 12 GB GPU will become relevant when card/entity attention and
larger batched updates justify it. We should profile CPU rollouts
first rather than moving the entire pipeline onto CUDA prematurely.
