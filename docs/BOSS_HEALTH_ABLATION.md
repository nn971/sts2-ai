# Act-1 boss HP ablation (public-only Silent / native Overgrowth)

## Purpose

Our earlier warm-start self-play reached 86 completed recovery rounds (2,752
episodes) after the prior 45 rounds but still had **zero wins**. Aggregate floor
progress cannot distinguish dying with 95% boss HP left from dying with 2%.

The boss diagnostic is based **only on the player's public observations**; it
does not inspect emulator internals, future RNG, unrevealed intents, or the
original game. No missing emulator feature is silently substituted.

The current Act-1 Overgrowth boss fights include **Ceremonial Beast, The Kin
and Vantom**. The tracker freezes the roster and HP seen on the first
observable boss-combat frame at floor 16 and records the surviving HP of
those starting enemy instances on defeat. For The Kin, the starting roster
contains multiple enemies. Summoned additions are excluded from the
denominator. This is an initial-roster enemy-HP proxy, **not** a claim that
native boss phases, revives, shields or summons are all accurately valued.
A loss before boss combat has no boss-health score.

Each completed training round records `boss_entries`, `boss_defeats`,
`boss_near_kills` (80% initial HP removed), the average damage fraction
*conditional on losing during boss combat*, and individual
`boss_defeat_records`: seed, encounter ID, starting HP, remaining HP, and
damage fraction. Censored episodes and emulator errors get no reward.
Fixed-seed evaluation records the same boss HP data per seed as well
as counts and means for completed runs.

### Explicit opt-in shaping

Let `q=1-remaining_hp/initial_hp` for a defeat inside the Act-1 boss fight
with a tracked roster. With `--boss-damage-weight w`:

```text
progress = (16 - w*(1-q))/28    [native boss-combat defeat only]
```

The rest of the reward remains unchanged:

```text
R = 1                         [full victory]
R = alpha*(0.7*progress+0.3*remaining_HP_ratio) [completed defeat]
```

`w=0` (default) reproduces the old behavior. `w=1` gives progress
`(15+q)/28` on boss death: more boss damage means more credit, yet a
loss during the boss is NEVER worth more floor progress than clearing
Act 1 and reaching Act 2 (at least `16/28`). This is intentionally
conservative: it **does not** add a large reward or a separate optimizer.
Other defeats remain unchanged, and a boss clear still contributes via
regular full-run progress. The annealed auxiliary coefficient `alpha`
continues to depend on actual completed victories.

The weight and training version enter the optimizer checkpoint fingerprint.
Do **not** `--resume` a checkpoint created before this change; export
portable weights to start a fresh experiment instead.

## Next training task: controlled paired trial

On CachyOS with fish, from `sts2-ai`:

```fish
git pull --ff-only
git submodule update --init --recursive
fish tools/run_boss_ablation.fish
```

This creates the missing portable warm-start JSON from the existing
`results/native-tempered-recovery-255.pt` checkpoint (86 completed
recovery rounds). It then trains **two independent 50-round × 32-episode
experiments** on the pinned emulator, using **15 isolated workers**. All
hyperparameters, RNG seed 45, temperature schedule 0.05→0.035 across 30
rounds, and evaluation seeds are identical; the **only reward change**
is `--boss-damage-weight 0` versus `1`.

- Baseline: `results/native-boss-control-50.pt`,
  `results/native-boss-control-50-report.json`,
  `results/native-boss-control-50-report.monitor.jsonl`,
  `results/native-boss-control-50.log`
- Shaped: `results/native-boss-shaped-50.pt`,
  `results/native-boss-shaped-50-report.json`,
  `results/native-boss-shaped-50-report.monitor.jsonl`,
  `results/native-boss-shaped-50.log`

Every round prints to terminal and the respective log through `tee`.
Every tenth round plus round 1 evaluates greedy play on the same 64 held-out
seeds and atomically records full per-seed boss diagnostics. The wrapper
fails on an emulator error (no false defeat reward); rerunning resumes from
the last complete-round checkpoint, provided neither code nor hyperparameters
changed. It skips arms with a completed report.

When both experiments complete, compare fixed-seed checkpoints:

```fish
python tools/compare_boss_ablation.py
```

**Selection criterion**: prioritize actual Act-1 clears and full wins.
Next compare boss-entry rate and paired completed-run frontier delta.
Boss damage fraction *conditional on boss death* is diagnostic, not
alone a fair success metric; it can increase simply because the policy
arrives at harder boss scenarios or fails in a different way. No boss
entries means this test cannot determine whether shaping helps.

The parallel trials each have their **own optimizer**, but every arm
starts from the identical saved model. The 50-round baseline is an
ablation, not an additional continuation of the same optimizer.
If an arm is promising, the training command can be resumed to a longer
run by increasing only `--rounds`, keeping all other options identical.

## CUDA decision

The current 128×32 network runs per public decision inside process-isolated
simulator workers. CUDA is not automatically faster on individual tiny
inferences. Each training round now prints `rollouts=…s` and
`optimizer=…s`, which can identify whether optimization is expensive
enough to warrant batching onto the RTX 5070. A future cross-worker batched
inference architecture, not a GPU flag on the current loop, is likely to be
the useful acceleration path.
