# H16 v2 — every-combat-decision PUCT and slowly growing geometric HP horizon

Status: **research implementation; focused CI and workstation smoke required**.
AI branch: `research/h16-act1-joint-puct-hp-potential-20261101`.
Emulator dependency: `research/act1-fixed-route-credit-pilot-20261101`
([PR #232](https://github.com/nn971/sts2-emulator/pull/232)).

## Target experiment

Silent A10, Act 1, a single explicit 16-floor synthetic route:

Combat, Combat, Unknown, Shop, Combat, Rest, Elite, Treasure, Unknown,
Combat, Rest, Combat, Elite, Unknown, Rest, Boss.

The fixed route varies encounters, event resolutions, combat RNG, rewards
and player policy across runs. A completed Act clear is recognized at the
certified floor-16 boundary. Defeat has Act-clear target 0. Censored
episodes are excluded from training; unsupported emulator actions fail
visibly. All player-model inputs use fair public observations.

## Restored combat search budget

H15 M18 used depth-8 search (with 32 simulations by default); other H15
reference experiments used 16 simulations, depth 8. H16 v2 explicitly
adopts the requested **16 simulations, depth 8, at every eligible combat
player decision**, starting in round 1.

- `--search-probability 1`: eligible combat roots are searched on every turn
  and after every other combat action, rather than just at turn starts.
- `--max-searches 0`: no per-run root cap. The 0 sentinel means unlimited
  *when* search probability is positive. With `--search-probability 0`
  no roots are searched.
- `--simulations 16 --search-depth 8`: up to eight player actions per
  simulation; PUCT exploration constant is 1.5.
- Only legal menus with at least two actions get search roots. A forced
  one-action menu does not require tree search.
- Search remains **combat-local**. Macro decisions train jointly through
  policy gradients; there is no macro PUCT. Held-out evaluation is
  search-free and measures the amortized player policy.
- PUCT uses full-depth expansion and value leaves. Run-state forks use
  exact emulator RNG, so the teacher is **potentially privileged**
  (`h16-native-act1-root-conditioned-privileged-v1`), not a certified
  public-history-fair conditional chance sampler.

This budget can be significantly more expensive than v1's 6% eligibility,
4 roots/run, 8 simulations and depth 3. The count scales with *actual*
combat decisions; it is not fixed at 4 roots or 8 simulations per run.

## Geometric horizon: short HP -> Act clear

For each player decision, start at the next room-completion boundary:
the current room when it is still in progress, the next room when the
current one is already completed, or floor 1 during Neow. Let
`h_1,...,h_R` be normalized HP at those future room-completion
boundaries through the Act boss. On a run ending before a boundary,
its HP entry and all later ones are zero. Let `Y` be the binary Act-clear
outcome.

The target at a continuous parameter `lambda in [0,1]` is

```text
G(lambda) = (1-lambda) * sum_{k=1}^R lambda^(k-1) * h_k
            + lambda^R * Y
```

This **includes the final boss-room HP term**, then assigns remaining
geometric mass to Act-clear. For R=0, target Y directly.
At lambda=0, G is the immediate **post-room** HP; at lambda=1, G=Y.
Intermediate targets change smoothly with lambda and give positive
future-floor feedback even on ultimately defeated runs.

A single horizon-conditioned sigmoid critic `V(s,lambda)` observes
public state plus `lambda,lambda^2` and is trained against G(lambda).
A separate Act-clear critic keeps BCE supervision on Y at all times.
The same completed trajectories also supply targets at an additional
random lambda per decision (default auxiliary loss weight 0.5),
thereby keeping the critic calibrated on longer horizons before the
policy itself begins relying on them. No extra game simulations are
needed for auxiliary horizon labels.

Unsearched actions use advantage `G(lambda)-V(s,lambda)` and a
phase-balanced policy gradient. Searched combat decisions use PUCT visit
cross-entropy, and the same V(s,lambda) at search frontiers.
Thus both macro and combat policy heads, the Act-clear head and the
horizon-conditioned value head update in one joint optimizer.

The training target is a Monte Carlo continuation objective, not a
temporal-difference fixed point or a policy-invariant reward-potential
transformation.

## Slow curriculum

Training round r (1-based) uses

```text
lambda(r) = 1 - exp(-(r-1) / tau),  tau = 200 rounds (default)
L(r)      = 1 / (1-lambda(r)) = exp((r-1) / tau)
```

Here L is the *untruncated* mean geometric horizon; on a finite 16-room
Act, the terminal-outcome mass is accumulated at the boundary.

| Round | Lambda (approx.) | Effective horizon (floors) |
|---:|---:|---:|
| 1 | 0 | 1.00 |
| 3 | 0.010 | 1.01 |
| 40 | 0.177 | 1.22 |
| 100 | 0.390 | 1.64 |
| 200 | 0.630 | 2.70 |
| 400 | 0.864 | 7.35 |

The schedule increases *very slowly*: the first 3-round pilot stays close
to post-room HP, and 100 rounds still average fewer than 2 future rooms.
Adjust `--horizon-tau-rounds` upward to slow it further. The limit as
rounds grow is Act-clear optimization.

Use `--hp-target post_room` as an optional matched control. This fixes
the **actor and search** lambda at zero but retains the same shared
architecture and auxiliary horizon supervision, isolating the policy's
horizon curriculum. The older `--hp-target immediate` control was
invalid (it used pre-decision HP); it is explicitly rejected.

## Workstation commands (fish)

```fish
set ai ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
cd $ai
git pull --ff-only
source .venv/bin/activate.fish
python -m pytest -q tests/test_h16_joint_act_pilot.py
or exit 1

set rev (git -C $emu rev-parse HEAD)
set h9 results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt
set macro results/nature/h6c-macro-attention-r80.json
```

The `--execute` flag launches emulator workers and training. Without
it, the runner prints a full config fingerprint and exits. The
following one-run smoke already enables full 16x8 search (the
search-free held-out evaluation is separate):

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-geometric-everydecision-smoke-v2 \
    --rounds 1 --runs-per-round 1 --heldout-runs 1 \
    --workers 1 --device cuda --execute
```

If the smoke passes, the first three-round feasibility pilot is:

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-geometric-everydecision-pilot-v2 \
    --rounds 3 --runs-per-round 16 --heldout-runs 8 \
    --workers 15 --device cuda --execute
```

For the fixed post-room-HP control, use the same command with a distinct
output directory:

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-postroom-everydecision-control-v2 \
    --hp-target post_room \
    --rounds 3 --runs-per-round 16 --heldout-runs 8 \
    --workers 15 --device cuda --execute
```

Each run writes `baseline.json`, `progress.json`, immutable round
checkpoints and optimizer state. Resume only with the exact same
parameters/fingerprint plus `--resume`. H16 v2 has an updated
checkpoint schema and **requires fresh output directories**; old v1
checkpoints and progress are not compatible with the new critic.

## Evaluation requirements

Monitor complete-run count, Act clears, failure floors, actual roots,
search transitions, override rates, potions and relic states. A full
rollout to Act clear or defeat supplies all labels; censored runs do not.
Compare the slowly growing geometric target with the post-room control
on matched seeds. With only three rounds, lambda is approximately 0.01:
this is mainly an integration pilot, not a test of long-range credit.
A longer comparison is justified only if the first pilot demonstrates
healthy throughput and stable updates.
