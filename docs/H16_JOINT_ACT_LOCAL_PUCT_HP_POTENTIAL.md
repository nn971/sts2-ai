# H16 pilot — full-Act joint learning with local PUCT and future-HP bootstrap

Status: **research implementation, awaiting workstation validation**.
AI branch: research/h16-act1-joint-puct-hp-potential-20261101.
Emulator dependency: research/act1-fixed-route-credit-pilot-20261101
([emulator PR #232](https://github.com/nn971/sts2-emulator/pull/232)).

## Question

Does a full-Act continuation signal make the shared combat/macro network
learn the value of retaining persistent resources, using fewer emulator
transitions than whole-Act tree search?

The **first full run is a complete Silent A10 Act 1** on the explicitly
hand-authored 16-floor route:

Combat, Combat, Unknown, Shop, Combat, Rest, Elite, Treasure, Unknown,
Combat, Rest, Combat, Elite, Unknown, Rest, Boss.

Every decision is made under the observable legal-action menu; maps are
fixed, but reference stochastic encounters, rewards, card choices, and
Nature outcomes still vary. Act-clear is recognized only by the certified
completed floor-16 boss / ActTransition observable boundary. A death has
terminal Act reward 0, a clear reward 1, and capped/unsupported runs
supply **no terminal labels**.

## Model and supervision

- Initialize the existing H9/H7 shared Transformer (source of both combat
  and macro action/value heads). The H9 tactical weights are retained.
- Attach two small context heads: a sigmoid Act-clear probability and a
  sigmoid predicted normalized HP at a future completed floor.
- Preserve full public decks, potion identities, relic identities and
  nested public persistent states using the H16 research-only codec.
  Macro legal action payloads are distinguished through public action
  semantics; existing H9 hand-card action tokens remain unchanged.
- Optionally apply the frozen H6 macro model as a **soft action prior**,
  incorporated in BOTH behavior sampling and its training log probability.
  It is never trained separately in this pilot.
- Train BOTH policy heads and BOTH continuation heads from the same complete
  Act cohort using one optimizer, not alternating frozen policy phases.

Let Y be the true Act-clear indicator. For a decision on floor f define:

    H_k(s_f) = HP after completed floor min(16, f+k) / MaxHP there

and H_k=0 if defeat occurs first. The critic trains V_A towards Y and
V_H towards the observed H_k, with k=2 as the default pilot. Neither
consumes a custom potion/relic scalar price. The actor uses

    G_alpha = (1-alpha) H_k + alpha Y
    A(s,a)  = G_alpha - ((1-alpha) V_H(s) + alpha V_A(s))

for NONSEARCH behavior-policy actions. The temporary mixture is a
bootstrapping curriculum, **not** a mathematically policy-invariant
potential-based shaping transformation. Replacing the mixture with pure
Act outcome training means alpha -> 1. There is no binary combat reward.

Every search-executed combat decision receives PUCT visit-distribution
cross entropy ONLY; its action is not incorrectly treated as sampled
from the raw actor for REINFORCE. Both classes of complete-run decisions
give value-head supervision. Macro and combat policy terms are averaged
by phase so the many micro-actions do not suppress macro learning.

## Local PUCT and fidelity

Opt-in searches sample a subset of eligible combat decisions (default 6%,
at most four roots per run) with 8 simulations/root and depth 3. Each
leaf uses the same blended continuation critic; reaching a combat exit
evaluates the complete observed post-combat state.

**Current limitation:** the full-run backend can fork exact emulator RNG
states but cannot yet draw independent correctly conditioned
public-history stochastic continuations for all native-structure run
sites. Search therefore uses
`h16-native-act1-root-conditioned-privileged-v1`.
It is an **approximate, potentially privileged exploratory search
teacher**. No such targets are labelled fair-native PUCT.
No PUCT is used for held-out performance measurement: evaluate the
amortized raw player policy on independent ordinary emulator runs.
This isolates whether distilled learning improves the player itself.

The fixed map is an **experimental synthetic layout**, not native
map parity. Emulator stochastic mechanics still own all random choices.

## Workstation commands (fish shell)

On the already configured machine (H9 R250 + H6 macro files remain local):

```fish
set ai ~/projects/sts2-ai
set emu ~/projects/sts2-emulator

cd $emu
git fetch origin
git switch research/act1-fixed-route-credit-pilot-20261101
git pull --ff-only
dotnet test tests/Sts2Emulator.Core.Tests -c Release --filter FullyQualifiedName~ExperimentalFixedRouteOvergrowthMapV111Tests
or exit 1
dotnet build src/Sts2Emulator.Cli -c Release
or exit 1

cd $ai
git fetch origin
git switch research/h16-act1-joint-puct-hp-potential-20261101
git pull --ff-only
source .venv/bin/activate.fish
python -m pytest -q tests/test_h16_joint_act_pilot.py
or exit 1

set rev (git -C $emu rev-parse HEAD)
set h9 results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt
set macro results/nature/h6c-macro-attention-r80.json
```

Preflight prints the complete pinned experiment configuration without
collecting runs:

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-fixedroute-futurehp-pilot-v1 \
    --rounds 3 --runs-per-round 16 --heldout-runs 8 \
    --workers 15 --device cuda
```

**Short executed smoke:** one training Act and one held-out Act, one
optimizer update, no search (search begins in round 2 by default):

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-fixedroute-smoke-v1 \
    --rounds 1 --runs-per-round 1 --heldout-runs 1 \
    --workers 1 --device cuda --execute
```

**Actual small pilot**, including local PUCT from round 2:

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-fixedroute-futurehp-pilot-v1 \
    --rounds 3 --runs-per-round 16 --heldout-runs 8 \
    --workers 15 --device cuda --execute
```

**Matched post-room HP control** (same seeds/budgets/map and initial
model; short-horizon action-dependent HP target, separate output):

```fish
python -u tools/train_h16_joint_act_pilot.py \
    --initial-h9 $h9 --macro-model $macro \
    --emulator-root $emu --emulator-revision $rev \
    --output-root results/h16-fixedroute-postroomhp-control-v1 \
    --hp-target post_room \
    --rounds 3 --runs-per-round 16 --heldout-runs 8 \
    --workers 15 --device cuda --execute
```

Both arms train the Act binary target as a separate critic head. The
only difference is the HP bootstrap target (two floors later versus
the next room-completion boundary). For decisions during combat or an
unfinished room, the control uses HP after that room completes. At a map
choice (where the preceding floor is already complete), it uses HP
after the next room. Neow's floor-zero decisions use floor 1; death
before the boundary gives zero. This replaces the invalid pre-decision
"immediate" HP target from the initial pilot. Both arms retain the
same architecture, local PUCT settings, temperature and macro prior.

To resume an interrupted run, repeat the identical command and add
`--resume`. Configuration and starting model fingerprints must match.
Train and held-out seed prefixes are separate. Progress, immutable
round checkpoints and optimizer-state snapshots are written in the
output directory. Never overwrite another experimental arm.

For a CPU-only integration smoke, use `--device cpu`, a fresh output
directory and `--workers 1`.

## Pilot evidence

Review `progress.json` for Act clears vs censored runs, raw held-out
Act-clear performance, joint optimizer steps for both heads, local PUCT
search counts, greedy-action override rate and search transitions,
resource inventory diversity and potion actions. Resource diversity is
a diagnostic of exploratory coverage, **not a direct success metric**.

The strongest signal of delayed-credit learning would be a
held-out advantage for future-HP over post-room HP under matched
resource-dependent states and independent Nature seeds. With 3 rounds
and eight held-out runs this is only a feasibility pilot; retain the
negative result if there is no evidence, then decide whether a larger
paired study is justified.

## Remaining constraints

1. A single realized future HP is a noisy sample of continuation value,
   though the fixed route removes map-topology variation.
2. Current stochastic search is root-conditioned and potentially
   privileged; source-fair native PUCT remains a separate RNG-fidelity task.
3. The existing shared codec remains a limited approximation to all
   persistent mechanics: public nested resource counters are added, but
   no promise is made about arbitrary unknown encounter-specific fields.
4. Three-Act wins are NOT yet the binary target; this first experiment
   ends at the certified full Act 1 boundary.
5. A full run that cannot complete due to an unsupported emulator
   action fails visibly rather than being scored as a death.

## Control-target correction

The original `--hp-target immediate` option supervised the policy with
HP at the pre-decision observation, which cannot distinguish alternative
actions from the same state in expectation. The corrected arm is explicitly
named `--hp-target post_room` and has a new output directory, preventing
accidental comparison or resume against the invalid old control. The
`--hp-target future` treatment and its definition are unchanged. Historical
immediate-HP results remain preliminary debugging records, not a valid
short-horizon control.
