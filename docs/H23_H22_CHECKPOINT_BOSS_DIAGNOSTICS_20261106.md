# H23 — Independent H22 checkpoint selection and boss PUCT diagnosis

**Status:** implemented experiment harness; no new CUDA evaluation has been run by this commit.
**Base:** H22 packed fixed-4 150-round Silent A0 generated-map training (P2 closed).
**Pinned emulator:** `203d5114024b55c69dabc309f10d6f196eac8cca`.
**Model source:** `results/h22-h21-packed-a0-long-v1/models/joint-rNNNN.pt`; do not copy weights into a fresh training job.

## Motivation

H22 completed 150 rounds and 9,600 training episodes. Heldout floor depth increased substantially but Act clears remained rare. The fixed 16-seed heldout cohort has already been used to inspect/select promising rounds, so **do not treat it as final generalization evidence**.

H23 asks two separate causal/measurement questions:

1. On **fresh identical seeds**, which trained checkpoint (round 95, 120 or 150) has the best Act-clear/boss-entry outcome?
2. Holding the checkpoint and *boss-entry public state* fixed, does a PUCT teacher at the first eligible boss decision improve the subsequent boss fight? Is depth-8 or full-combat 16/64 simulations better?

All H23 experiments are read-only against training files and use existing H21 packed deterministic CUDA policy. Search results are diagnostic only; any future training intervention is a separate experiment.

## Stage A: Paired 48-seed checkpoint selection

The existing `audit_h22_independent.audit_checkpoint` implements checksum verification, search-free execution and exclusion from training/heldout seeds. H23 wraps it for three checkpoints with the **same seed list** and a common paired report, storing per-checkpoint outputs so interrupted evaluations can resume between checkpoints.

Run in fish:

```fish
cd ~/projects/sts2-ai
git switch research/h23-h22-checkpoint-boss-diagnosis-20261106
source .venv/bin/activate.fish

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
test "$rev" = 203d5114024b55c69dabc309f10d6f196eac8cca; or begin
    echo "STOP: emulator revision differs from recorded H22 environment"
    exit 1
end

set progress results/h22-h21-packed-a0-long-v1/progress.json
set select_out results/h23-checkpoint-selection-v1
test -f $progress; or begin
    echo "STOP: H22 progress missing"
    exit 1
end

# Preflight: verify checkpoints and seed design; no CUDA work.
python tools/compare_h22_boss_checkpoints.py \
    --progress $progress --emulator-root $emu \
    --output-root $select_out \
    --round 95 --round 120 --round 150 \
    --seeds 48 --workers 15

# Execute: 144 search-free Act evaluations on 48 shared independent seeds.
python -u tools/compare_h22_boss_checkpoints.py \
    --progress $progress --emulator-root $emu \
    --output-root $select_out \
    --round 95 --round 120 --round 150 \
    --seeds 48 --workers 15 --execute
```

Report `selection-summary.json` includes per-checkpoint floor, boss reach, Act clears, boss HP at entry (completed floor-15 post-room HP), enemy HP removed in lost boss fights, and *paired seed-level comparisons*. The chosen checkpoint is a **selection candidate**, not a demonstrated generalization winner; at low Act-clear rates, even 48 seeds are imprecise.

A partial completed checkpoint output remains reusable on restart; files with inconsistent hashes, seeds or fingerprint fail closed. The runner refuses to overwrite a completed comparison report.

## Stage B: Boss-root search experiment

First run a small matched diagnostic on **12 new seeds**, distinct from the Stage-A selection and H22 training/heldout. Choose the Stage-A candidate (or manually nominate a frozen checkpoint if selection has no meaningful winner).

```fish
set selected (python -c 'import json;print(int(json.load(open("results/h23-checkpoint-selection-v1/selection-summary.json"))["candidate_for_boss_diagnosis"][1:]))')
set boss_out results/h23-boss-search-pilot-v1

# Preflight only.
python tools/diagnose_h22_boss_search.py \
    --progress $progress --emulator-root $emu \
    --round $selected --seeds 12 --workers 8 \
    --output-root $boss_out

# Compare four arms (same fresh seeds and frozen network):
#   policy       — no PUCT
#   depth8_16    — 16 simulations, 8 action-depth horizon
#   full16       — 16 simulations, to combat boundary
#   full64       — 64 simulations, to combat boundary
python -u tools/diagnose_h22_boss_search.py \
    --progress $progress --emulator-root $emu \
    --round $selected --seeds 12 --workers 8 \
    --output-root $boss_out --execute
```

The opt-in `search_floor=16` gate in the existing Act collector:
- permits at most one first eligible boss search per episode;
- disables search and search-related **RNG draws** at every earlier floor;
- leaves the original training code behavior unchanged when absent.

The report strictly checks **the first public boss snapshot hash and the entire pre-boss chosen-action prefix** for every seed across all four arms. All arms replay the same deterministic Native RNG seed. These checks establish the same observed boss-entry state and action prefix; they are not a proof of parity between independent stochastic Nature providers. The full search retains the 512-step watchdog and explicitly fails if unable to finish, rather than treating a cutoff as a win.

The metric hierarchy is:
1. conditional boss-clear rate across the **same reached-boss seeds**;
2. paired first-boss-action changes, plus boss HP removed for losses in both arms;
3. number of root searches, emulator search transitions and compute cost.

The main question is whether search **improves actual boss outcomes**, not whether its visit policy differs from the actor. Samples are deliberately small to reveal defects quickly. If >0 boss-entry states are seen and results are plausible, expand to 24–32 fresh seeds with a new tag and output root, prioritizing `policy`, `depth8_16`, `full16` first if full64 is too costly.

## Stage C: Untouched final checkpoint validation

Only **after selecting** the candidate from Stage A (and not from the final cohort), invoke the original independent checkpoint audit once with a different `--audit-tag`, on 32–64 additional seeds:

```fish
python -u tools/audit_h22_independent.py \
    --progress $progress --emulator-root $emu \
    --round $selected --audit-count 64 --workers 15 \
    --audit-tag h23-independent-final-v1 \
    --output results/h23-independent-final.json --execute
```

For a literal fish output path, prefer the following rather than the example filename interpolation above:

```fish
set final_out results/h23-independent-final-r$selected.json
python -u tools/audit_h22_independent.py \
    --progress $progress --emulator-root $emu \
    --round $selected --audit-count 64 --workers 15 \
    --audit-tag h23-independent-final-v1 \
    --output $final_out --execute
```

Do not rerun the final audit repeatedly to tune checkpoints or hyperparameters.

## Validity and limits

- Frozen H22 public-v1 model and emulator, deterministic CUDA actor, 15/8 workers per stage. No learning or optimizer mutations.
- Both selection and boss diagnostic seeds are distinct from historical training and heldout; stage C is an entirely separate seed namespace.
- The H16 search adapter is **root-conditioned and privileged**; the comparison diagnoses its usefulness *under that access*, not whether its value estimates are fair against stochastic Nature.
- Identical per-seed public boss state is checked. We do not claim that two paths with divergent earlier actions are the same boss state; such a result is a hard failure.
- Boss-entry HP is read from the first public boss combat state (Stage B) or as post-floor-15 HP (Stage A). The two measures should agree for each completed route; differences require a fidelity review.
- Failed full-combat watchdogs, emulator errors and censored episodes invalidate that arm; they are never counted as losses or wins.
- At 12 or 48 boss samples, uncertainty is high. Preserve per-seed paired details and negative/neutral results. Distinguish conditional boss-clear probabilities from unconditional Act clear rates.

## Next decision

If deeper PUCT significantly improves conditional boss wins on paired entries, add a separate **boss-specific search-supervision** experiment with more genuine boss roots per training sample and a cost/coverage budget. If the critic/search continues to fail from high-HP boss entries, test terminal combat credit, long-horizon value calibration and target distribution before further long training. If multiple independent checkpoints tie, avoid treating the historical best heldout floor as proof of superiority.
