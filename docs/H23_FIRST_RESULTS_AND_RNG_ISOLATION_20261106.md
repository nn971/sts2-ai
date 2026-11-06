# H23 diagnostic findings — 2026-11-06

Status: first evaluation analyzed; follow-up RNG-isolated boss experiment prepared. All figures below come from the user's completed local `selection-summary.json` and `boss-comparison.json` artifacts, each with H22 fingerprint `b4894d6a234f73c84df85472b2ef6bb1117a30c1851ba36f5db5097ad4e2c9f1`. This file records conclusions without claiming those files are committed to Git.

## Paired 48-seed search-free checkpoint selection

| H22 checkpoint | Mean max floor | Boss reached | Act clears | Conditional boss-clear rate | Mean boss-entry HP fraction |
|---|---:|---:|---:|---:|---:|
| r0095 | 13.3333 | 29/48 | 2/48 | 2/29=6.90% | 71.81% |
| r0120 | 12.2292 | 21/48 | 2/48 | 2/21=9.52% | 77.48% |
| r0150 | 12.9792 | 20/48 | 1/48 | 1/20=5.00% | 80.49% |

- R95 reached boss exclusively on 12 seeds vs 4 exclusive for R120; R95 exclusively on 13 vs 4 for R150.
- Both R95/R120 Act wins were **the same seeds, 29 and 41**. R150 cleared only seed 41.
- Selection rule chooses **R95**, primarily for boss reach and mean floor rather than a demonstrated conditional boss-combat advantage.
- These 48 seeds were used to **select** the model. Never use them as fresh final validation.

## First boss PUCT pilot — 12 Act episodes, 5 boss entries

| Arm | Boss wins | First boss actions changed vs policy | Search transitions | Mean boss HP removed on defeat |
|---|---:|---:|---:|---:|
| policy | 0/5 | -- | 0 | 59.595% |
| depth8_16 | 0/5 | 0/5 | 692 | 55.433% |
| full16 | 0/5 | 0/5 | 3,090 | 55.433% |
| full64 | 0/5 | 0/5 | 12,312 | 55.433% |

- All 5 boss-entry public states and preboss action prefixes matched across arms.
- PUCT ran and consumed simulation transitions but **never changed the initial boss action**; all three PUCT arms had the same outcomes and aggregate damage.
- First pilot has an experimental confound: in historical H16 collector, PUCT seed initialization consumes a draw from the *behavior action-sampling RNG*. Therefore later sampled actions can change even when initial boss action agrees. The observed 4.16 percentage-point lower damage under PUCT must **not** be interpreted as a search-caused regression.
- At most **one** root per episode was searched. This tests an early boss intervention, not repeated PUCT control throughout boss combat.
- Zero wins on five entries has very wide uncertainty; there is no positive evidence for deeper PUCT here, but it does not establish its inferiority.

## H23 corrective follow-up

The H23 diagnostic code now accepts opt-in `independent_search_rng` and derives the PUCT seed from (native run seed, decision index) using BLAKE2b, without advancing the outer player action sampler. The historical trainer and older PUCT behavior remain the default. The boss comparison runner opts into independent search RNG, adds the RNG provenance to its identity, and records the first boss root's search distribution/argmax/chosen probability alongside the frozen actor's action distribution. The report computes total-variation distance between the two policies and whether their greedy actions differ.

Run the follow-up on a **different seed tag and output root**, avoiding collisions with old artifacts:

```fish
cd ~/projects/sts2-ai
git switch research/h23-h22-checkpoint-boss-diagnosis-20261106
git pull --ff-only

set emu ~/projects/sts2-emulator
set progress results/h22-h21-packed-a0-long-v1/progress.json
set boss_out results/h23-boss-rngisolated-v2

# Dry-run identity and frozen model digest.
python tools/diagnose_h22_boss_search.py \
    --progress $progress --emulator-root $emu \
    --round 95 --seeds 24 --workers 8 \
    --tag h23-boss-rngisolated-v2 --output-root $boss_out

# Run frozen-policy and all search budgets with independent PUCT RNG.
python -u tools/diagnose_h22_boss_search.py \
    --progress $progress --emulator-root $emu \
    --round 95 --seeds 24 --workers 8 \
    --tag h23-boss-rngisolated-v2 --output-root $boss_out --execute
```

The pairwise action-prefix and boss-entry hash checks remain mandatory. Evaluate `boss-comparison.json` for first-action differences, boss wins, enemy HP removed conditional on both losses, actor-versus-PUCT probability shifts, and transition cost. If the search distributions differ but sampled first actions do not, a deliberately separate deterministic-action intervention (e.g. teacher argmax at root) may be needed to identify the teacher's advantage. Do not modify the player's normal evaluation distribution in-place.

## Decision

**Retain R95 as a selection candidate and freeze all H22 checkpoints.** Prioritize understanding why PUCT does not change observable boss decisions, then test several boss decisions per encounter if single-root search remains ineffective. Use untouched final-audit seeds only after checkpoint selection and any new method tuning are complete.
