# H23 second boss-search pilot — 2026-11-06

Source: user-provided `boss-comparison(1).json`; native Silent A0 generated maps, frozen H22 R95 SHA256 `c89fe7e8811316ce6057936fb223e378d198930019beefd9a695c9ff890b4fe2`, emulator `203d5114024b55c69dabc309f10d6f196eac8cca`, 24 Act-1 seeds (`h23-boss-rngisolated-v2`), 11 paired boss entries. No censoring is reported by the per-arm aggregate, but episode completeness is **not included in the summary** and must be checked in the raw arm files if material.

| Arm | Boss reached | Boss wins | First boss action changed from policy | Root overrides (teacher argmax) | Search transitions | Mean boss HP removed on lost fights |
|---|---:|---:|---:|---:|---:|---:|
| Policy only | 11 | 0 | – | 0 | 0 | 55.7879% |
| PUCT, 16 sims, depth 8 | 11 | 0 | 4/11 | 3 | 1,522 | 53.5423% |
| PUCT, 16 sims, full combat | 11 | 0 | 4/11 | 3 | 6,908 | 53.5423% |
| PUCT, 64 sims, full combat | 11 | 0 | 4/11 | 3 | 27,927 | 53.5423% |

**Interpretation:** At least some searched first boss decisions changed relative to the policy, but no arm won. Each searched arm has identical *aggregate* results; this report alone cannot establish exact per-seed search-policy or trajectory equality, and it does not report root policy probabilities or actor-vs-search total variation. Mean damage difference (search − policy) was −0.0224563 on jointly lost boss encounters; the evidence does not establish that search caused it.

**Provenance gap:** The uploaded JSON identity has **no** `search_rng` field, although current H23 branch code records `search_rng: independent_deterministic_v1`. The arm summary also lacks `mean_actor_teacher_tv`, `boss_roots_teacher_top_differs_actor_top`, and per-root search policy masses. Therefore the run evidently used an earlier version of the H23 script, despite `rngisolated-v2` in the seed tag and output name. The intended independent search-vs-behavior RNG fix is **unverified** in this data.

The next experiment should use the latest H23 branch, verify `search_rng` identity and collect **actor-vs-search total-variation** at the first boss root. Reuse this exact 24-seed tag for a paired *repair comparison* with a new output directory, to eliminate seed-set variation; retain separate untouched seeds for any confirmatory generalization study. For compute efficiency first run `policy`, `depth8_16`, `full16` without `full64`, which spent ~18.35× as many transitions as depth8 without any measured aggregate benefit. Examine the raw per-seed arm files to distinguish sample variance, distribution shift, and possible identical teacher output across budgets.

No changes to H22 training, checkpoints or loss/reward are supported by these 11 boss entries.
