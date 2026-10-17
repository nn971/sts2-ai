# Draw-only Nature: first serious single-combat experiment (2026-10-17)

**Implementation home:** `nn971/sts2-ai`; emulator branch is an independently versioned dependency.  
**AI delivery stack:** [PR #82](https://github.com/nn971/sts2-ai/pull/82).  
**Emulator target when written:** `nn971/sts2-emulator@90253aee9a62da76585a9b515d73aef6a0b7330b`, branch `feature/nature-143-native-policy-comparator`. Re-pin deliberately after any interface change.  
**Research status:** `NOT_ESTABLISHED`; declared-law reference probabilities are `UNVERIFIED` for the installed game.

## Hypotheses, in strict order

1. **H1 — fixed-player adversity:** a learned draw-only Nature produces a statistically measurable decrease in a genuinely fixed player's expected combat reward relative to a seeded `ReferenceNature` that samples `p`.
2. **H2 — robust player learning:** the player trained against that adversary improves *under ordinary/reference draws* relative to matched-budget ordinary-only PPO on an independent evaluation set.
3. **H3 — transport:** results hold on other encounters or different decks. This is a distinct later claim, not inferred from H1/H2.

H1 is the first serious run; **do not require or start joint player training before H1 is established**.

## Current emulator boundary

The emulator owns the combat mechanics and hidden deck constraints. Its external `semantic_*` contract publishes pending `enemy_intent` and `draw_card`, together with a read-only `chanceLaw` envelope: schema `sts2-v111-chance-law-v1`, law/event/site IDs, ordered `outcomes` with positive integer `weight`, `totalWeight`, conditioning and support fingerprints, and honest `probabilityStatus=UNVERIFIED`. The physical draw outcomes are `card_instance:<integer>`; intent outcomes are stable intent IDs. The emulator also supplies an independent seeded `semantic_reference_*` session, which samples all supported pending Nature sites.

**Draw-only training policy**: use the learned adversary `q` only on unresolved `draw_card`. At an `enemy_intent`, sample from the law's reported `p` using an independent, logged AI-side reference RNG, and resolve by the emulator's existing semantic intent resolver. Do **not** derive weights from card IDs, mock hidden shuffle order, run a second Nature shuffle, or claim native RNG cursor parity. Restrict the first benchmark to these two event kinds; reject unknown stochastic boundaries. Validate the law IDs and ordered outcomes against the actual pending candidates before every choice. For a pure-reference control, additionally exercise the emulator's `semantic_reference_*` sampler as an independent cross-check; a mixed external/control session is not exposed as a dedicated emulator mode.

The AI adapter is an experiment policy router, **not a new emulator mechanic**. Keep the old PR #82 two-site and native-style paired comparator modes working independently.

## Experiment design and progression

### Gate 0 — compatibility and capability verification

- Build the **exact** pinned feature commit separately; do not change the emulator repo/submodule in this AI PR.
- Probe `hello`, ensure `stochasticCapabilities`, `referenceNatureSessionSchemaId`, `semanticChanceSchemaId` and the `chanceLaw` envelope are present.
- Compare each reported `chanceLaw.outcomes` against the actual pending legal intent IDs or physical card-instance IDs, verify `totalWeight`, record `lawId`, `conditioningFingerprint` and `supportFingerprint`. Reject drift.
- Demonstrate a full real combat-to-terminal trace, source snapshot replay, pre-outcome fair player observation, and one versus 15 process-independent scheduling.
- Confirm **nontrivial draw event count**; simply completing combat does not establish a learnable draw adversary.
- Keep the old pinned `8cbb15a` regression jobs untouched until the new integration independently passes.

### Gate 1 — one well-defined combat

Start with a **fixed Act 1 Silent weak-enemy combat** (candidate: Shrinker Beetle). The combat opponent, deck, relics, ascension, initial HP and setup phase must be declared and reproducible; encounter randomness is sampled independently under reference law. Use the ordinary supported starter deck first to validate mechanics and quantify draw-event frequency.

If draws are strategically too sparse, define a **symmetric experimental loadout** with Backflip/Quick Slash and a small fixed deck. The latest semantic pilot supports their source draw continuations, but a fixed source-route extra-deck fixture with reference/control parity is not yet established. Do not silently reuse the seeded fixture with a different starting hand, or compare arms from different openings. A deliberately conditioned fixture may be used as a separately labeled scientific environment, subject to full joint replay and exactly matched player visibility.

Player actions: use the genuine legal action IDs, fail closed on unsupported effects, and explicitly version the admitted action scope. For the first repeatable frozen player, train under the *same* reference distribution, not a test-only random v15 initialization.

### Gate 2 — frozen player, learned draw-only Nature (H1)

- Train/freeze a competent combat player; pre-register its checksum, scope and scoring function.
- Train Nature against it while enemy intents remain reference sampled, with 5 independent Nature optimization seeds and initially 10–20k combats/seed, subject to wall-clock throughput and clear learning-curve diagnostics.
- Compare learned `q` with `p` and a simple fixed draw heuristic, using independent source seeds and independent player/intent action RNG seeds.
- Evaluate *out of sample*, and report return, win rate, exit HP, Nature draw entropy, `D_KL(q||p)` per event, counts of nontrivial draw decisions, fail-closed episodes and outcome/replay mismatch.
- Cluster bootstrap at source-run level and report across-restart spread; multiple evaluation actor seeds per trained Nature are needed to separate training from actor-sampling noise.
- Optionally use `min_q E_q[R_player] + lambda KL(q||p)` to limit deviation. `p` is the **declared emulator** distribution, not a native-game-calibrated law; no installed-game unbiased importance estimate.

### Gate 3 — adversarial PPO test (H2)

Only after H1: restart players from the *same genuine pretrained* checkpoint; compare reference-only PPO with draw-only-Nature PPO under matched episode budgets, actual player optimizer steps, sampled actions, and compute ledger. Evaluate **both frozen players under pure reference/ordinary randomness as the primary metric** on independent source groups and actor seeds. Reuse the replicated study and holdout protocols from PR #82, extending them to the versioned draw-only mode. Retain honest `decision_grade=false` until effect size, coverage, statistical power and native-law evidence justify upgrading it.

## Explicit blockers and independent tracks

1. New emulator compatibility has not yet passed `sts2-ai` cross-repo CI; latest emulator head moves rapidly.
2. Fixed-deck combat initialization plus repeated nontrivial draw choices must be verified before scaling training.
3. Genuine trained/frozen combat player needs to be produced or located; a test-only pooled v15 checkpoint is engineering evidence only.
4. Structural Chance Law correctness for the chosen draw constraints must have independent exact small-deck tests; self-sampling `ReferenceNature` is insufficient.
5. Full S3/S4 cross-provider equivalence, game-native `p` calibration (N5), and Inspector fidelity (#122) are separate acceptance tracks. They need not be completed to run an explicitly **declared-law emulator experiment**.

**Implementation order:** AI-side typed chance-law parser + independent reference-choice routing and tests; pinned latest-emulator probe/trace CI; frozen-player reference training; then the H1 five-seed serious run. No 5/20-combat smoke is scientific evidence.


## Gate 0 live result (2026-10-17)

- New AI implementation: `src/sts2_ai/nature/draw_only_p0.py` parses the **emulator-authored** Chance Law, checks weights/physical instances against the pending event, chooses enemy intents from a separate seeded `p` sampler, and delegates only physical `draw_card` outcomes to learned `TwoSiteNature`. `draw_only_episode.py` performs real source-routed combat, logs `p/q` and law identities, and replays a terminal snapshot. `tools/probe_nature_draw_only.py` supports the structural player or an opt-in immutable SHA-pinned exported v15 player; neither is a demonstrated competent player.
- Cross-repo workflow: [nature-draw-only-unified-law.yml](../.github/workflows/nature-draw-only-unified-law.yml), independently building **exact emulator commit `90253aee9a62da76585a9b515d73aef6a0b7330b`**.
- **Passed:** emulator Release build (zero errors/warnings), new Python Ruff and declared-law parser/reference-sampler unit tests. No Nature-trained performance claim.
- **Blocked before any player/Nature decision:** initial source-route combat creation fails inside the emulator's `semantic_begin_seeded` with `InvalidOperationException: Controlled turn-intent checkpoint disagrees with the automatic pipeline.` This is a saved automatic-pipeline/intent checkpoint validation failure in `PrototypeAutomaticPipelineStochasticStateV111.Validate`, not an AI policy failure. [Reproduction CI #39021533533](https://github.com/nn971/sts2-ai/actions/runs/39021533533). Reported under [emulator issue #143](https://github.com/nn971/sts2-emulator/issues/143#issuecomment-6232360281), leaving emulator source untouched.
- **Next gate:** obtain an emulator commit with that source-route initial intent validation fixed; deliberately re-pin CI and demonstrate a real terminal draw-only combat with nontrivial choices, then compare external reference sampling with the emulator's `semantic_reference_*` session. Only after that should the H1 serious training budget be spent. Preserve strict failure until then.


### Fixed-fixture compatibility check

An additional Gate 0 entry uses exactly the previously tested C# `SemanticSeededEntryV111` fixture for an isolated Shrinker Beetle combat (the explicit `seeded-fixed-shrinker` mode). This entry does **not** depend on an Act 1 map source route and would be appropriate for H1 as a declared conditioned combat.

However, the same pinned unified-emulator build now rejects this independently with `InvalidOperationException: Map-generated active combat room has no encounter-history entry.` ([CI #39022089161](https://github.com/nn971/sts2-ai/actions/runs/39022089161)). This happens *before* the AI draws any chance outcome; the latest run-state validation appears incompatible with a previously supported seeded combat fixture. Both failures are tracked under [emulator issue #143](https://github.com/nn971/sts2-emulator/issues/143#issuecomment-6232430740).

**Gate 0 remains blocked on emulator startup invariants across both tested entry modes.** Do not silently soften the C# invariants or construct a fabricated player state in Python. Once upstream provides a stable externally playable fixed combat or fixes the source-route validation, rerun the pinned C# probe and verify real draw decisions, terminal replay and legality before starting Nature learning.


## AI-side H1 training infrastructure added (2026-10-17)

New AI-only implementation, independent from the older two-site PPO experiments:

- `src/sts2_ai/nature/draw_only_p0.py`: validates typed emulator law support and weighs ordinary `enemy_intent` using an independently seeded reference sampler. Accepts `draw_mode=learned` for physical draws controlled by `TwoSiteNature`, or `draw_mode=reference` for a second independently sampled law-conforming draw arm. For the current pilot, reference physical draw weights **must be uniform** before its existing uniform-mixed softmax is admissible; fail closed on any changed law instead of silently changing the adversary's KL reference.
- `src/sts2_ai/nature/draw_only_episode.py` + `src/sts2_ai/nature/draw_only_workers.py`: complete actual C# combat, SHA-pinned frozen model, source-authored fair observation/legal action menu, exact reported law fingerprints/proposal likelihoods, terminal hidden-snapshot round-trip, and spawn-isolated 15-process batches. Neither the C# game state nor the player's observation is synthesized from Python.
- `tools/probe_nature_draw_only.py` accepts explicit `--entry-mode=source-route|seeded-fixed-shrinker` and `--draw-mode=learned|reference`; a second full scripted replay checks the complete actual choice/action sequence and resulting state fingerprints.
- `tools/train_draw_only_nature_h1.py` is the **opt-in, resumable H1 protocol**. Default is a read-only immutable plan, *not execution*. Given a competent frozen portable v15 checkpoint, `--execute` runs 40 × 256=10,240 genuine single-combat episodes for one training seed, updating **Nature only** between frozen-policy cohorts, then evaluates against the same model in 1,000 unseen source runs **per draw mode**. Separate `--training-seed` runs support 5 independent initializations. Train/evaluation source IDs use distinct namespaces; all holdout rows, episode digests, player/emulator SHA256, per-decision law IDs and log p/q, exact draw entropy/KL, exit HP, victory and source-resampled uncertainty are recorded. Actor RNG variance and native law calibration remain explicitly unresolved. Full-game-trained player checkpoint competence must be established separately; test-only checkpoints are rejected unless `--allow-untrained-smoke` is explicitly supplied.

Suggested fish commands *after the pinned fixed-combat emulator successfully reaches a terminal draw decision* (the current S3/S4 validation regression prevents it):

```fish
set dll /path/to/sts2-emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings /path/to/sts2-emulator/bindings/python
set rev (git -C /path/to/sts2-emulator rev-parse HEAD)
set model /path/to/genuinely-pretrained-v15.json

# Explicitly review the immutable plan before opting in:
python -u tools/train_draw_only_nature_h1.py \
  --player-model $model --emulator-dll $dll \
  --emulator-bindings $bindings --emulator-revision $rev \
  --training-seed 29 --rounds 40 --episodes-per-round 256 \
  --eval-episodes 1000 --workers 15 \
  --output-root results/nature/draw-h1-seed29

# Repeat with --execute once Gate 0 and initial player competence pass.
# Re-run identically with --resume after a saved round checkpoint.
# Independent Nature seeds: 29, 101, 211, 307, 401.
```

**H1 inference gate:** both arms must show source-compatible openings, actual legal draw selections, nontrivial deferred draws and complete terminal traces; a learned Nature should improve its adverse effect on *out-of-sample* score relative to the independent law-sampled control, with uncertainty reported across independent Nature training seeds. One benchmark/smoke, one PPO update, or zero observed draw opportunities cannot establish H1. The draw-only adversary currently uses a modest feature-table policy and supports conditioning on the physical candidate card model, so a richer contextual Nature model is a later extension.
