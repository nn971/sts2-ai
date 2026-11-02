# H16 v5 fresh-seed checkpoint diagnosis

Purpose: check whether the apparent floor-7 improvement at R107 generalizes
beyond the eight repeatedly evaluated held-out seeds, and whether local PUCT
adds value beyond the frozen actor. Read-only: the evaluator never updates
checkpoints or the training progress file.

## Experimental design

- Checkpoints: **R39, R79, R107** from the same H16 v5 run.
- **Identical fresh seeds** across all six arms (3 checkpoints x 2 modes).
  The default prefix `h16-v5-fresh-paired-20261102` is disjoint from the
  `h16-fixedroute-train-...` and `h16-fixedroute-heldout-...` cohorts.
- **Policy arm:** original stochastic actor at temperature 0.85 with no search.
- **PUCT arm:** the same actor with 16 simulations, depth 8, search at every
  eligible combat decision, using the original seeded sampling implementation.
- **Common horizon:** all arms use the R107 geometric lambda (~0.4114) for
  critic evaluation. The policy logits are horizon-independent; PUCT values
  are not. Fixing lambda prevents the horizon schedule from confounding the
  checkpoint comparison. Override with `--horizon-lambda` for an ablation.
- Native-structure emulator (ordinary prototype RNG), fixed-route Silent A10,
  the original pinned emulator revision and fixed macro-model prior.
- Run 64 fresh seeds as a pilot; optionally run an independent 128-seed
  confirmation cohort with a **different seed prefix and output**.

## Preflight and execution (fish shell)

Run from `~/projects/sts2-ai` with the matching emulator under
`~/projects/sts2-emulator`. Set `macro` to **the exact macro checkpoint
used in H16**, whose SHA256 is recorded in progress.json; the evaluator
enforces the digest. All three H16 checkpoints must be present in the
`results/.../models/` directory.

```fish
cd ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
set progress results/h16-v5-combat-credit-r150/progress.json
set macro /absolute/path/to/the/original/h16-macro-checkpoint.pt
set out results/h16-v5-combat-credit-r150/diagnostic-r39-r79-r107-fresh64.json

# Validate inputs, model digests and evaluation manifest; no rollout.
python tools/diagnose_h16_paired_checkpoints.py \
    --progress $progress --emulator-root $emu \
    --macro-model $macro --output $out

# Optional tiny end-to-end smoke on a *separate* new seed prefix.
python tools/diagnose_h16_paired_checkpoints.py \
    --progress $progress --emulator-root $emu --macro-model $macro \
    --seed-prefix h16-v5-diagnostic-smoke-20261102 --seeds 1 \
    --workers 2 --batch-size 1 \
    --output /tmp/h16-v5-diagnostic-smoke.json --execute

# Full 64-seed diagnostic; safe to rerun after interruption.
python tools/diagnose_h16_paired_checkpoints.py \
    --progress $progress --emulator-root $emu --macro-model $macro \
    --seeds 64 --batch-size 16 --workers 15 \
    --output $out --execute
```

The output contains `identity`, `analysis.arms`,
`analysis.paired.puct_vs_policy`,
`analysis.paired.checkpoint_progress`, and per-seed `runs`.
An append-only `<output>.partial.jsonl` journal enables resumption after
a crash; changed model bytes or evaluation settings invalidate the journal.
Use separate output and seed prefix for confirmation.

The diagnostic may run alongside ongoing training because it reads immutable
checkpoint files, but concurrent use of the same GPU will reduce throughput.
It is generally preferable to run it during a pause in training.

## Interpretation

Primary: paired floor-7 survival (including win-only-on-one-arm counts),
post-floor-7 HP, mean max floor, and Act clears. Secondary: DefeatDamage
**stratified by death floor**, with the number of defeats at each floor.
DefeatDamage can decrease when the policy starts converting marginal losses
into wins.

Checkpoints improve on genuinely new seeds **without PUCT**: evidence of
policy learning. PUCT greatly improves the same checkpoint: useful planning
signal not fully internalized by the policy. Neither helps: search quality or
the underlying policy/value representation may be bottlenecked. Small
differences in a 64-seed pilot warrant confirmation, not a strong claim.
The default PUCT implementation uses privileged exact-state emulator forks;
results are **not** public-history-fair.
