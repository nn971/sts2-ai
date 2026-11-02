# H16 v5 R79: PUCT 16 versus 64 simulations

This is a **read-only** budget ablation. The existing `paired-diagnostic.json`
already contains 64 runs each for the R79 policy-only and 16-simulation PUCT
arms. Only the **64-simulation** arm needs to be executed.

The same frozen R79 checkpoint, Silent A10 fixed-route map, emulator revision,
macro prior, critic horizon (lambda = 0.4113950303216448), depth (8), sampled
PUCT visits, and 64 run seeds are used. The runner checks the reference JSON
SHA256, checkpoint SHA256, macro checkpoint SHA256, and training fingerprint.

## Fish shell: smoke, pilot, full comparison

From the `sts2-ai` repo after pulling
`research/h16-paired-checkpoint-diagnosis-20261102`:

```fish
cd ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
set progress results/h16-v5-combat-credit-r150/progress.json
set reference results/h16-v5-combat-credit-r150/paired-diagnostic.json
set macro /absolute/path/to/the/original/h16-macro-checkpoint.pt
set output results/h16-v5-combat-credit-r150/puct16-vs64-r79.json

# Preflight only. Validates reference and frozen model digests.
python tools/diagnose_h16_puct_budget.py \
    --reference $reference --progress $progress \
    --emulator-root $emu --macro-model $macro --output $output

# Two-seed smoke on the ACTUAL 64-seed cohort.
python tools/diagnose_h16_puct_budget.py \
    --reference $reference --progress $progress \
    --emulator-root $emu --macro-model $macro --output $output \
    --limit 2 --execute

# Extend the same journal to 16 seeds; inspect preliminary printed results.
python tools/diagnose_h16_puct_budget.py \
    --reference $reference --progress $progress \
    --emulator-root $emu --macro-model $macro --output $output \
    --limit 16 --execute

# Complete all 64 seeds, REUSING the existing 16 evaluations.
python tools/diagnose_h16_puct_budget.py \
    --reference $reference --progress $progress \
    --emulator-root $emu --macro-model $macro --output $output \
    --limit 64 --execute
```

The journal is `<output>.partial.jsonl`. Each subsequent call resumes
without re-running completed seeds; keep worker and batch-size settings
unchanged across calls. A JSON report is written on completion.

The resulting JSON includes:

- `analysis.summary.policy`, `analysis.summary.puct16`,
  `analysis.summary.puct64`: floor-7 survival, Act clears, max floor,
  post-floor-7 HP, per-floor defeat damage, and search transitions.
- `analysis.paired.puct64_vs_puct16`: wins gained/lost on identical run
  seeds, paired mean floor delta, exact McNemar p-value, deterministic
  bootstrap 95% interval.
- `analysis.paired.puct64_vs_policy`: separate baseline comparison.
- `puct64_runs`: one per-seed record for the 64-simulation arm.

**Interpretation:** The seed is shared across arms, but once selected
actions diverge, the subsequent combat states may differ. Thus this tests
the **run-level effectiveness** of increasing the search budget, not a
state-matched root-action quality comparison. The PUCT implementation uses
privileged exact-state forks. Also, R79 was selected on these 64 seeds;
a separate fresh-seed confirmation is needed before a strong claim.
