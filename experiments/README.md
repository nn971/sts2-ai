# Experiments

Use one directory per durable experiment family. Keep small configs, notes, and result summaries in Git; put large logs/checkpoints under ignored `runs/` or external storage.

A useful experiment directory eventually looks like:

```text
experiments/2026-xx-strategic-search-baseline/
├── README.md
├── config.toml
├── manifest.json
└── summary.md
```

Every result should identify the exact parent and emulator revisions.
