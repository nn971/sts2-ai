# Contributing

## Scope discipline

Before adding code, decide which repository owns the behavior.

If the code answers **what mechanically happens in Slay the Spire 2**, it belongs in `sts2-emulator`.

If the code answers **which mechanically valid action is preferable, how to search, learn, cache, evaluate, or train**, it belongs here.

Examples:

| Change | Repository |
|---|---|
| Correct poison damage ordering | `sts2-emulator` |
| Implement shop RNG | `sts2-emulator` |
| Add batch state stepping | `sts2-emulator` |
| Add MCTS | `sts2-ai` |
| Train a value model | `sts2-ai` |
| Store search targets | `sts2-ai` |
| Define a fair observation encoding | `sts2-ai` (using emulator projection APIs) |

## Reproducibility

Any committed benchmark or strategic result should identify:

- parent Git commit;
- emulator Git commit;
- target game build;
- experiment configuration;
- seeds;
- model/checkpoint version;
- information policy;
- relevant dataset/strategy-store snapshot.

Use the utilities in `sts2_ai.evaluation.manifest` rather than hand-writing these fields when possible.

## Tests

Run:

```fish
./scripts/test.fish
```

The default unit suite uses mock emulator backends and therefore does not require the submodule. Emulator integration tests will live behind an explicit integration marker once stable bindings exist.

## Formatting / static checks

```fish
./scripts/lint.fish
```
