# Development conventions

## Shell examples

Project documentation uses Fish syntax for interactive shell commands.

## Package style

- Type annotate public functions.
- Prefer immutable/frozen dataclasses for portable records.
- Keep emulator semantics behind protocols/adapters.
- Keep ML-framework-specific code in adapters so the core research schema remains lightweight.
- Make provenance fields explicit rather than relying on global process state.

## Tests

Unit tests should use mock backends where mechanics are irrelevant. Integration tests should explicitly opt into the real emulator binding.

## Configuration

Checked-in configs are declarative JSON/TOML/YAML-like artifacts. The starter uses TOML to avoid an extra YAML dependency.

## Large artifacts

Do not commit checkpoints or large generated corpora. Store manifests and checksums instead.
