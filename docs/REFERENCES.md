# References and design influences

This starter repository intentionally keeps the research architecture general. Several choices have established precedents:

- Git submodules: the parent repository records a `gitlink` to an exact commit in the emulator repository, with `.gitmodules` recording its location. This is why the emulator revision can form part of experiment identity. See the official Git documentation: <https://git-scm.com/docs/gitsubmodules> and <https://git-scm.com/docs/git-submodule>.
- Expert Iteration: Thomas Anthony, Zheng Tian, and David Barber, *Thinking Fast and Slow with Deep Learning and Tree Search*, NeurIPS 2017. The search/generalization feedback loop described in `SEARCH_AND_LEARNING.md` follows this broad idea rather than committing to its exact algorithm.
- `sts2-emulator`: the sibling repository's purpose, fidelity contract, RNG design, parity methodology, and performance requirements are the mechanical foundation for this project.

References are design context rather than claims that these exact algorithms will be optimal for Slay the Spire 2.
