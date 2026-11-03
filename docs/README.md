# Weave documentation

## Start here

- [Install and configure](install.md): native setup, GPU environments, paths, updates and startup services.
- [Docker](docker.md): persistent storage, CPU/NVIDIA images and container lifecycle.
- [Using Weave](user-guide.md): graphs, variables, tokenizers, shared weights and reference models.

## Train and test

- [Training operations](training.md): stages, devices, monitoring, validation, resume and snapshots.
- [Writing train.py](training-guide.md): loaders, caching, augmentations and trainer hooks.
- [Checkpoint chat and decoders](checkpoint-chat-decoders.md): catalogs, constrained decoding and regression cases.
- [MCP](mcp.md): control the workbench and search its documentation.

## Understand the implementation

- [Repository map](structure.md): source folders, runtime artifacts and compatibility files.
- [Reference-model fidelity](reference-models.md): what each reference reproduces and what it omits.
- [Design advice](design-advice.md): shape errors, advisory heuristics and their sources.
- [Contributing](../CONTRIBUTING.md): development setup and checks.

## Research and reviews

These record decisions and observations at a particular time; use the guides above for current instructions.

- Research: [graph layout](research/layout.md), [UI component libraries](research/ui-components.md).
- Reviews: [project workflow](reviews/project-workflow.md), [UI/UX](reviews/ux-2026-10-27.md), [training performance](reviews/performance-2026-11-01.md), [installation](reviews/installation-2026-11-03.md), [code and dead-file review](reviews/code-2026-11-03.md).

[License](../LICENSE) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
