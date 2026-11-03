# Developing Weave

Use Linux or WSL2. Start with the [installation guide](docs/install.md), then install development dependencies:

```bash
uv sync --locked --group dev --inexact
npm ci
npm run build
```

Computational tests require a compatible PyTorch build in the test interpreter. A production install keeps training in a separate environment; installing web development dependencies does not add torch.

Run the API with `bash scripts/start.sh`. In another terminal, `npm run dev` serves Vite on port 5173 and proxies `/api` to port 8173.

## Checks

```bash
npm run typecheck
.venv/bin/python -m pytest tests/test_installation.py tests/test_mcp_bridge.py -q
# Once PyTorch is installed in this interpreter:
.venv/bin/python -m pytest tests -q
# Browser checks:
npx playwright install chromium
npm run build
npx playwright test
```

Some tests require optional hardware, third-party reference packages or fixtures; check skip messages. Browser tests write screenshots under the ignored `data/` directory. Several tests exercise process launches and project storage: use an isolated checkout/data directory rather than a server holding active training runs.

TypeScript tests such as `npm run test:routing` use Node's native TypeScript support and require Node 22.18+ (the installer/UI build itself supports 20.19+).

## Source map

See the [repository map](docs/structure.md) for source boundaries, generated storage and indirectly loaded modules.

- `frontend/src/`: React, TypeScript, React Flow, CodeMirror and graph operations.
- `backend/core.py`: contracts and tensor-shape inference.
- `backend/compiler.py`, `backend/codegen.py`: PyTorch code generation.
- `backend/runtime.py`: specialized exported PyTorch operations.
- `backend/training/`: recipes, orchestration, reporting, caches and checkpoint chat.
- `backend/installation.py`, `scripts/`: native installation and server startup.

The backend and frontend share the same versioned graph document. Avoid changes that make a visual block's behavior differ from its exported computation. Changes to graph operations should check shapes, generated code and parameter sharing as appropriate.

## API and guides

The running server exposes interactive API documentation at `/docs` and its schema at `/openapi.json`. Core endpoints include projects, templates, validation, code generation, export and tokenizer inspection/upload. [MCP.md](docs/mcp.md) describes tool access to the API and guides.

[USER_GUIDE.md](docs/user-guide.md) contains the graph editor's detailed behavior and reference-model limits. Update the relevant guide when changing user-visible behavior; keep README focused on what the app does and how to start it.

README screenshots use an isolated installation and example models, never personal project data. The overview screenshot uses manually arranged nodes; node positions do not change the computation.
