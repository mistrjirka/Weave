# Repository map

The root contains the project entry points: README, contributing instructions, licenses, build/dependency manifests, Docker/Compose configuration and tool configuration. Detailed documentation belongs under `docs/`.

| Location | Purpose |
| --- | --- |
| `frontend/src/` | Graph editor, typed API schemas, graph transformations, shared UI and training screens |
| `frontend/src/chat/` | Checkpoint-chat panel, transcript, regression runner and their schemas |
| `frontend/public/` | Browser assets and font licenses |
| `backend/` | FastAPI app, shape validation, code generation, model operations and installation |
| `backend/training/` | Training orchestration, workspace/snapshot handling, reporting, caches, evaluation and chat workers |
| `backend/training/chat_presets/` | Bundled tool-catalog data |
| `docs/` | Current user/developer guides, with `docs/README.md` as the index |
| `docs/research/` | Design research and technical tradeoffs |
| `docs/reviews/` | Dated audits, measurements and review findings |
| `docs/images/` | Public documentation screenshots |
| `examples/` | Complete optional training recipes shipped in training packages |
| `references/` | Pinned upstream architecture sources/configurations and licenses |
| `scripts/` | Native install/configure/start wrappers and example service |
| `tests/` | Python and TypeScript checks; browser tests in `tests/browser/`, fixtures in `tests/fixtures/` |

## Files that look unused but are loaded indirectly

Generated exports and worker processes are additional entry points. An ordinary import scan is insufficient evidence for deleting them.

- `backend/runtime.py`, `export_helpers.py`, `sparse_runtime.py` and several training modules are copied into exports. They are not necessarily imported by the web process.
- `backend/training/chat_worker.py`, `evaluation_worker.py`, `tuning_worker.py` and `supervisor.py` run in separate processes.
- `hook_template.py` is the current starter recipe. `template.py` and the legacy paths in `text_training.py` still support saved stages, migration and documented tests. They are not duplicate starters to remove without a compatibility migration.
- `references/*_architecture.py`, reference model sources and `.npz` fixtures support fidelity checks. Reference JSON and licenses are also used during export.
- Cache-key arguments such as `modified_ns` and `minute` deliberately change the memoization key even when a function body does not read them.
- `.d.ts` declarations and files mentioned in font CSS are consumed by tooling or the browser.
- `requirements.txt` is a compatibility export from `uv.lock`; `training-requirements.txt` describes a separate hardware-dependent environment.

`backend/documentation.py` maps stable MCP guide IDs to their installed paths. Old calls such as `read_weave_doc(guide="TRAINING_GUIDE.md")` remain valid; callers may also use `docs/training-guide.md`. Exported stages retain the standalone filename `weave_support/TRAINING_GUIDE.md`.

## Generated and private storage

These directories are ignored by Git and are not source folders:

- `node_modules/`, `.venv/`, `.venv-training/`: installed dependencies.
- `dist/`: built browser application.
- `data/`: local databases, tokenizer assets, run state and test artifacts. Only `data/.gitkeep` is committed.
- `weave-state/`, `weave-projects/`, `weave-settings/`: default Docker bind mounts.
- The configured external projects directory, normally `~/shared/weave-projects`: editable user projects.

Do not clean these as part of a source-code audit. They can contain valuable user data and checkpoints.

## Further decomposition

`main.tsx` and `TrainingWorkspace.tsx` remain large orchestration components. Their size is a maintenance concern, not evidence that their children are dead. Extract coherent features with their own lifecycle and tests when changing those features. Avoid moving all source files at once or adding barrel files that hide dependency ownership.
