# Training in Weave

Open a project and choose **Training**. Stages and run records are separate from the editable graph: starting a run freezes the graph, variables, generated PyTorch, tokenizer, script, and configuration. Existing projects are not migrated or trained automatically.

## Enable local execution

```bash
WEAVE_TRAINING_ENABLED=1 \
WEAVE_TRAINING_PYTHON=/absolute/path/to/cuda-environment/bin/python \
./scripts/start.sh
```

The web application remains on `0.0.0.0:8173`. Unlock Training with the operator code in `data/training/operator.token`, or set `WEAVE_TRAINING_TOKEN` on the server. The browser receives an HttpOnly, same-site session cookie. Training APIs reject cross-origin browser requests. Keep this code private: editable scripts intentionally execute with the server user's permissions; this is a trusted local workbench, not a sandbox for untrusted Python.

Install the application's requirements in its web environment. Install PyTorch, Transformers, sentencepiece, jsonschema and optionally pyarrow in the selected training interpreter. The interpreter can differ from the API server's interpreter. CUDA wheels must support the actual GPU architectures. This host's V100 and 2080 Ti were tested with the isolated `data/training-env` environment, PyTorch 2.6.0 + CUDA 12.6. GPU numbering follows `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

A run supervisor is independent of browser connections and API request lifetime. Browser closure and API restarts do not stop a launched job. **Stop safely** requests a checkpoint at an optimizer boundary. **Force stop** terminates the owned process group and cannot save unfinished work. Terminal mode prepares a package and displays a command pointing at its shared run directory; it does not launch a remote worker. Downloaded packages can run independently, but live monitoring requires their reports to be written into the host's run directory.

## Stages and editable recipes

Create, duplicate, rename, reorder, and save stages. Browser drafts protect unsaved script edits. **New stage from checkpoint weights** resets the optimizer/scheduler; **Resume** restores their state, scaler, counters and RNG. The template checks model/support-code/tokenizer identity, script identity, dataset hashes, prepared records, batch schedule and relevant settings before resuming. Increasing the total target step count is allowed. If these contracts change, start a new stage from weights instead.

Recipes:

- **Text pretraining:** causal next-token cross entropy. Long text records are split into overlapping context windows, preserving the boundary target. Prepared examples are sampled deterministically with replacement using seed + optimizer step. Consumed-record traces document what was actually sampled; a dataset manifest alone is not proof every record was consumed.
- **Chat/tool-call SFT:** assistant-only next-token loss over `messages`, with optional `tools`. A generation-aware chat template must supply assistant masks; missing masks are an error. Alternatively provide explicit `input_ids` and `labels`, with `-100` on ignored targets. SFT records are truncated to the configured context, so use the preview to check supervision survives.
- **Custom Python:** an editable reporting example, not a claim of real training. Supply your own model/data/loss/optimizer and checkpoint logic. All settings and a `custom` JSON object are available in `config.json`.

Built-in text recipes require a single int64 `[B,T]` token input and a single floating-point `[B,T,V]` vocabulary-logit output. Noncausal attention is rejected for autoregressive recipes to avoid future-target leakage. Encoder-decoder, diffusion, arbitrary objectives and other graph contracts use custom scripts.

Data can be uploaded (512 MB limit) or referenced by a host path. TXT, JSONL, JSON arrays and Parquet are supported. JSON arrays over 64 MB must be converted to JSONL/Parquet; preparation streams those formats into disk-backed records with a bounded token cache. It retains an in-memory offset index. Huge individual documents still require memory to tokenize that document. Right-padding is ignored in the loss; built-in autoregressive recipes use causal attention. No arbitrary padding-mask input is assumed. The preview highlights supervised target tokens; the first token is never a next-token target.

## GPU strategies

- **Single device:** CPU or one GPU.
- **DDP:** one process per GPU, full model/optimizer replicas, indexed data sharding, token-weighted gradient accumulation and aggregated losses. Checkpoints include per-rank RNG. DDP does not pool VRAM or guarantee heterogeneous-GPU speedups.
- **Split model:** a supported single chain of top-level modules, assigned individually to devices. Containers keep residual paths and loop internals on one device. Autograd-preserving transfers happen before module calls. Shared/tied parameters must stay on one GPU. Functional branches and custom root blocks need containers or a custom placement script. This initial implementation uses FP32 and sequential model parallelism, not an overlapping pipeline schedule.
- **Training + evaluation:** one GPU trains; the second loads completed immutable checkpoints. The queue retains the newest pending checkpoint rather than accumulating unbounded evaluation jobs. Evaluation never reads actively mutating weights.

The built-in recipes support FP32, CUDA FP16 with a gradient scaler, and supported BF16 hardware. Native BF16 is checked on selected GPUs. Split placement currently requires FP32. Capability/runtime errors appear in Logs. Existing loop/shared-weight semantics are retained.

## Reporting API

Exports include `weave_training.py`, a standard-library-only helper:

```python
from weave_training import Run

with Run() as run:
    run.capabilities(metrics=True, samples=True, checkpoint=True, commands=True)
    # Your forward / loss / backward / optimizer step belongs here.
    run.log({"train/loss": 1.25, "train/learning_rate": 0.0003}, step=100, tokens=64000)
    run.progress(completed=100, total=1000, unit="optimizer steps")
    run.sample(prompt="Hello", output="Hello there", checkpoint="checkpoints/step-100.pt",
               metadata={"decoding": "greedy", "seed": 42})
    # Save a complete checkpoint atomically before publishing it:
    # run.checkpoint(path_inside_run_directory, metadata={...})
    commands = run.poll_commands()
```

Use `atomic_checkpoint_path` with `torch.save` to publish a complete file. Checkpoint metadata for a snapshot request must include `snapshot_requests: [command['id'], ...]`; the template demonstrates this. Poll commands only at safe training boundaries. Reporting is rank-zero-only by default. Unknown scalar metric names require no UI changes. Nonfinite scalar metrics generate warnings and are omitted from charts.

Static script checks are advisory: a method call in source does not prove it executes. The dashboard separately reports metrics/samples/capabilities that have actually arrived. Missing instrumentation never fabricates loss/progress or automatically fails a custom job. Metrics and script events are durably appended locally and incrementally indexed in SQLite. SSE reconnects with event IDs; polling also recovers status and artifacts. The dashboard shows the most recent 10,000 events; full history remains on disk and in snapshots.

Charts support step/token/time axes, raw or smoothed values and run comparison. Hardware records use NVML (nvidia-smi fallback) and psutil. GPU utilization and used VRAM are **device-wide**, including other jobs; launcher RSS is labeled separately. Per-block gradient norms and activation RMS are optional, sparse summaries, disabled by default because they add synchronization overhead. No full tensor tracing is enabled automatically.

Evaluation uses fixed prompts and greedy decoding, saving prompt/output/checkpoint/settings. Structured tool-call prompts can additionally specify `tools`, `expected_tool` and `expected_arguments`; JSON parse success, schema validity and labeled tool/argument correctness are reported when applicable. Generated tools are never executed. These are limited mechanical checks, not evidence that arbitrary real-world tool use is safe or successful.

## Snapshots and lineage

**Create snapshot** requests a completed checkpoint and writes an archive under the run's `snapshots/` directory before offering a download. A finished custom run with no checkpoint can produce a clearly labeled run-record archive, not a resumable weight backup.

Snapshots preserve the selected weights/state, graph, support code, tokenizer assets, edited script, configuration, environment, logs, metrics, generated examples, source/preparation manifests, consumed-record hashes, and ancestor-stage provenance. Later stages copy earlier stages' scripts/configuration/events and consumed-record traces at the selected parent checkpoint; they do not silently rely solely on a mutable source project. Historical console logs can cover the source run beyond the selected checkpoint; the selected-checkpoint record and consumed traces identify the actual weight lineage.

Default snapshots exclude source and prepared dataset records, retaining their hashes. **Include source dataset files** copies currently available, hash-matching files from the current stage and its ancestors, with a size estimate. Missing/changed files are explicitly listed. A hash is an identity, not a backup; streaming/custom loaders must describe their own coverage. Snapshot imports verify each file's size/SHA-256, reject unsafe or duplicate paths, and do not execute imported scripts. Bundled current-stage data paths are reconnected. Select a checkpoint and use **Create stage from this run** to restore its editable script/settings.

Checkpoints are written atomically and published only after completion. Snapshot archives stream large files without loading a whole model into server RAM. Exact continuation is tested for deterministic fixtures; changed software, nondeterministic GPU kernels and custom data loaders can prevent bitwise reproduction. Custom scripts own their checkpoint semantics. There is no automatic artifact deletion; disk usage is shown.

## Validation

```bash
npm run build
.venv/bin/python -m pytest tests/test_training.py tests/test_training_api.py -q
WEAVE_CUDA_PYTHON=/absolute/path/to/cuda/python \
  .venv/bin/python -m pytest tests/test_training_gpu.py -q
npx playwright test tests/browser/training.spec.ts
```

Real two-GPU acceptance compares single-device, DDP and split-model weights; checks incompatible weight-sharing placement; and runs a separate evaluation GPU. Browser tests cover stage editing, actual training, token-mask preview, charts, snapshots, reloads, black theme, accessibility and phone layouts.
