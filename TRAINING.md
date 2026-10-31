# Training in Weave

Open a project and choose **Training**. Stages and run records are separate from the editable graph: starting a run freezes the graph, variables, generated PyTorch, tokenizer, script, and configuration. Project graphs are not changed and training never starts automatically.

## Enable local execution

```bash
WEAVE_TRAINING_ENABLED=1 \
WEAVE_TRAINING_PYTHON=/absolute/path/to/cuda-environment/bin/python \
./scripts/start.sh
```

The web application remains on `0.0.0.0:8173`. Training opens without a token by default. Operators can opt into authentication by setting `WEAVE_TRAINING_TOKEN`; existing generated `operator.token` files are no longer used as an implicit access gate. When optional authentication is enabled, the browser uses an HttpOnly, same-site session cookie. Training APIs reject cross-origin browser requests. Editable scripts run with the server user's permissions; this is a trusted local workbench.

## Editable Python files and SSH / VS Code

Each stage has a stable workspace on the **Weave server**:

```text
~/shared/weave-projects/<project-name>/project.json
~/shared/weave-projects/<project-name>/stages/<stage-id>/
    train.py              # Your typed configuration and training hooks
    my_helpers.py         # Optional files you create
    runs/                 # Standalone checkpoints, reports and prepared data
    weave_support/        # Managed by Weave; inspect rather than edit
        model.py          # Generated PyTorch network
        trainer.py        # Optional hook-based Trainer
        data_stream.py    # Optional resumable token stream
        launch.json       # Generated launch/parent-checkpoint context
        text_training.py  # Compatibility for older recipes
        weave_training.py # Portable reporting helper
        requirements.txt
        tokenizers/
```

Choose the base folder during installation with `bash scripts/configure.sh --projects-dir ~/shared/weave-projects`, or set `WEAVE_PROJECTS_DIR`. The default is `~/shared/weave-projects`; it is independent of `WEAVE_DATA` and the selected Python environment. Each folder uses a lowercase, hyphenated form of the project name; projects with the same name receive a short ID suffix. Renaming a project also renames its folder. Restart Weave after changing the base folder. Existing folders migrate to the new layout without overwriting conflicting files. Exact unedited long starter scripts become the short recipe. Custom scripts keep their logic and comments; only imports of moved support modules are qualified. Original scripts are preserved under `weave_support/migrations/train.original.py`. Historical run exports are not changed.

The Training tab shows the absolute paths and provides Copy path buttons. Open those paths in an SSH-connected VS Code window or in a terminal on the server. They are not browser downloads or paths on your laptop. `weave_support/model.py`, tokenizer assets, the standard trainer and reporting helper are generated in the managed support folder when you save. Keep your own helper code alongside train.py, outside that folder. New scripts use normal package imports such as `from weave_support.model import Model` and `from weave_support import Run`. There are no sys.path changes or pip-installed Weave packages.

The default view is the Python script, with CodeMirror syntax highlighting, line numbers, folding, bracket matching, undo, search, and Ctrl/Cmd+S to **Save file**. New recipes configure training directly in Python with a typed dataclass; **Launch settings** choose the interpreter, launch location, starting checkpoint and optional validation override. Python owns devices, precision, data, batch schedule and optimization. Older recipes retain **Preset settings** and `config.json`. The reporting-hook menu is available for manual loops. Fresh templates create a new stage instead of overwriting your script.

The selected interpreter's actual Python, PyTorch and CUDA versions are detected and displayed. The executable path is a path on the server. Use these versions when choosing syntax and APIs; scripts do not run in the browser's JavaScript environment.

Browser saves write the real files atomically. Clean browser views poll for SSH edits and reload them. If a server file changes while the browser has an unsaved draft, the UI reports a conflict and stale saves return HTTP 409. Use **Reload server files**, or duplicate the stage to preserve your draft. Scripts remain separately recoverable in browser draft storage. Legacy inline-script stages migrate to files on first load without replacing an existing server file.

Starting training freezes a separate run directory. Subsequent SSH/browser edits to stage files do not change active or historical runs. Extra Python/config helper files in the stage workspace are copied into the frozen run, while graph-owned files are freshly exported from the graph.


Install the application's requirements in its web environment. Install PyTorch, Transformers, sentencepiece, jsonschema and optionally pyarrow in the selected training interpreter. The interpreter can differ from the API server's interpreter. CUDA wheels must support the actual GPU architectures. This host's V100 and 2080 Ti were tested with the isolated `data/training-env` environment, PyTorch 2.6.0 + CUDA 12.6. GPU numbering follows `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

A run supervisor is independent of browser connections and API request lifetime. Browser closure and API restarts do not stop a launched job. **Stop safely** requests a checkpoint at an optimizer boundary. **Force stop** terminates the owned process group and cannot save unfinished work. Terminal mode prepares a package and displays a command pointing at its shared run directory; it does not launch a remote worker. Downloaded packages report independently over HTTP using their included reporting.json; terminal jobs using the original shared server run directory continue to write directly to it.

## Stages and editable recipes

Create, duplicate, rename, reorder, and save stages. New templates contain a data-loader placeholder and editable functions for AdamW, cosine learning-rate decay and a next-token loss example. Fill in your own loader before starting training; no dataset path, row encoder or sequence-length policy is prefilled. Replace any hook or use a complete custom PyTorch loop. There is no user-facing `config.json` in new recipes; generated `weave_support/launch.json` records launch context and checkpoint lineage, not overrides of Python training settings.

```python
from weave_support.trainer import Trainer, TrainingConfig, StepResult

config = TrainingConfig(devices=[1], precision="fp16", max_epochs=3, max_optimizer_steps=None)
# Define your model/tokenizer, get_dataloaders, configure_optimizer,
# configure_scheduler and training_step in ordinary Python.
Trainer(model=Model, config=config, tokenizer=load_tokenizer,
        dataloaders_fn=get_dataloaders, optimizer_fn=configure_optimizer,
        scheduler_fn=configure_scheduler, training_step_fn=training_step).train()
```

The loss hook accepts `(model, batch, device)` and returns a scalar mean loss, `(loss, metrics)`, or `StepResult(loss, metrics, weight=...)`. A tuple averages microbatches equally. The default recipe passes the supervised-token count as `weight`, giving correct token weighting across uneven masks, accumulation windows and DDP ranks. Forward/backward, AMP, clipping and stepping remain ordinary PyTorch. Custom auxiliary losses, including MoE balancing loss when needed, belong in this hook. Optional `evaluation_step_fn` and `samples_fn` replace validation and generation. Optimizer/scheduler hooks receive `(model, config)` and `(optimizer, config)` respectively. Configuration is a typed Python dataclass with runtime validation, not a complete type checker for arbitrary custom code.

- **Text pretraining:** implement `get_dataloaders` with your own dataset, tokenizer, chunk lengths and batching. Supply already-shifted next-token labels. The bundled TokenStream is an optional utility for packing/token-source attribution and stateful cursors; it is not inserted into new scripts.
- **Chat/tool-call SFT:** your loader supplies assistant-only labels, usually with `-100` for ignored targets. Select conversation boundaries and attention/padding masks in your own data/model code. The starter only provides the mean cross-entropy loss example.
- **Custom Python:** use Trainer with arbitrary model/data/loss hooks, or replace it entirely with your own loop and the portable Run helper. Select the Custom Python recipe for graph contracts other than causal next-token prediction.

**New stage from checkpoint weights** resets optimizer/scheduler state. **Resume** restores them plus scaler, counters, per-rank RNG and loader state. Exact resume requires unchanged training/model code and data fingerprints, the same process count and compatible configuration. A loader must implement `state_dict/load_state_dict`, or its dataset must implement them with `num_workers=0`. The optional TokenStream utility implements this protocol. Worker prefetch queues cannot be reconstructed from a dataset cursor alone; use a stateful loader when adding workers. An opaque loader produces a warning and weights-only checkpoints; it never silently restarts data while claiming exact resume. Legacy recipes retain their existing indexed sampling/resume behavior. To extend a run while preserving script identity, derive `max_epochs` or `max_optimizer_steps` from a runtime value such as an environment variable; editing arbitrary Python requires starting from weights.

Built-in text recipes require a single int64 `[B,T]` token input and a single floating-point `[B,T,V]` vocabulary-logit output. Noncausal attention is rejected for autoregressive recipes to avoid future-target leakage. Encoder-decoder, diffusion, arbitrary objectives and other graph contracts use custom scripts.

Define dataset paths, tokenization and sequence lengths in your loader. Legacy recipes additionally support uploads (512 MB limit) and token previews. TXT, JSONL, JSON arrays and Parquet are supported. JSON arrays over 64 MB must be converted to JSONL/Parquet. The optional TokenStream utility streams and packs data; legacy preparation uses disk-backed records with a bounded cache and an in-memory offset index. Huge individual documents still require memory to tokenize that document. Right-padding is ignored in the loss; built-in autoregressive recipes use causal attention. No arbitrary padding-mask input is assumed. The preview highlights supervised target tokens; the first token is never a next-token target.

## Epochs, optimizer updates and learning-rate schedule

Use `max_epochs=3, max_optimizer_steps=None` for three complete loader passes, or `max_epochs=None, max_optimizer_steps=2000` for a fixed number of weight updates. If both limits are set, the first reached stops training. Every epoch flushes its final partial accumulation group; batches are never pulled from the next epoch to fill it. Counters distinguish completed epochs, microbatches per rank, optimizer updates and global input token slots (including padding).

The starter's `learning_rate` is the starting rate and `min_learning_rate` is the cosine floor. Trainer sets `config.total_optimizer_steps` after constructing the loader: for a sized loader it uses `ceil(len(loader) / accumulation) * epochs`, capped by an explicit update limit. The cosine hook steps after successful optimizer updates. For an unsized stream, an epoch still ends at StopIteration, but cosine decay needs a loader length or an explicit optimizer-update budget. The starter raises a clear error rather than consuming the corpus just to count it. You can provide another scheduler or no scheduler for an epoch-only unsized stream. Repeating epochs requires a finite, repeatable loader; use an explicit optimizer-update limit for an endless stream.

Sequence lengths belong to the loader; there is no shared `sequence_length` setting in the hook trainer. Optional `generation_context_length` limits only sample generation, and defaults to no extra cropping. Model positional/context limits still apply. Existing frozen exports retain their older helper code; new hook recipes use `max_optimizer_steps` rather than `max_steps`.

## GPU strategies

- **Single device:** CPU or one GPU.
- **DDP:** one process per GPU, full model/optimizer replicas, explicit rank/worker data sharding, token-weighted gradient accumulation and aggregated losses. Checkpoints include per-rank RNG. DDP does not pool VRAM or guarantee heterogeneous-GPU speedups.
- **Split model:** a supported single chain of top-level modules, assigned individually to devices. Containers keep residual paths and loop internals on one device. Autograd-preserving transfers happen before module calls. Shared/tied parameters must stay on one GPU. Functional branches and custom root blocks need containers or a custom placement script. This initial implementation uses FP32 and sequential model parallelism, not an overlapping pipeline schedule.
- **Asynchronous validation (hook Trainer):** independent of single-device, DDP or split training. Set `async_evaluation=True, evaluation_device="cpu"` (or a GPU index such as `1`) in Python. In **Launch settings**, select **Asynchronous · continue training** and a **Validation device** to override the script for that launch. **Use train.py settings** preserves the script's choices. Inline validation uses the training device. CPU validation uses FP32 even when training uses FP16. GPU validation inherits training precision and checks the selected device's capabilities.
- **Training + evaluation (legacy strategy):** retains compatibility with older recipes that use one training GPU and a separate evaluation GPU.

Background evaluation runs the user's validation/loss/sample hooks in a fresh Python process on an immutable weights snapshot. One job runs at a time and only the latest pending checkpoint is retained, so intermediate evaluations may be skipped when validation is slower than training. Results retain the checkpoint's optimizer-update number and a separate evaluation channel; remote evaluation cannot finish the parent training run. Worker failures become warnings and appear in evaluation logs. After the last update, the run waits for final validation to finish before marking the run completed.

Validation still briefly uses the training process to capture/save stable weights. Evaluations between regular checkpoints save weights only, without serializing the optimizer. Choose a separate GPU to avoid competing for training VRAM/compute; CPU validation still shares host CPU, RAM and disk bandwidth. The worker constructs a separate model and invokes `get_dataloaders`; hooks can check `config.evaluation_only` to return `(None, val_loader)` and avoid building the training loader. Optimizers and schedulers are not constructed in the evaluator. Model construction/data hooks must be repeatable. Separate-process execution avoids sharing a live CUDA model across threads, following [PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).

The built-in recipes support FP32, CUDA FP16 with a gradient scaler, and supported BF16 hardware. Native BF16 is checked on selected GPUs. Split placement currently requires FP32. Capability/runtime errors appear in Logs. Existing loop/shared-weight semantics are retained.

## Independent training computers

Choose **Terminal · prepare package and monitor**, then download **Training package**. Each package includes user-editable train.py, reporting.json and your own helper files; generated code, tokenizers and dependencies are separated under weave_support/. Install dependencies from weave_support/requirements.txt. The visible reporting address defaults to `http://localhost:8173`. This works on the same computer or through an SSH tunnel. For another computer, set the server’s reachable URL in Training, save it, and download the package again. Localhost always refers to the training computer, not the browser computer.

Extract it, edit train.py for that computer’s dataset paths and devices, install its Python dependencies, then run:

```bash
python train.py
```

Legacy packages keep their original `config.json` and `--config config.json` command.

Standalone recipes print their generated `runs/run-<id>/` output folder. UI-launched jobs keep using their frozen server run directory. The helper detects reporting.json beside the source script/configuration, or use `WEAVE_REPORT_CONFIG=/path/to/reporting.json`. A scoped reporting key is generated automatically and included in the file; it grants event reporting and cooperative command polling for that run only. It does not grant launch/project access. Do not publish that connection file. No operator access token is required by default.

HTTP delivery and remote command polling run in a background thread with a two-second request timeout; failed or stalled requests do not execute on the optimizer thread. A real stalled-HTTP-server regression test verifies that optimizer updates complete before the response is released. Local JSONL logging remains synchronous and has small serialization/disk costs; GPU-to-CPU metric conversion and checkpoints are also not free. Clean shutdown waits up to three seconds for the sender, with unsent events retained for later synchronization. Metrics, progress, generated examples, explicit console events, warnings, checkpoint metadata and optional NVIDIA device readings appear in Training. Events remain durable in the training computer’s run directory and are retried; the server deduplicates event IDs. If the script finishes during an outage, resend the local outbox after the server returns:

```bash
python weave_support/weave_training.py --sync weave-run --connection reporting.json
# Built-in standalone templates print their output folder at startup:
python weave_support/weave_training.py --sync runs/run-<id> --connection reporting.json
```

The UI shows the last report time; it does not claim that an interrupted connection means the process stopped. Remote checkpoint bytes and datasets are **not uploaded**. Server snapshots preserve reports and the originally packaged code, without claiming resumable remote weights. Remote provenance events record code fingerprints and the dataset metadata requested by the script; modified remote files must be backed up on that computer. Only explicit `run.console(text)` output is forwarded; arbitrary stdout is not intercepted. Commands arrive on a successful report exchange and are acted on only when the script polls and implements them. Force-stop remains a terminal operation.

## Reporting API

Exports include a local `weave_support/` package with the portable reporting helper. There is **no pip-installed Weave package**, no `requests` dependency, and no HTTP code to put in your training loop. The helper uses the Python standard library, a background sender and a durable local event file. `reporting.json` provides the connection automatically. Built-in training templates already contain their metrics, samples, checkpoint and command integration.

For a custom loop, the short integration is:

```python
from weave_support import Run

# Construct your own model, optimizer and data loader as usual.
with Run(optimizer=optimizer, total_steps=total_steps) as run:
    for batch in loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)  # Your scalar training objective.
        loss.backward()
        optimizer.step()
        # scheduler.step() goes here if used.
        run.log_loss(loss)
```

Step counts, each optimizer parameter group's applied learning rate, and progress are collected automatically through PyTorch's public `register_step_post_hook`. Hooks are removed on context exit. A scheduler's next learning rate is not confused with the rate applied to the update. Gradient accumulation counts optimizer updates, not backward calls. `total_steps` is a reporting target; your loop still controls when training stops. Use `start_step=checkpoint_step` when resuming, and `log_every=10` to reduce reporting frequency (the first and final updates are still reported). An attached optimizer that updates without any loss reporting produces a visible `missing-loss` warning; this is advisory and never aborts training.

`run.log_loss(loss, tokens=total_tokens, metrics={"train/gradient_norm": norm})` can include additional values. Loss remains the one explicit per-update reporting call because only the script knows the correct objective, masks, weighting and accumulation. For DDP, pass the correctly reduced objective or keep the built-in template's explicit aggregate reporting. We do not patch Tensor.backward, optimizer.step, torch.save, or inspect arbitrary local variables. See [PyTorch optimizer hook documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.register_step_post_hook.html).

GPU sampling (when available), reporting retries, lifecycle and error events are handled by the helper/supervisor. Custom generation, checkpoint contents and safe command handling depend on the script's algorithm; the built-in recipes already implement these. The editor's **Insert reporting hook** menu provides snippets, and static warnings plus the received-event dashboard help catch missing instrumentation.

Existing scripts using `run.log(...)` or `run.watch_optimizer(optimizer, every=10)` continue to work.

Explicit reporting example:

```python
from weave_support import Run

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

Charts support step/token/time axes, raw or smoothed values and run comparison. Hardware records use NVML (nvidia-smi fallback) and psutil. GPU utilization and used VRAM are **device-wide**, including other jobs; launcher RSS is labeled separately. Legacy recipes also offer optional per-block gradient and activation summaries; hook-based recipes can add ordinary PyTorch hooks for their own diagnostics. No full tensor tracing is enabled automatically.

Evaluation uses fixed prompts and greedy decoding, saving prompt/output/checkpoint/settings. Structured tool-call prompts can additionally specify `tools`, `expected_tool` and `expected_arguments`; JSON parse success, schema validity and labeled tool/argument correctness are reported when applicable. Generated tools are never executed. These are limited mechanical checks, not evidence that arbitrary real-world tool use is safe or successful.

## Data history without copying the corpus

The default `data_fingerprint="stat"` records absolute source paths, byte sizes and modification times, without reading every file to hash it. `data_fingerprint="sha256"` hashes each declared file once when starting a run; this adds disk I/O proportional to corpus size. Declared files are not a claim that every file was consumed.

The optional TokenStream emits `_weave_source` metadata with the original file, row and token span for every packed example. Trainer records consumed microbatches per optimizer update in `consumed-rank-*.jsonl`, including skipped FP16 updates. It does not save full batch tensors by default. Custom datasets can expose `source_files` (or `files`) and add `_weave_source` to their batch dictionaries. Without these, Trainer warns that exact attribution is unavailable. `trace_batches="sha256"` additionally hashes actual CPU batch contents; GPU batches require a copy to CPU and add synchronization overhead. `trace_batches="none"` disables consumed-batch tracing.

A local Weave snapshot preserves the frozen loader/hook code, resolved Python configuration, file manifest, consumed-batch traces, training events and selected checkpoint. A loader cursor is saved through its state protocol; arbitrary loader objects are not pickled. This keeps custom lambdas, open files and worker processes out of snapshot serialization. The full corpus is copied only when explicitly selecting Include data. Stat fingerprints can detect normal file modifications but do not provide the same content assurance as SHA256. Imported custom loaders may need their data paths changed to bundled files; Weave does not rewrite arbitrary Python path expressions.

Terminal reporting forwards the declared-file manifest and rank-zero trace events; other rank traces and checkpoint bytes remain on the training computer. Preserve that computer's output folder for a complete distributed archive.

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
