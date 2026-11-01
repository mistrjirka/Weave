# Writing train.py for Weave

This guide describes the optional helpers shipped beside every generated model. Your recipe is ordinary Python/PyTorch. You can use the hook-based Trainer, use only reporting, or write the entire loop yourself. No pip-installed Weave package is needed.

## What lives where

| Location | Purpose | Who edits it |
| --- | --- | --- |
| `train.py` | Configuration, data/loss/optimizer/scheduler hooks and entrypoint | You |
| `data.py`, `augmentation.py`, other local helpers | Your readers, sampling, transforms and batching | You |
| `weave_support/model.py` | PyTorch network generated from your graph | Weave; edit the graph for lasting changes |
| `weave_support/load_tokenizer.py` | Loads the input-owned tokenizer saved with the graph | Weave |
| `weave_support/trainer.py` | Devices, AMP, accumulation, evaluation, reporting and checkpoints | Weave |
| `weave_support/token_cache.py` | Optional persistent token/document cache | Weave |
| `weave_support/cache_identity.py` | Stable identities for your own cache implementation | Weave |
| `weave_support/data_stream.py` | Optional packed stream with a restorable cursor | Weave |
| `weave_support/weave_training.py` | Reporting for a custom loop | Weave |
| `weave_support/launch.json` | Generated launch/checkpoint context and explicit GUI compute/validation overrides | Weave |

Save your stage to refresh its support files. Generated support can be inspected, but keep custom helpers outside that folder. Starting a run freezes another copy of the script, local helpers, graph, model and tokenizer. Editing the stage does not change that running copy. Cache files should live in a persistent location shared by stages/runs, rather than inside each frozen run directory.

The browser shows the real server paths and selected Python/PyTorch versions. Use that interpreter or your own compatible environment. Install the frozen package's `weave_support/requirements.txt`. The cache helper uses NumPy and fast tokenizers; it does not depend on PyTorch or HTTP. Building caches uses POSIX file locks (Linux/macOS); coordinate shared-filesystem locking with your cluster administrator for multiple hosts. Reading a completed cache is independent of the builder lock.

## The execution flow

```text
load generated Model → load tokenizer → get_dataloaders(tokenizer, config)
  → determine update budget → configure optimizer and scheduler
  → training_step → Trainer backward/accumulation/update/reporting
  → validation and generated examples → checkpoints
```

`get_dataloaders` may return arbitrary PyTorch-compatible iterables. The optional cache does not decide data formats, train/validation splits, chunk lengths, augmentation or batching. Sequence length belongs in your loader/model contract, not in TrainingConfig.

The starter's data hook intentionally raises NotImplementedError. Implement it or use the complete [cached JSONL example](examples/train_cached_jsonl.py), copied into a project stage as a separate recipe. That example is single-process, next-token training with whole short documents and dynamic right-padding; it is a teaching example, not a replacement for your custom bucket sampler.

## Configuration and hooks

```python
from weave_support.trainer import Trainer, TrainingConfig, StepResult

config = TrainingConfig(
    devices=[1],                     # [] CPU; [0] GPU 0; [0, 1] automatic DDP
    precision="fp16",                 # fp32, fp16 (CUDA), or supported bf16
    micro_batch_size=4,               # Your loader must actually use this value
    gradient_accumulation_steps=2,
    max_epochs=None,
    max_optimizer_steps=2000,
    learning_rate=3e-4,
    min_learning_rate=3e-5,
    async_evaluation=True,
    evaluation_device="cpu",          # Or a GPU index, e.g. 0
)
```

| Settings | Meaning |
| --- | --- |
| `devices`, `strategy`, `placement` | Device selection. `auto` uses one process or DDP; `split` needs a supported model and explicit module placement. |
| `precision`, `cpu_threads`, `seed` | AMP/compute configuration and base seed. Seed your custom generators too. |
| `micro_batch_size` | A convention passed to your data hook; Trainer does not resize user batches. |
| `gradient_accumulation_steps` | Number of microbatches per attempted weight update. The final partial group is flushed at an epoch boundary. |
| `max_epochs`, `max_optimizer_steps` | Loader passes versus successful optimizer updates. Set either to None; the first active limit reached stops training. |
| `total_optimizer_steps` | Read-only estimate set after the loader is built; scheduler hooks can use it. |
| `learning_rate`, `min_learning_rate`, `weight_decay` | Values available to your optimizer/scheduler hooks. They do not override a custom hook's implementation. |
| `gradient_clip` | Norm limit applied by Trainer after accumulation/unscaling. |
| `log_every`, `checkpoint_every`, `eval_every` | Intervals in optimizer updates. A skipped FP16 update does not advance the scheduler. |
| `eval_batches`, `sample_prompts`, `generation_tokens` | Validation budget and optional qualitative samples. |
| `generation_context_length` | Optional context crop for generated samples only. It does not set training sequence lengths. |
| `async_evaluation`, `evaluation_device` | Background validation on CPU or a selected GPU. The GUI can explicitly override these for a launch. |
| `evaluation_only` | Read-only: true inside a background evaluator. A data hook can return `(None, val_loader)`. |
| `data_fingerprint`, `trace_batches` | Source stat/SHA256 provenance and optional consumed-batch tracing. |
| `find_unused_parameters` | Optional DDP override, useful for some routed/conditional graphs. |
| `compile_mode` | `"off"` (default), `"default"`, `"reduce-overhead"`, or `"max-autotune"`. Compiles the model while preserving checkpoint keys. New batch shapes may compile separately. |
| `fused_optimizer` | Opt-in CUDA AdamW execution; preserves optimizer groups and restored state. Rejects CPU/custom optimizers rather than silently replacing them. |
| `profile_steps` | `0` disables profiling; `1`–`100` captures a bounded CPU/CUDA trace after two warm-up updates. Each rank writes `profiles/rank-N.json`. |

An unsized stream can end an epoch at StopIteration. The starter's cosine scheduler still needs `len(train_loader)` or an explicit optimizer-update budget. A finite stream must be repeatable for multiple epochs. A random sampler run for a token budget does not guarantee each document is visited once.

In **Launch settings → Training device settings**, choose **Use train.py settings** (the default) or **Use GUI overrides**. An override selects single-device, DDP, or split training, devices, precision, and optional block placement. It applies before worker startup and data hooks, and is saved in the run's `weave_support/launch.json`; your Python source is unchanged. Batch sizes and schedules remain owned by your script/data loader. Split training currently requires FP32 and either complete manual placement or a matching saved tuning profile. Validation execution is independent: select **Asynchronous** and **CPU** to keep validation off the training GPUs. Your data hook still needs to return a validation loader. These settings affect new launches, not running jobs. Exact resume still requires a compatible training configuration; use a new stage from weights when changing the training topology.

### Data hook

```python
def get_dataloaders(tokenizer, config):
    # Your reading/cache/dataset/sampling/batching choices.
    return train_loader, optional_validation_loader
```

For DDP, shard data explicitly with DistributedSampler or a rank-aware IterableDataset. Such a custom iterable declares `distributed_sharding = True`; that flag promises you implemented sharding, it does not perform sharding. Every rank must yield the same number of microbatches per epoch. Add worker sharding when using DataLoader workers. See [PyTorch's dataset and worker documentation](https://docs.pytorch.org/docs/stable/data.html).

### Loss hook: forward and loss only

```python
import torch
from torch.nn import functional as F

def training_step(model, batch, device):
    count = int((batch["labels"] != -100).sum())       # Count before GPU transfer.
    logits = model(batch["input_ids"].to(device))       # [B, T, V]
    labels = batch["labels"].to(logits.device)         # [B, T]
    if count == 0:
        raise ValueError("Batch has no supervised targets")
    loss = F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    return StepResult(loss, weight=count)
```

Labels here are already shifted by the loader: input `[a,b,c]` predicts `[b,c,d]`. Shift once. `-100` excludes padding or other untrained positions from cross-entropy. `weight=count` lets Trainer combine mean losses correctly when batches have different numbers of supervised tokens, including across ranks. A scalar loss or `(loss, metrics)` is also supported, with equal microbatch weights. Adapt this hook for another objective or model signature.

Trainer calls backward, scales gradients, accumulates, clips, updates and reports. Do not repeat backward/optimizer.step in this hook. Optional `evaluation_step_fn` defaults to the loss hook under inference mode. `samples_fn(model, tokenizer, config, device)` can yield dictionaries containing `prompt`, `output`, and optional `metadata` for custom generation. Optimizer and scheduler hooks remain plain PyTorch functions; keeping AdamW and a short cosine/warmup function in your recipe is often clearer than a large configuration framework.

### Reading loss and learning-rate curves

An optimizer update completes the configured gradient accumulation. The learning-rate metric records the rate used for that update; the scheduler advances after a successful optimizer update. An FP16 overflow that skips the update also skips the scheduler step. A resumed checkpoint restores the scheduler counter and optimizer learning rates.

`log_every=10` records the loss of every tenth optimizer update, weighted over that update's microbatches and ranks. It does not average all ten updates between log points. Variable sequence lengths and sampled documents can therefore produce a noisy training-loss curve even when validation loss is improving. Compare held-out validation on the same data, and inspect batch/context metadata before interpreting a jump as instability.

A cosine schedule declines slowly just after warmup. Judge its progress against the resolved `total_optimizer_steps`, rather than wall-clock time: at 8% of a long run, the learning rate can still be close to its peak. This is expected behavior, not evidence that the scheduler failed.

### Execution performance and memory-saving token loss

The GUI's **Execution and profiling (advanced)** section can override compilation, fused AdamW and profiling for a new launch without rewriting your script. Compilation is lazy: the first forward/backward for each shape can take much longer. Variable-length loaders with a small set of buckets are easier to compile than an unbounded range of shapes. CUDA graphs are attempted by `reduce-overhead` when eligible; they are not guaranteed. Split placement currently requires eager execution. Benchmark steady-state throughput, excluding compilation and profiler capture, before keeping an option enabled.

Profiling is optional and adds overhead during capture. Download the rank-0 trace from the run's Logs tab, or open `profiles/rank-N.json` from the run folder in Perfetto or Chrome tracing. Traces include tensor shapes; they do not copy dataset contents. Compare GPU kernels, CPU gaps, transfers and NCCL communication. Asynchronous validation on a training GPU still shares its memory and compute.

For a generated graph with one terminal **Vocabulary head**, the optional helper avoids materializing all vocabulary logits at once:

```python
from weave_support.trainer import language_model_step

def training_step(model, batch, device):
    return language_model_step(model, batch, device, chunk_size=1024)
```

This implements the same already-shifted, mean cross-entropy with `ignore_index=-100` and token-weighted `StepResult`. It preserves embedding/head weight tying, including their combined gradients. It checkpoints chunks of the final projection, recomputing them during backward: memory use improves, but extra projection work can make it slower. It uses ordinary PyTorch on CPU or CUDA without a third-party kernel. Omit `chunk_size` to use ordinary full logits. For custom losses, multiple outputs, auxiliary objectives or different input signatures, keep your own training hook. This helper does not automatically include MoE auxiliary losses.

At the model level, `model(input_ids)` still returns logits. `model(input_ids, loss_targets=labels, loss_chunk_size=1024)` returns the scalar token loss only on eligible freshly exported graphs. In DDP, call the wrapped model as usual; do not bypass its forward with `model.module`.

Existing run snapshots remain unchanged. When resuming, **Apply current Weave runtime fixes** re-exports the run's recorded graph with the current runtime and records the new code in the restart snapshot. Tensor names/shapes, optimizer mappings, data identity and cursor compatibility are still checked before training updates. Ordinary resume keeps the frozen model runtime. A numerical runtime fix can change outputs despite compatible tensor shapes; the revision records this change.

## Optional document/token cache

### Any source format: you supply the reader

```python
from weave_support.token_cache import cache_documents

def read_documents():
    for path in files:
        if path.suffix == ".jsonl":
            with path.open() as source:
                for line in source:
                    yield json.loads(line)["text"]
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(columns=["body"]):
                yield from batch.column("body").to_pylist()
        else:
            raise ValueError(f"Add a reader for {path}")

cache = cache_documents(
    directory=Path.home() / "shared/weave-token-cache",
    sources=files,
    tokenizer=tokenizer,
    read_documents=read_documents,              # Factory, called only on a miss
    policy={"reader": "mixed-text-v1"},         # Bump when preprocessing changes
    eos_token_id=tokenizer.eos_token_id,        # Optional, explicit separator
    batch_size=512,                            # Tokenization batch, not GPU batch
    # total_documents=known_document_count,    # Optional GUI percentage
)
```

Add `import json` and `from pathlib import Path` in a real recipe. `files` is your ordered list of sources. Readers can combine formats, columns, archives or databases. For sources that are not local files, pass `sources=[]` and include a stable dataset revision/snapshot identity in `policy`; the helper cannot infer changes to an external service. Readers must be finite for this cache format.

Default readers yield strings. Text is batch-tokenized with `add_special_tokens=False`; EOS is appended only when requested. An EOS ID must already exist in the tokenizer. Do not automatically invent a new separator that exceeds the model vocabulary. Empty individual documents are permitted; a cache with no tokens is rejected.

The cache preserves document boundaries. `len(cache)` counts documents; `cache[i]` returns a CachedDocument whose `tokens` are a read-only NumPy view, `labels` are optional, and `metadata` is your JSON dictionary. `cache.tokens` and `cache.offsets` expose the flat token storage and document offsets for custom bucket/index implementations. Copy slices before in-place augmentation or tensor operations that modify them. NumPy mappings reopen lazily in DataLoader workers instead of copying the entire corpus when pickled.

The helper does not pick a validation fraction, concatenate documents, select buckets, pad, shift labels or choose sequence lengths. Use a normal Dataset/IterableDataset, sampler and collate function over these arrays. The default project template remains small.

### Custom records, SFT labels and metadata

```python
from weave_support.token_cache import CachedDocument

def encode_conversation(row):
    encoded = tokenizer.apply_chat_template(
        row["messages"], tools=row.get("tools"), tokenize=True,
        add_generation_prompt=False, return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids = encoded["input_ids"]
    mask = encoded.get("assistant_masks")
    if mask is None or len(mask) != len(ids) or not any(mask):
        raise ValueError("This template must provide an assistant-generation mask")
    labels = [token if assistant else -100 for token, assistant in zip(ids, mask)]
    return CachedDocument(ids, labels, {"source": row["source"], "row": row["row"]})

cache = cache_documents(
    directory=cache_directory,
    sources=files,
    tokenizer=tokenizer,
    read_documents=read_conversation_records,
    encode=encode_conversation,
    policy={"reader": "chat-v1", "chat_template": tokenizer.chat_template},
)
```

Custom readers can yield any record type; `encode(record)` returns CachedDocument. Custom encoding owns EOS/special-token placement, so do not also pass `eos_token_id`. Custom encoding is called per record; use a reader that yields pre-encoded CachedDocuments with `encode=lambda document: document` if you need your own batched encoder.

If labels are supplied, every document must have aligned labels; values are token IDs or -100. Labels in the cache align with token positions. For next-token training your dataset still shifts both arrays consistently: inputs `tokens[:-1]`, targets `labels[1:]`. Assistant masks require a chat template supporting generation regions; the helper does not guess which tokens belong to an assistant. Use the format used during training, as described in [Transformers' chat-template documentation](https://huggingface.co/docs/transformers/chat_templating).

### Persistence, progress and concurrent builds

A key covers the fast tokenizer's encoding backend, special-token roles, EOS, source paths/size/mtime and the explicit preprocessing policy. Tokenizer export folders are excluded, so a new Weave run or a reboot does not invalidate a completed cache. Same-size files with artificially preserved mtimes cannot be distinguished; include trusted content digests or dataset revisions in your policy if needed. Changing arbitrary Python preprocessing code is not detected automatically: bump its policy version.

A completed cache is published atomically after validating its manifest/files. Failed builds remove their own temporary directory; a hard-killed process can leave an ignored `.tmp` directory. Those partial caches are not resumed. Cache construction on a shared POSIX filesystem uses a per-key lock, so concurrent processes build once; waiting builders reuse the result. `lock_timeout` defaults to one hour. Corrupt completed caches raise an error instead of silently training from truncated data. Avoid deleting a cache while another run reads it. Token IDs use uint16 or uint32 based on the largest tokenizer ID, including added tokens; labels use signed int64.

**GUI progress is automatic with Trainer.** It reports tokenizer/loader initialization and cache phases (`waiting`, `building`, `complete`, `reused`), document counts and token counts. Supply `total_documents` for a percentage; otherwise the progress bar is indeterminate. Progress updates are emitted about every five seconds between tokenization batches. A single very large batch or document can take longer. The default also prints to terminal/Logs. Optional `on_progress(progress_dict)` adds your own callback and does not disable GUI reporting. Independently launched Trainer recipes send these preparation events through the existing background HTTP reporter. A custom loop can attach `cache_progress_reporting(callback)` explicitly, or use terminal progress.

### Where augmentation belongs

| Transformation | Placement | Consequence |
| --- | --- | --- |
| Cleaning, fixed formatting or deterministic text changes | Reader/custom encoder before caching | Runs once; include its version/settings in policy |
| Random masking, cropping or token-level corruption | Dataset after reading cached IDs | Can vary per visit/epoch; copy read-only arrays first |
| Padding, packing, mixing examples or selecting bucket lengths | Sampler/collate/custom dataset | Your batching policy remains editable |
| Paraphrasing, character noise or stochastic tokenization | Keep original records accessible and encode after augmentation, or cache deliberate variants | A token-only cache cannot reconstruct the original text/fields losslessly |

For reproducibility, derive augmentation randomness from a stable sample ID/seed/epoch or checkpoint your generator state and data cursor. Seed DataLoader workers explicitly when using other random libraries; [PyTorch documents those worker seeding rules](https://docs.pytorch.org/docs/stable/data.html). Caching is optional for small datasets, continuously changing sources or tasks where live transformations are the intended training distribution. This format covers finite tokenized documents and optional aligned labels, not image/audio tensors or unbounded streams.

## Provenance and exact resume

Expose `dataset.source_files` (or `files`) as existing paths. A dataset built over TokenCache can use `self.source_files = cache.source_files`, which includes original sources and cached artifacts. Trainer records sizes/mtimes by default; `data_fingerprint="sha256"` additionally reads and hashes full sources. For precise consumed records, add `_weave_source` to each batch (JSON-compatible metadata identifying source rows/documents/spans). It is reporting metadata, not a model argument; your training_step chooses what to pass to the model.

Exact resume requires loader `state_dict/load_state_dict`, or dataset state methods with `num_workers=0`. Cache storage alone is not a resumable sampler. Save the consumed cursor, epoch, generator state or deterministic order information needed to produce the next batch. DataLoader prefetch queues with multiple workers cannot be reconstructed from a dataset cursor alone. An ordinary map-style Dataset plus shuffle is supported for training but may offer weights-only checkpoints. Trainer warns rather than pretending exact resume is available. Code/model/data identity and process count are checked when resuming; use a new stage from weights when changing the recipe.

Snapshots capture graph, code, checkpoint state, tokenizer and traces. They record source identities, but source files/caches are copied only when requested with Include source data; account for disk size. Editing an active stage affects future runs, not the existing frozen run.

## Evaluation and inference

A background evaluator starts a fresh process, loads immutable checkpoint weights and invokes your hooks. Use `config.evaluation_only` to avoid constructing the training loader. Shared completed caches are reused; separate preparation streams prevent evaluator startup from replacing training progress. CPU evaluation uses FP32; sharing a training GPU competes for memory/compute. The run waits for final evaluation before marking itself complete. See [training operations](TRAINING.md) for placement and launch details.

Checkpoint chat/text completion uses the frozen model and tokenizer without importing train.py. A custom tokenizer mutation in your script may need to be represented in saved tokenizer settings or a chat-template override for inference. A base pretrained model is not automatically a chat model. Chat tool-call text is displayed, not executed.

## Your own loop with reporting only

```python
from weave_support import Run

with Run(optimizer=optimizer, total_steps=total_updates) as run:
    for batch in train_loader:
        optimizer.zero_grad()
        loss = your_loss(model, batch)
        loss.backward()
        optimizer.step()
        run.log_loss(loss)
```

Optimizer updates and learning rates are tracked automatically. Add `run.sample(...)`, `run.checkpoint(path, metadata=...)` or `run.poll_commands()` where your custom loop supports those features. This reporting helper does not train, save weights or handle cooperative stop on your behalf. HTTP sends/polling use a background thread; records are also written locally. A missing browser/server does not hold up optimizer requests, although serialization, local disk writes, tensor-to-host metrics and checkpoints have costs.

## Troubleshooting

- **Running but no metrics:** inspect data preparation and Logs first. A data hook may be reading/indexing/tokenizing before the first optimizer update. The cache helper reports that phase automatically.
- **Cache rebuilt for identical data:** use persistent cache storage, the same policy/source identities and content-based tokenizer identity. Do not use tokenizer.name_or_path as a key; frozen runs have different folders.
- **Cache rejected as incomplete:** preserve it for inspection, then remove that individual cache key directory and rebuild. Do not clear unrelated caches or active readers.
- **Loss stays absent:** Trainer reports the default loss; a custom loop needs log_loss/run.log. An all-masked batch has no learning signal and should be rejected or filtered.
- **Variable-length batches fail stacking:** supply a collate function or yield pre-batched tensors with `batch_size=None`. Cache storage does not select a collator.
- **Shapes differ:** inspect the generated model's inputs/outputs, shifted label shape and vocabulary capacity. An input tokenizer must fit the embedding table and output head.
- **My edits disappeared from a run:** runs are frozen snapshots. Start a new run from the stage you edited; keep code changes in the stage's train.py/local helpers.

### Adopting an existing token cache

`TokenCacheImport(token_file, offsets, dtype="<u2")` can be passed as
`existing=` to `cache_documents`. Offsets contain N+1 token positions, starting
at zero and ending at the file's token count. This migration option skips the
reader and tokenization. You must verify that the imported IDs were produced
from the supplied sources, tokenizer and preprocessing policy. The helper
checks bounds, dtype and token ranges, but cannot prove their provenance.
On the same filesystem it hard-links the immutable token file; across
filesystems it copies it. Never modify an imported token file in place.
Subsequent calls reuse the completed cache without needing `existing`.

## Consuming a saved GPU tuning profile

See [host-wide GPU tuning](TRAINING.md#host-wide-gpu-performance-tuning) for benchmarking and automatic compatibility checks. Default `TrainingConfig(use_gpu_tuning=True)` permits a matching launch profile; set it false to preserve your exact manual allocation. No profile is required to train, and a benchmark does not edit train.py.

For a normal map-style dataset, use a standard `DataLoader(..., sampler=DistributedSampler(dataset), batch_size=config.micro_batch_size)`. The Trainer can rebuild that standard loader with measured unequal batches while preserving the global microbatch and matched step counts. Custom collation is retained. It does not replace DataLoader subclasses, IterableDataset batching, or custom batch samplers.

For a custom stream, `config.rank_micro_batch_sizes` is the proposed full-update allocation, ordered like `config.devices`; `config.local_micro_batch_size` selects this process's share. The original `config.micro_batch_size` remains the untuned per-GPU baseline. Implement weighted data sharding, equal optimizer/accumulation boundaries, appropriate tail handling, and worker sharding yourself. Set `loader.weave_tuning_applied=True` only after doing so; the runtime still checks that rank epoch/batch boundaries agree. A loader that ignores the allocation is allowed and gets an explicit tuning-skipped diagnostic, rather than dropping/slicing its records. Use token-weighted `StepResult` for correct loss weighting across unequal batches; the hook Trainer rejects scalar/tuple losses for an active unequal allocation. A cached-document/VSL sampler remains your experiment code and is not changed by profiling.

### Pause, edit, and resume

**Pause training** finishes at a safe optimizer boundary, writes a resumable checkpoint and stops the worker processes. **Resume training → Previous script** restores that checkpoint's model, optimizer, scheduler, mixed-precision scaler, RNG and data cursor. It starts a new execution linked to the previous run, keeping the optimizer update count. The loader must implement the documented state protocol; a weights-only checkpoint cannot restore its data cursor.

To apply saved Python edits, choose **Resume using → Edited stage script**. Save the editor first, or edit the displayed server `train.py` path over SSH. Weave freezes that script and the stage's user helper files, while retaining the checkpoint's generated model and tokenizer. It restores existing optimizer/scheduler state; it does not silently apply a new learning-rate schedule or reset progress. Changes to logging, validation, sample hooks and compatible loss logic can be used. Dataset fingerprints, data cursor compatibility, process count and non-mutable training settings remain checked. Arbitrary Python behavior cannot be proven equivalent: changes to loss or augmentation can change the experiment even when tensor shapes match.

Model state must load strictly. Optimizer parameter names, shapes, group order, optimizer type and scheduler type must match. Older checkpoints without parameter-name mappings need one resume with **Previous script** and a new checkpoint before edited-script resume is supported. Changes to topology, batch configuration or optimization settings should start a new stage from weights.

Every attempt gets a durable `resume-attempts/<attempt>/` record with the previous and requested Python source, source run, selected checkpoint, validation settings, source hash, timestamp and any pre-launch rejection. A launched execution has its own complete frozen files and `revision.json`, with SHA-256 hashes, checkpoint ancestry and the restoration policy. The run's settings dialog exposes this record's path. Previous files and checkpoints are preserved; datasets are referenced by their recorded provenance, not copied in full for every restart.

If restoration fails, the UI shows a compatibility error and **Retry resume**. Fix the editable script and retry with **Edited stage script**, or select **Previous script** to use the last checkpoint's recorded source. Both retry from the last usable checkpoint, and neither overwrites your editable files. Failed attempts remain in history. Repeated submissions while the same continuation is active return that continuation instead of launching another worker.

The activity badge in the top navbar opens **Active training**, listing running/preparing jobs across all projects with direct dashboard links. It refreshes every five seconds while the page is visible. It does not limit concurrent launches. Each training dashboard also has a **Run** selector. Bookmarked run URLs retain their selected history; when another run of that stage is active, a prominent notice links to the running training.

To reuse a recipe, open **Stage setup → Python script → Copy script from project…**. Choose another saved project and one of its training stages, review the saved server script and path, then select **Replace editor script**. Nothing is written until you use **Save file**. This copies `train.py` only: its Python configuration comes with it, while the current graph, GUI launch overrides, dataset files and user helper files stay separate. Adapt paths/imports as needed. Copying does not change a frozen running job.

### Waiting for a shared cache in DDP

All ranks can call `cache_documents`; its file lock ensures only one process builds a given cache. Waiting ranks forward the builder's document/token counts to the UI, including when another terminal process is building it. Counts inferred from an older builder's buffered output files are marked approximate.

If your loader does additional preparation only on rank 0, use the CPU preparation barrier instead of a GPU collective:

```python
from weave_support.trainer import preparation_barrier

# Inside get_dataloaders(), on every rank:
if rank == 0:
    prepare_extra_files()
preparation_barrier()
```

Every rank must call this helper in the same order. The Trainer also waits on this CPU group after all loader hooks finish, before starting training collectives. `TrainingConfig(preparation_timeout_seconds=86400)` controls the preparation timeout (24 hours by default). Normal training collective timeouts remain unchanged. Calling `torch.distributed.barrier()` directly in a data hook still uses the default training group and can time out during a long preparation job.
