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
| `weave_support/launch.json` | Generated launch/checkpoint context and explicit GUI validation override | Weave |

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

An unsized stream can end an epoch at StopIteration. The starter's cosine scheduler still needs `len(train_loader)` or an explicit optimizer-update budget. A finite stream must be repeatable for multiple epochs. A random sampler run for a token budget does not guarantee each document is visited once.

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
    logits = model(batch["input_ids"].to(device))       # [B, T, V]
    labels = batch["labels"].to(logits.device)         # [B, T]
    count = int((labels != -100).sum())
    if count == 0:
        raise ValueError("Batch has no supervised targets")
    loss = F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    return StepResult(loss, weight=count)
```

Labels here are already shifted by the loader: input `[a,b,c]` predicts `[b,c,d]`. Shift once. `-100` excludes padding or other untrained positions from cross-entropy. `weight=count` lets Trainer combine mean losses correctly when batches have different numbers of supervised tokens, including across ranks. A scalar loss or `(loss, metrics)` is also supported, with equal microbatch weights. Adapt this hook for another objective or model signature.

Trainer calls backward, scales gradients, accumulates, clips, updates and reports. Do not repeat backward/optimizer.step in this hook. Optional `evaluation_step_fn` defaults to the loss hook under inference mode. `samples_fn(model, tokenizer, config, device)` can yield dictionaries containing `prompt`, `output`, and optional `metadata` for custom generation. Optimizer and scheduler hooks remain plain PyTorch functions; keeping AdamW and a short cosine/warmup function in your recipe is often clearer than a large configuration framework.

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
