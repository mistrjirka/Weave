# Serius performance and learning-rate investigation

Measured locally on 2026-11-01. These results describe this model and machine;
they are not expected speedups for every Weave graph.

## Setup

- PyTorch 2.6.0+cu126, Triton 3.2.0, FP16 automatic mixed precision and scaling.
- PCI bus order: GPU 0 is an RTX 2080 Ti with 22 GB; GPU 1 is a V100 SXM2 32 GB.
- Serius has 49,160,448 parameters: width 768, MHA with 12 heads, SwiGLU width
  2,176, five distinct transformer blocks, fourteen block executions per token,
  and a tied 16,000-token embedding/output matrix.
- All optimizer comparisons use DDP on both GPUs, `compile_mode="default"`,
  accumulation 2 and an effective batch of 128 sequences. Length 511 uses
  32 sequences per rank/microbatch; length 8,191 uses two.
- Timing uses synchronized wall-clock medians after four warm-up updates;
  compilation and warm-up are excluded. There are 16–20 timed updates per trial.
  Fixed synthetic batches isolate execution cost from data loading, validation
  and reporting. Their losses are not evidence of learning quality.
- Every comparison starts from the same model checkpoint with fresh optimizer
  state. AdamW uses LR 0.0003, betas (0.9, 0.95), and weight decay 0.1.

## RoPE training history limits the quality conclusions

The initial run trained through update 2,937 with a RoPE precision bug: rotation
angles were cast to FP16 before sine/cosine. Rounding becomes particularly
severe at long positions. The resumed run explicitly refreshed the runtime and
used FP32 angles and trigonometry from update 2,938 through 3,653. Computing
these values in FP32 before casting the resulting sine/cosine tables also
matches the approach in the
[Transformers Llama implementation](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L137-L156).

The old run's trace records 62 updates at the 4k bucket and 170 at the 8k bucket,
out of 2,937 completed updates. Smaller contexts also had angle rounding, though
less severe. The first observed long-context update was 2,280; the loader's
2,288-update curriculum boundary counts consumed batches, including skipped
FP16 updates, rather than only successful optimizer updates.

The fix changes future computation; it does not undo the previous training.
Continued improvement in validation does not establish parity with training
correctly from initialization. All optimizer experiments use the fixed runtime
and the same inherited checkpoint, so their timing comparison remains useful,
but their short quality checks describe continuation from affected weights.
A fresh training run under the fixed runtime is needed for a clean baseline.

## Optimizer throughput

| Optimizer | 511-token sequences | 8,191-token sequences |
| --- | ---: | ---: |
| AdamW, default CUDA execution | 114,021 tokens/s | 43,739 tokens/s |
| Fused AdamW | 114,566 tokens/s | 43,728 tokens/s |
| Muon + fused auxiliary AdamW, replicated per rank | 110,176 tokens/s | 42,836 tokens/s |
| Muon + fused auxiliary AdamW, distributed optimizer | 106,762 tokens/s | Not measured |

Fused AdamW changes short-context throughput by about +0.5%; the long-context
results are effectively tied. This optimizer change is not a major source of
throughput improvement here. Compilation provides the larger improvement; the
shared runtime fixes also address correctness. The earlier matched-shape DDP experiment measured
approximately 80k tokens/s in eager execution and 114k with compilation/fused
AdamW at short context; at 8k it measured about 39k versus 44k.

Muon is slower per update in these tests. This does not determine how many
updates it would need to reach a target validation loss. The
[Muon scaling paper](https://arxiv.org/abs/2502.16982) motivates evaluating that
separately rather than interpreting raw tokens/s as training efficiency.

The experiment uses the MIT-licensed
[KellerJordan/Muon reference](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806).
Only hidden matrices use Muon; the tied embedding/head and normalization weights
use fused AdamW. The original Muon LR is 0.02, momentum 0.95, with five
Newton–Schulz iterations. These GPUs lack native BF16, so the experimental
orthogonalization normalizes in FP32 then uses compiled FP16 matrix operations.
A numerical check against FP32 on the model's three matrix shapes gave finite
outputs and approximately 0.27–0.31% relative RMS difference. This is an adapted
implementation, not the newer built-in `torch.optim.Muon`; the installed
PyTorch 2.6 does not provide that optimizer. Neither experimental Muon variant
has been installed into the project recipe or Weave's templates.

## Short real-text check

Each trial continued the same step-2,937 weights for 100 updates on identical
512-token training buckets, using fresh optimizer states and constant learning
rates. Evaluation used the same eight held-out text batches before and after.
This deliberately differs from the actual variable-length training schedule;
it is a stability check, not a full convergence or exact-resume comparison.

| Optimizer | Initial held-out loss | Loss after 100 updates |
| --- | ---: | ---: |
| Fused AdamW, LR 0.0003 | 3.6184 | 3.6330 |
| Fused AdamW, 8,192-token chunked loss | 3.6184 | 3.6337 |
| Muon LR 0.02 + auxiliary AdamW LR 0.0003 | 3.6184 | 4.0059 |
| Muon LR 0.005 + auxiliary AdamW LR 0.0003 | 3.6184 | 3.6670 |

All updates remained finite. Neither tested Muon setting beat the AdamW
control. The slight worsening of the AdamW control also illustrates why a
short fixed-context continuation with reset optimizer state should not be
compared directly with the original run's validation curve. Muon's learning
rate and weight decay need their own tuning: equal numeric weight decay does
not mean equal shrinkage when the learning rates differ. These tests do not
establish which optimizer is better when trained and tuned from scratch.

The adapted replicated FP16 Muon also accumulated small parameter differences
between the heterogeneous GPUs: relative RMS differences of about 0.031% at
LR 0.02 and 0.010% at LR 0.005 after 100 updates. The AdamW control remained
identical across ranks. This is another reason to keep this experimental
implementation out of an existing training run; an implementation that shares
each computed update, such as the distributed reference variant, avoids
independently computing the orthogonalization on both GPU architectures.

## Other measured headroom

Separate compiled/fused DDP trials at length 511 measured:

| Execution | Tokens/s | Peak allocated memory per GPU |
| --- | ---: | ---: |
| Original microbatch/accumulation | 114k | 10.5 GiB |
| 8,192-token chunked output loss | 118k | 7.4 GiB |
| Double microbatch, accumulation 1, chunked loss | 122k | 13.2 GiB |

The larger microbatch preserves the effective batch but changes the custom
loader's interpretation of its saved microbatch cursor. It has not been applied
to an exact resume. Without allocator fragmentation handling, the unchunked
larger microbatch ran out of memory on the 2080 Ti. With
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, it fit but reached only 118k
tokens/s while allocating 19.6 GiB.

At 8k context, chunked loss measured 44.2k versus 43.9k tokens/s and reduced
allocated memory from about 10.5 to 7.4 GiB. Treat these small throughput
changes as preliminary, not a guaranteed speedup. The shared helper's loss and
all gradients are checked against ordinary cross entropy, including masked
labels and tied embedding weights.

Serius's editable stage now uses `language_model_step(..., chunk_size=8192)`.
The full script was backed up before this change. Configuration, data-loader,
optimizer and scheduler code are unchanged; the existing frozen run and
checkpoint are unchanged. Use **Resume training → Edited stage script** to
retain update 3,653 and pick up the changed loss hook. The saved GUI override
still selects DDP on GPUs 0 and 1, and compilation/fused AdamW remain enabled
in `train.py`. The larger-microbatch experiment and Muon are not applied.

A profiler capture of one compiled, fused-AdamW update on real 511-token text
showed GPU kernels active for about 578 of the 583 ms between its first and last
kernel. Matrix multiplies accounted for 46% of summed kernel duration, compiled
fused kernels 23%, attention 12%, NCCL 8%, and fused AdamW less than 1%.
Kernel durations can overlap, so these percentages are not wall-clock shares;
profiling also adds overhead. This sample points to GPU computation, rather
than large CPU launch gaps, as the current short-context limit. It does not
establish the same proportions for the long-context buckets.

## Learning rate and loss

The actual Serius scheduler plans 45,763 successful optimizer updates, with
457 warm-up updates, peak LR 0.0003 and minimum LR 0.00003. At the stopped
update 3,653, only 7.98% of this schedule had completed.

| Update | Configured LR used for that update |
| ---: | ---: |
| 457 | 0.0003000 |
| 3,653 | 0.0002967 |
| 10,000 | 0.0002715 |
| 20,000 | 0.0001939 |
| 30,000 | 0.0001029 |
| 40,000 | 0.0000406 |
| 45,763 | 0.0000300 |

Every logged LR matches the configured formula to floating-point precision.
There are no LR increases after warm-up. The checkpoint preserves scheduler
counter 3,653 and the next LR; the restart did not reset warm-up or decay.
The slow initial decline follows the cosine curve.

Training metrics log one update every ten updates, rather than the mean of
all ten. Recent 64-token buckets averaged loss 4.03, while 2,048-token buckets
averaged 3.24. Differences in available context and sampled documents account
for substantial point-to-point variation. Sequence lengths above 1,024 first
appear at logged update 2,280, changing the batch mix. Held-out validation loss
improved from 3.60 at update 2,900 to 3.46 at update 3,600. These observations
do not show a scheduler failure or recent divergence.

Raw local measurements, logs and the full LR/loss figure are in the ignored
`data/performance-serius/` directory. The original run was stopped at update
3,653; experiments do not modify its checkpoint, data cache or frozen script.

## Fresh WSD training run

After the RoPE investigation, the user requested training again from random
initialization with warmup–stable–decay (WSD). Serius's editable `train.py` now
uses 1,000 warm-up updates to LR 0.0003, holds that rate through update 36,610,
and linearly decays over the final 9,153 updates to 0.00003 at update 45,763.
Fused AdamW, compilation, chunked loss, DDP on both GPUs, and the same token
cache are retained. Validation every 100 updates preserves the previous run's
live cadence. The earlier script was backed up before editing; previous runs
and checkpoints remain available.

WSD is supported by [MiniCPM](https://arxiv.org/abs/2404.06395). The peak is a
conservative starting choice informed by the 0.0003 AdamW pretraining recipe
for four-pass LoopLM in
[Closing the Loop](https://arxiv.org/html/2610.00673v1), rather than a measured
optimum for this smaller model. That paper uses a cosine schedule; this run
combines its LR reference with the separately researched WSD schedule.
SmolLM2-135M's published LR of 0.003 and MiniCPM's larger rates are not directly
transferred across different batches, architectures and parameterizations.

New Weave starter scripts also default to WSD, with proportional 2% warmup
and 20% final linear decay. These are editable Python constants. Existing user
scripts are not automatically converted.


## Validation plateau audit of the fresh run

The fresh Serius run's 8,192-token validation loss fell from 8.19 at update
100 to 5.41 at update 1,000, then stayed around 5.36–5.41 through update
2,100. This is a real measured plateau. Until update 2,279, the project loader's
short-sequence curriculum only supplied buckets up to 1,024 tokens (actual
inputs up to 1,023). Validation already used 8,192-token inputs throughout.
The first 8k training batch completed at update 2,280. Thus early training
loss and validation loss measure substantially different context regimes.

A read-only audit loaded the frozen update-2,000 checkpoint on CPU in FP32.
For the **same first 8,192 held-out tokens**, only the attention context and
position resets were changed:

| Evaluation layout | Mean next-token loss |
| --- | ---: |
| One 8,192-token sequence | 5.133783 |
| Sixteen 512-token sequences | 3.283151 |
| Eight 1,024-token sequences | 3.217267 |

This supports a short-to-long context generalization problem during the
curriculum, rather than a flat metric caused by reporting stale weights.
The subsequent live result strengthens that explanation: the unchanged 8k
validation loss dropped from **5.3942 at update 2,200 to 3.9515 at update 2,300**,
after the first long-context training batch at update 2,280. Neither the running
script nor evaluation settings were changed during the audit.
It does not by itself establish final convergence or an optimal curriculum.
Separate short- and long-context validation curves would make this distinction
clearer without changing the existing 8k metric midway through training.

Correctness checks on the actual run:

- Independent ordinary cross-entropy and chunked projection loss agreed
  within 2.4e-7 on a 512-token held-out prefix. Independent token-wise loss
  also reproduced the complete 8k sequence loss.
- The first full validation batch recomputed on CPU gave 5.248840; the saved
  GPU FP16 evaluator reported 5.248831, an absolute difference of 9.3e-6.
- Changing future tokens left all earlier logits unchanged (maximum absolute
  difference zero). The input/label arrays have the expected next-token shift.
- All checkpoint parameters were finite, and embedding/head weights remained tied.
- All 6,932,204 cached training chunks were checked against document offsets:
  none included a held-out document or extended beyond its source document.
- Async evaluators loaded their own immutable checkpoint with strict state
  loading and emitted the corresponding checkpoint step. The plateau began
  before validation moved to GPU 0 at update 1,191. Evaluation used inference
  mode, and loss aggregation weighted batches by supervised token count.

Raw audit scripts and results are in the ignored local directory
`data/performance-serius/validation-audit/`. The audit did not modify the
project script, checkpoint, cache, or running training process.


## CPU validation thread sweep

A read-only CPU FP32 sweep used the same frozen update-2,000 Serius checkpoint
and first held-out validation batch (two sequences of 8,192 tokens), including
the vocabulary projection and chunked cross-entropy. GPU training continued
concurrently; these are batch timings, not complete evaluator startup timings.

| CPU threads | Seconds per validation batch |
| ---: | ---: |
| 2 | 58.46 |
| 4 | 38.66 |
| 8 | 17.77 |
| 12 | 12.36 |
| 16 | 12.99 |
| 24 | 10.23 |
| 48 | 11.72 |

Repeats at two and 24 threads took 58.83 and 10.29 seconds, respectively.
Every configuration produced loss 5.248840. At 48 threads, the 20-batch pass
is estimated at 234 seconds, plus data setup, checkpoint loading and generation.
This is roughly five times faster than two threads. The slightly faster 24-thread
result shows that using all hardware threads is not always the fastest setting.
The machine has 24 physical cores and 48 logical CPUs.

As requested, new Weave training/CPU validation configurations default to all
CPUs available to the process, respecting Linux CPU affinity. Explicit limits
remain supported. Serius's explicit four-thread limit was removed so it follows
the automatic default; its stage now selects asynchronous CPU validation every
100 updates, with the existing 20-batch evaluation budget. The previous script
and settings were backed up. A checkpointed resume preserves training state
while replacing the old frozen runtime with the new CPU default.

Raw measurements are in the ignored local directory
`data/performance-serius/cpu-validation/`. If validation takes longer than the
100-update interval, it continues in the background and the queue retains the
latest pending checkpoint; it does not force training to wait for every pass.

## Reusing validation workers: measured prototype

At the user's request, Serius was stopped safely at update 4,121. Its saved
stage selects asynchronous validation on V100 GPU 1 every 100 updates, retaining
20 batches per pass. The live device change had applied at update 4,000; the
in-flight CPU job was subsequently cancelled by the cooperative stop.

Existing V100 job timelines show 28–31 seconds between subprocess launch and
initial data preparation, approximately 0.3 seconds of cached data preparation,
10–11 seconds of loss evaluation, and under one second of sample generation.
A read-only import profile of the stage took 31.57 seconds: PyTorch cumulative
import time was 8.77 seconds and the generated tokenizer helper's cumulative
import time was 21.82 seconds, including Transformers AutoTokenizer and its
model/flex-attention/Dynamo import chain. These cumulative times overlap within
nested imports; they are not independent wall-clock components.

A standalone prototype retained one actual Serius model on an otherwise idle
V100, reloaded checkpoints strictly, and evaluated the same 40 held-out 8k
sequences in 20 batches. A fresh AMP context was used for every pass, preventing
cached FP16 weights from surviving an in-place checkpoint replacement.

| Pass | Checkpoint update | Checkpoint reload | Loss evaluation | Mean loss |
| --- | ---: | ---: | ---: | ---: |
| First | 4,121 | 0.69 s | 6.32 s | 3.439360 |
| Warm, different weights | 4,000 | 0.66 s | 4.17 s | 3.445572 |
| Warm, original weights again | 4,121 | 0.61 s | 4.19 s | 3.439360 |

The changed loss for update 4,000 and exact repeat for update 4,121 verify that
the prototype did not accidentally retain the previous checkpoint's weights.
These timings exclude tokenization setup, generated samples and contention
with GPU training, so they are not a prediction of a complete production pass.

PyTorch allocated about 200 MB for the initial model, retained about 219 MB of
tensors after evaluation, peaked at 1.62 GB allocated, and reserved 2.43 GB in
its CUDA allocator. CUDA context/library memory is additional. A production
worker can keep the frozen model, tokenizer, repeatable validation loader,
CUDA state and allocator warm; the dataset can remain memory-mapped instead of
loading the full token cache into RAM.

Recommended production design: one separate persistent worker per run/device,
receiving immutable checkpoint paths and settings. Reload every checkpoint
strictly, open a new AMP context per job, reset validation loader/RNG state to
preserve the existing evaluation sample set, and attribute all outputs to the
loaded checkpoint. Device changes or worker failure should replace the worker;
pausing training must terminate it and release GPU memory. Stateful custom
scripts need a fresh-process fallback. This prototype does not yet replace the
production evaluator, which still launches one process per pass.

The prototype and import profile are in the ignored local directory
`data/performance-serius/validation-startup/`. Model weights, the training script
and the frozen run were not changed by these measurements.

## Production reusable evaluation and DDP contention

The reusable worker is now implemented in Weave's portable Trainer. New starter
scripts enable it with `reuse_evaluation_worker=True`; direct TrainingConfig
construction keeps False for backward compatibility. It retains model,
tokenizer, repeatable loader, CUDA context and allocator per run/device. Every
job strictly reloads immutable weights, opens its own reporting scope and AMP
context, and resets supported loader/sampler and RNG state. Device changes
replace the worker; stop/failure closes it. One-shot iterators and persistent
DataLoader workers fall back to fresh processes. Custom hook globals or custom
RNGs may require explicitly disabling reuse.

CPU evaluation no longer initializes CUDA RNG contexts, and GPU evaluation only
preserves RNG on model devices. A real V100 worker test verified checkpoints
4,121 → 4,000 → 4,121, with losses 3.439360 → 3.445572 → 3.439360. One worker PID
served all three requests. Only the V100 appeared in that process's GPU memory
record (2,696 MiB including its CUDA context). Complete warm jobs took 6.35 and
5.90 seconds on the idle V100, including generated samples, weight loading and
reporting; the initial cold job took 37.04 seconds.

A separate isolated benchmark measured concurrent validation while training
Serius with DDP on both GPUs. It used FP16 AMP, compiled model, fused AdamW,
8,192-token logit chunks, two accumulation microbatches and 16,384 nominal tokens
per microbatch per rank. Training CPU threads were fixed at four per rank.
Real cached bucket windows were reused for controlled timing; no user training
weights were modified. Each shape ran baseline / validation / baseline /
validation blocks of 50 optimizer updates, with one full 20-batch validation job
in each validation block. The same warm worker served every job.

| Training sequence | Baseline tokens/s | Tokens/s with one validation per 100 updates | Throughput reduction |
| --- | ---: | ---: | ---: |
| 1,023 | 104,607 | 96,020 | 8.2% |
| 8,191 | 41,242 | 39,320 | 4.7% |

The cadence column combines each paired 50-update baseline and validation block
into a 100-update window with one validation pass; its no-validation reference
uses the repeated baseline measurements. The long-context baseline varied
between 77.7 and 81.2 seconds per 50 updates, so these are approximate costs, not
universal speed guarantees for arbitrary model/data shapes. Shared-GPU worker
jobs took about 10.9–11.8 seconds after warmup. Compilation/worker startup and
training-side snapshot creation were excluded from the throughput table.

Temporary validation snapshots now automatically use sufficiently large Linux
`/dev/shm`, falling back to disk elsewhere or on insufficient space. Two trials
writing the actual 196.7 MB model snapshot from GPU 0 took 2.40/1.86 seconds on
disk, versus 0.227/0.224 seconds in RAM. Explicit `evaluation_snapshot_storage="disk"`
remains available. Normal periodic, pause and user checkpoints remain durable
on disk. Tests cover RAM becoming full without deleting an active worker's
previous snapshot, low-space fallback, and switching back to explicit disk.

The speedup primarily removes repeated validation startup; that startup already
overlapped training, so it does not multiply training throughput. Sharing the
V100 still consumes GPU compute, accounting for the remaining 5–8% approximate
cost in these controlled windows. RAM snapshots remove most of the separate
training-side snapshot pause.

Serius's existing script was backed up, and only the reuse configuration was
added. Training resumed from update 4,121 as run
`run-1b4d9a9a8eb54e129b8865ef57aa3c87`, retaining its optimizer, scheduler, scaler,
data cursor, WSD recipe and both DDP GPUs. Validation remains asynchronous on
V100 GPU 1, every 100 updates, with 20 batches. Raw measurements and isolated
probes are under `data/performance-serius/validation-reuse/` (ignored).

Live verification of the resumed production run completed evaluations at updates
4,200 and 4,300 in the same V100 worker (PID 2332514). Their losses were 3.420539
and 3.406250. The warm update-4,300 job loaded its RAM snapshot in 0.118 seconds
and evaluated in 10.766 seconds; creating the snapshot paused the training rank
for 0.189 seconds. The first live snapshot also took 0.189 seconds. No error
events were emitted. Playwright verified the V100/async/100-update controls on
mobile and the desktop metric display without browser errors.
