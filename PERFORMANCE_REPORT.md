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
