"""Training stage. Run with: python train.py"""

import math

import torch
from torch.nn import functional as F

from weave_support.load_tokenizer import load_tokenizer
from weave_support.model import Model
from weave_support.trainer import Trainer, TrainingConfig, StepResult

# WSD: warm up, hold the peak rate, then decay over the final part of training.
WARMUP_FRACTION = 0.02
WSD_DECAY_FRACTION = 0.20

config = TrainingConfig(
    devices=[],  # CPU; [0] for GPU 0, [0, 1] for DDP.
    precision="fp32",
    compile_mode="off",  # Try "default" or "reduce-overhead" after benchmarking.
    fused_optimizer=False,  # CUDA AdamW only; preserves your optimizer groups.
    profile_steps=0,  # Capture this many updates (1–100) after two warm-up updates.
    micro_batch_size=16,
    gradient_accumulation_steps=2,
    max_epochs=3,
    max_optimizer_steps=None,  # Optional update limit; the first limit reached stops training.
    learning_rate=3e-4,  # Peak rate, reached after warmup.
    min_learning_rate=3e-5,
    weight_decay=0.01,
    gradient_clip=1.0,
    log_every=1,
    checkpoint_every=250,
    eval_every=50,
    async_evaluation=False,  # True: evaluate checkpoints in a separate process.
    reuse_evaluation_worker=True,  # Keep model/tokenizer warm; False for stateful custom hooks.
    evaluation_device="cpu",  # "cpu" or a GPU index, e.g. 1.
    sample_prompts=[],
)


# Data: implement your own loading, tokenization and sequence lengths.
def get_dataloaders(tokenizer, config: TrainingConfig):
    """Return (train_loader, optional_validation_loader)."""
    raise NotImplementedError("Add your data loaders here.")


def configure_optimizer(model, config: TrainingConfig):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def configure_scheduler(optimizer, config: TrainingConfig):
    # Trainer calculates this from loader length, epochs and the update limit.
    total = config.total_optimizer_steps
    if total is None or total < 1:
        raise ValueError(
            "WSD needs len(train_loader) or max_optimizer_steps. "
            "For an unsized stream, set an update limit or supply a different scheduler."
        )
    if not (
        0 <= WARMUP_FRACTION < 1
        and 0 < WSD_DECAY_FRACTION < 1
        and WARMUP_FRACTION + WSD_DECAY_FRACTION < 1
    ):
        raise ValueError("WSD warmup and decay fractions must leave a stable phase.")
    warmup = min(total - 1, math.ceil(total * WARMUP_FRACTION))
    decay = min(total - warmup, max(1, math.ceil(total * WSD_DECAY_FRACTION)))
    decay_start = total - decay
    floor = config.min_learning_rate / config.learning_rate

    def lr(step):
        update = step + 1  # LambdaLR(0) sets the first update's rate.
        if update <= warmup:
            return update / warmup
        if update <= decay_start:
            return 1.0
        progress = min(1.0, (update - decay_start) / decay)
        return floor + (1.0 - floor) * (1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr)


# Example objective: next-token prediction.
# The loader supplies input_ids and already-shifted labels.
def training_step(model, batch, device):
    # Count on the CPU before transfer; no GPU synchronization for this scalar.
    count = int((batch["labels"] != -100).sum())
    if not count:
        raise ValueError("Batch has no supervised targets.")
    logits = model(batch["input_ids"].to(device))
    labels = batch["labels"].to(logits.device)
    loss = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
    )
    return StepResult(loss, weight=count)


if __name__ == "__main__":
    Trainer(
        model=Model,
        config=config,
        tokenizer=load_tokenizer,  # Use None for pretokenized data without a tokenizer.
        dataloaders_fn=get_dataloaders,
        optimizer_fn=configure_optimizer,
        scheduler_fn=configure_scheduler,
        training_step_fn=training_step,
    ).train()
