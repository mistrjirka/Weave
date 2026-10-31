"""Training stage. Run with: python train.py"""

import torch
from torch.nn import functional as F

from weave_support.load_tokenizer import load_tokenizer
from weave_support.model import Model
from weave_support.trainer import Trainer, TrainingConfig, StepResult

config = TrainingConfig(
    devices=[],  # CPU; [0] for GPU 0, [0, 1] for DDP.
    precision="fp32",
    micro_batch_size=16,
    gradient_accumulation_steps=2,
    max_epochs=3,
    max_optimizer_steps=None,  # Optional update limit; the first limit reached stops training.
    learning_rate=3e-4,  # Starting learning rate.
    min_learning_rate=3e-5,
    weight_decay=0.01,
    gradient_clip=1.0,
    log_every=1,
    checkpoint_every=250,
    eval_every=100,
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
    if config.total_optimizer_steps is None:
        raise ValueError(
            "Cosine decay needs len(train_loader) or max_optimizer_steps. "
            "For an unsized stream, set an update limit or supply a different scheduler."
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.total_optimizer_steps,
        eta_min=config.min_learning_rate,
    )


# Example objective: next-token prediction.
# The loader supplies input_ids and already-shifted labels.
def training_step(model, batch, device):
    logits = model(batch["input_ids"].to(device))
    labels = batch["labels"].to(logits.device)
    count = int((labels != -100).sum())
    if not count:
        raise ValueError("Batch has no supervised targets.")
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
