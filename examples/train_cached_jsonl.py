"""Optional teaching recipe; copy into a Weave stage. Default train.py stays small.

TRAINING_DATA=/absolute/path/to/text.jsonl python train.py
Each JSONL record has {"text": "..."}. Documents remain separate. This example
uses whole documents/right-padding; use your own chunking/bucketing for long texts.
This simple map-style loader does not implement exact cursor resume or DDP sharding.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from weave_support.load_tokenizer import load_tokenizer
from weave_support.model import Model
from weave_support.token_cache import cache_documents
from weave_support.trainer import Trainer, TrainingConfig, StepResult

config = TrainingConfig(
    devices=[],
    precision="fp32",
    micro_batch_size=2,
    max_epochs=None,
    max_optimizer_steps=10,
)


class Documents(Dataset):
    def __init__(self, cache):
        self.cache = cache
        self.source_files = cache.source_files

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        document = self.cache[index]
        if len(document.tokens) < 2:
            raise ValueError("Filter documents with fewer than two cached tokens")
        ids = torch.from_numpy(np.array(document.tokens, dtype=np.int64))
        return {"input_ids": ids[:-1], "labels": ids[1:], "source": {"document": index}}


def get_dataloaders(tokenizer, config):
    if config.resolved_strategy != "single":
        raise ValueError(
            "This example needs your own sharding before enabling DDP/split"
        )
    source = Path(os.environ["TRAINING_DATA"]).expanduser().resolve()
    if tokenizer.eos_token_id is None:
        raise ValueError("Configure the correct existing EOS token for this tokenizer")

    def read_documents():
        with source.open() as stream:
            for line in stream:
                yield json.loads(line)["text"]

    cache = cache_documents(
        Path(
            os.environ.get(
                "TOKEN_CACHE_DIR", str(Path.home() / "shared/weave-token-cache")
            )
        ),
        sources=[source],
        tokenizer=tokenizer,
        read_documents=read_documents,
        policy={"reader": "jsonl-text-v1"},
        eos_token_id=tokenizer.eos_token_id,
    )
    dataset = Documents(cache)

    def collate(rows):
        length = max(len(row["input_ids"]) for row in rows)
        inputs = torch.full(
            (len(rows), length), tokenizer.eos_token_id, dtype=torch.long
        )
        labels = torch.full_like(inputs, -100)
        for index, row in enumerate(rows):
            n = len(row["input_ids"])
            inputs[index, :n] = row["input_ids"]
            labels[index, :n] = row["labels"]
        return {
            "input_ids": inputs,
            "labels": labels,
            "_weave_source": json.dumps([row["source"] for row in rows]),
        }

    # Supply a real held-out loader for validation; never reuse training data for it.
    return (
        DataLoader(dataset, batch_size=config.micro_batch_size, collate_fn=collate),
        None,
    )


def configure_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def configure_scheduler(optimizer, config):
    if config.total_optimizer_steps is None:
        raise ValueError("Cosine decay requires a known optimizer-update budget")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_optimizer_steps, eta_min=config.min_learning_rate
    )


def training_step(model, batch, device):
    logits = model(batch["input_ids"].to(device))
    labels = batch["labels"].to(logits.device)
    count = int((labels != -100).sum())
    if count == 0:
        raise ValueError("No supervised targets")
    loss = F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    return StepResult(loss, weight=count)


if __name__ == "__main__":
    Trainer(
        model=Model,
        config=config,
        tokenizer=load_tokenizer,
        dataloaders_fn=get_dataloaders,
        optimizer_fn=configure_optimizer,
        scheduler_fn=configure_scheduler,
        training_step_fn=training_step,
    ).train()
