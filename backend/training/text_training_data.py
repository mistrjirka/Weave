"""Text dataset preparation, batching and loss."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from weave_training import atomic_json, file_hash


class TokenDataset:
    """Disk-backed prepared records; only a small working set lives in RAM."""

    def __init__(self, path, offsets):
        from collections import OrderedDict

        self.path, self.offsets = Path(path), offsets
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index not in self.cache:
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[index])
                self.cache[index] = json.loads(stream.readline())
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
        return self.cache[index]


def load_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        for batch_rows in parquet.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch_rows.to_pylist()
    elif path.suffix == ".json":
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError(
                "Use JSONL or Parquet for JSON datasets over 64 MB to keep preparation memory bounded."
            )
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("JSON data must contain an array of records.")
        yield from rows
    else:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield (
                        {"text": line.rstrip()}
                        if path.suffix == ".txt"
                        else json.loads(line)
                    )


def prepare_data(config, directory, tokenizer, *, evaluation=False):
    source = config.get("eval_dataset" if evaluation else "dataset")
    if not source:
        return []
    expected = config.get("dataset_sha256" if not evaluation else "eval_dataset_sha256")
    before = file_hash(source)
    if expected and before != expected:
        raise ValueError("Dataset contents changed since this run was created.")
    rows = load_rows(source)
    suffix = "eval" if evaluation else "train"
    target = directory / f"prepared-{suffix}.jsonl"
    offsets = []
    output = target.with_suffix(".pending").open("wb")
    for index, row in enumerate(rows):
        if "input_ids" in row:
            ids = row["input_ids"]
            labels = row.get("labels", ids)
        elif config["recipe"] == "sft":
            if tokenizer is None:
                raise ValueError(
                    "SFT requires a tokenizer on the token input, or pretokenized input_ids and labels."
                )
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tools=row.get("tools"),
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
            )
            ids = encoded["input_ids"]
            mask = encoded.get("assistant_masks")
            if not mask or not any(mask):
                raise ValueError(
                    "Assistant-only SFT needs a chat template with generation/assistant masks. Supply a compatible template or explicit input_ids and labels (-100 for ignored tokens)."
                )
            labels = [token if active else -100 for token, active in zip(ids, mask)]
        else:
            if tokenizer is None:
                raise ValueError(
                    "Text records require a tokenizer selected on the model input."
                )
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            labels = list(ids)
        if len(ids) != len(labels):
            raise ValueError(
                f"Record {index}: input_ids and labels have different lengths."
            )
        if any(not isinstance(v, int) or v < 0 for v in ids):
            raise ValueError(f"Record {index}: token IDs must be nonnegative integers.")
        if any(
            type(value) is not int or (value < 0 and value != -100) for value in labels
        ):
            raise ValueError(
                f"Record {index}: labels must be token IDs or -100 for ignored targets."
            )
        capacity = config.get("vocabulary_size")
        if capacity and (
            max(ids, default=0) >= capacity or max(labels, default=0) >= capacity
        ):
            raise ValueError(
                f"Record {index}: token IDs exceed the vocabulary head size ({capacity}). Check the tokenizer and head configuration."
            )
        chunks = (
            range(0, max(1, len(ids) - 1), config["sequence_length"])
            if config["recipe"] == "causal"
            else [0]
        )
        for start in chunks:
            chunk_ids = list(ids[start : start + config["sequence_length"] + 1])
            chunk_labels = list(labels[start : start + config["sequence_length"] + 1])
            if len(chunk_ids) < 2:
                continue
            if not any(v != -100 for v in chunk_labels[1:]):
                raise ValueError(
                    f"Record {index}: no supervised next-token targets after truncation."
                )
            identity = hashlib.sha256(
                json.dumps(
                    {"input_ids": chunk_ids, "labels": chunk_labels}, sort_keys=True
                ).encode()
            ).hexdigest()
            record = {
                "input_ids": chunk_ids,
                "labels": chunk_labels,
                "record": len(offsets),
                "source_record": index,
                "chunk_start": start,
                "sha256": identity,
            }
            offsets.append(output.tell())
            output.write((json.dumps(record) + "\n").encode())
    output.flush()
    os.fsync(output.fileno())
    output.close()
    if file_hash(source) != before:
        raise ValueError("Dataset changed while it was being read.")
    if not offsets:
        raise ValueError(
            "Dataset is empty or contains no sequences with next-token targets."
        )
    os.replace(target.with_suffix(".pending"), target)
    atomic_json(directory / f"prepared-{suffix}-offsets.json", offsets)
    atomic_json(
        directory / f"data-{suffix}.json",
        {
            "path": str(Path(source).resolve()),
            "sha256": file_hash(source),
            "prepared_sha256": file_hash(target),
            "rows": len(offsets),
            "coverage": "source and prepared records verified; consumed record IDs recorded per optimizer step; deterministic sampling with replacement",
            "sequence_length": config["sequence_length"],
            "recipe": config["recipe"],
        },
    )
    return TokenDataset(target, offsets)


def batch(rows, device):
    width = max(len(row["input_ids"]) for row in rows) - 1
    inputs = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    targets = torch.full_like(inputs, -100)
    for i, row in enumerate(rows):
        length = len(row["input_ids"]) - 1
        inputs[i, :length] = torch.tensor(row["input_ids"][:-1], device=device)
        targets[i, :length] = torch.tensor(row["labels"][1:], device=device)
    return inputs, targets


def language_loss(logits, targets):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != targets.shape
    ):
        raise ValueError(
            "This recipe expects one floating-point [batch, sequence, vocabulary] output. Use a custom script for other contracts."
        )
    targets = targets.to(logits.device)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
