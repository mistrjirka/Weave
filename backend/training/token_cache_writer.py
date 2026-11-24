"""Write and validate token-cache document storage."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import shutil
import struct
import time

import numpy as np

def _write_existing(directory, existing, dtype, maximum, report):
    if np.dtype(existing.dtype) != np.dtype(dtype):
        raise ValueError(
            "Imported token dtype must match the tokenizer's uint16/uint32 storage"
        )
    source = Path(existing.token_file).resolve()
    offsets = np.asarray(existing.offsets)
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(
            "Imported offsets must be monotonic integer document bounds starting at zero"
        )
    count, documents = int(offsets[-1]), len(offsets) - 1
    if count < 1 or source.stat().st_size != count * np.dtype(dtype).itemsize:
        raise ValueError("Imported offsets do not cover the whole nonempty token file")
    report({"phase": "importing", "documents": 0, "tokens": 0})
    tokens = np.memmap(source, dtype=dtype, mode="r")
    for begin in range(0, count, 16_000_000):
        if np.any(tokens[begin : begin + 16_000_000] > maximum):
            raise ValueError(
                "Imported token file contains IDs outside the tokenizer vocabulary"
            )
    del tokens
    try:
        os.link(source, directory / "tokens.bin")
    except OSError:
        shutil.copyfile(source, directory / "tokens.bin")
    offsets.astype("<u8").tofile(directory / "offsets.bin")
    (np.arange(documents + 1, dtype="<u8") * 3).tofile(
        directory / "metadata-offsets.bin"
    )
    with (directory / "metadata.jsonl").open("wb") as target:
        for start in range(0, documents, 100_000):
            target.write(b"{}\n" * min(100_000, documents - start))
    for path in directory.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    return {
        "documents": documents,
        "tokens": count,
        "dtype": dtype,
        "labels": False,
        "metadata_bytes": documents * 3,
    }


def _write_documents(
    directory,
    read_documents,
    encode,
    tokenizer,
    eos,
    batch_size,
    dtype,
    maximum,
    report,
):
    from contextlib import ExitStack

    from .token_cache import CachedDocument

    documents, count, metadata_bytes, has_labels = 0, 0, 0, None
    last_report = time.monotonic()
    with ExitStack() as stack:
        files = {
            name: stack.enter_context((directory / name).open("wb"))
            for name in (
                "tokens.bin",
                "labels.bin",
                "offsets.bin",
                "metadata-offsets.bin",
                "metadata.jsonl",
            )
        }
        files["offsets.bin"].write(struct.pack("<Q", 0))
        files["metadata-offsets.bin"].write(struct.pack("<Q", 0))
        reader = iter(read_documents())
        while batch := list(itertools.islice(reader, batch_size)):
            if encode is None:
                if not all(isinstance(text, str) for text in batch):
                    raise TypeError(
                        "Default read_documents must yield strings; supply encode for other records"
                    )
                ids = tokenizer(
                    batch,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                if len(ids) != len(batch):
                    raise ValueError(
                        "Tokenizer must return one ID sequence per document"
                    )
                encoded = [
                    CachedDocument([*tokens, eos] if eos is not None else tokens)
                    for tokens in ids
                ]
            else:
                encoded = map(encode, batch)
            for document in encoded:
                if not isinstance(document, CachedDocument):
                    raise TypeError("encode must return CachedDocument")
                tokens = np.asarray(document.tokens)
                if tokens.ndim != 1 or (
                    tokens.size
                    and (
                        tokens.dtype.kind not in "iu"
                        or np.any(tokens < 0)
                        or np.any(tokens > maximum)
                    )
                ):
                    raise ValueError(
                        "Document tokens must be one-dimensional integer tokenizer IDs"
                    )
                labeled = document.labels is not None
                if has_labels is None:
                    has_labels = labeled
                if labeled != has_labels:
                    raise ValueError(
                        "Labels must be supplied for every document or for none"
                    )
                tokens.astype(dtype).tofile(files["tokens.bin"])
                if labeled:
                    labels = np.asarray(document.labels)
                    if labels.shape != tokens.shape or (
                        labels.size
                        and (
                            labels.dtype.kind not in "iu"
                            or np.any(
                                (labels != -100) & ((labels < 0) | (labels > maximum))
                            )
                        )
                    ):
                        raise ValueError(
                            "Labels must align with token IDs and contain tokenizer IDs or -100"
                        )
                    labels.astype("<i8").tofile(files["labels.bin"])
                if not isinstance(document.metadata, dict):
                    raise TypeError("Document metadata must be a JSON dictionary")
                metadata = (
                    json.dumps(document.metadata, allow_nan=False, ensure_ascii=False)
                    + "\n"
                ).encode()
                files["metadata.jsonl"].write(metadata)
                documents += 1
                count += tokens.size
                metadata_bytes += len(metadata)
                files["offsets.bin"].write(struct.pack("<Q", count))
                files["metadata-offsets.bin"].write(struct.pack("<Q", metadata_bytes))
            if time.monotonic() - last_report >= 5:
                report({"phase": "building", "documents": documents, "tokens": count})
                last_report = time.monotonic()
        if not documents or not count:
            raise ValueError("The reader produced no tokenized documents")
        for target in files.values():
            target.flush()
            os.fsync(target.fileno())
    if not has_labels:
        (directory / "labels.bin").unlink()
    return {
        "documents": documents,
        "tokens": int(count),
        "dtype": dtype,
        "labels": bool(has_labels),
        "metadata_bytes": metadata_bytes,
    }
