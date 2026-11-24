"""Optional persistent document/token storage; readers and training policy stay in Python."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .cache_identity import token_cache_key

_progress_sink = ContextVar("weave_cache_progress", default=None)


@contextmanager
def cache_progress_reporting(callback):
    """Trainer attaches a reporting sink automatically, including external runs."""
    token = _progress_sink.set(callback)
    try:
        yield
    finally:
        _progress_sink.reset(token)


@dataclass
class CachedDocument:
    """One document's IDs, optional aligned labels, and JSON source metadata."""

    tokens: Sequence[int] | np.ndarray
    labels: Sequence[int] | np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TokenCacheImport:
    """Adopt a trusted existing flat token file and N+1 document offsets.

    The caller verifies that these IDs match sources/tokenizer/policy. Imported
    token files must remain immutable; on one filesystem a hard link avoids a
    second corpus copy. Other filesystems require copying the token file.
    """

    token_file: str | Path
    offsets: Sequence[int] | np.ndarray
    dtype: str = "<u2"


class TokenCache:
    """Read-only, memory-mapped documents. Safe to reopen in DataLoader workers."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        try:
            self.manifest = json.loads((self.directory / "manifest.json").read_text())
            meta = self.manifest
            if meta.get("format") != 1 or not meta.get("complete"):
                raise ValueError("missing completion marker or unsupported format")
            if meta["dtype"] not in ("<u2", "<u4"):
                raise ValueError("unsupported token dtype")
            expected = {
                "tokens.bin": meta["tokens"] * np.dtype(meta["dtype"]).itemsize,
                "offsets.bin": (meta["documents"] + 1) * 8,
                "metadata-offsets.bin": (meta["documents"] + 1) * 8,
                "metadata.jsonl": meta["metadata_bytes"],
            }
            if meta["labels"]:
                expected["labels.bin"] = meta["tokens"] * 8
            if meta["documents"] < 1 or meta["tokens"] < 1:
                raise ValueError("empty cache")
            for name, size in expected.items():
                if (self.directory / name).stat().st_size != size:
                    raise ValueError(f"{name} has an unexpected length")
            for name, end in (
                ("offsets.bin", meta["tokens"]),
                ("metadata-offsets.bin", meta["metadata_bytes"]),
            ):
                with (self.directory / name).open("rb") as source:
                    first = struct.unpack("<Q", source.read(8))[0]
                    source.seek(-8, 2)
                    last = struct.unpack("<Q", source.read(8))[0]
                if first != 0 or last != end:
                    raise ValueError(f"invalid bounds in {name}")
        except (OSError, KeyError, TypeError, ValueError, struct.error) as exc:
            raise ValueError(
                f"Invalid/incomplete token cache at {self.directory}: {exc}. Keep a backup and remove that cache directory before rebuilding."
            ) from exc
        self.source_files = [
            *meta["sources"],
            *(str(self.directory / name) for name in expected),
            str(self.directory / "manifest.json"),
        ]
        self._maps = {}

    def __len__(self):
        return self.manifest["documents"]

    def _map(self, name, dtype):
        if name not in self._maps:
            self._maps[name] = np.memmap(self.directory / name, dtype=dtype, mode="r")
        return self._maps[name]

    @property
    def tokens(self):
        return self._map("tokens.bin", self.manifest["dtype"])

    @property
    def offsets(self):
        return self._map("offsets.bin", "<u8")

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin, end = map(int, self.offsets[index : index + 2])
        if not 0 <= begin <= end <= self.manifest["tokens"]:
            raise ValueError("Corrupt token-cache document offsets")
        meta_offsets = self._map("metadata-offsets.bin", "<u8")
        a, b = map(int, meta_offsets[index : index + 2])
        if not 0 <= a <= b <= self.manifest["metadata_bytes"]:
            raise ValueError("Corrupt token-cache metadata offsets")
        metadata = json.loads(self._map("metadata.jsonl", "u1")[a:b].tobytes())
        labels = (
            self._map("labels.bin", "<i8")[begin:end]
            if self.manifest["labels"]
            else None
        )
        return CachedDocument(self.tokens[begin:end], labels, metadata)

    def __getstate__(self):
        return {**self.__dict__, "_maps": {}}


from .token_cache_writer import _write_documents, _write_existing


@contextmanager
def _builder_lock(path, timeout, report):
    # POSIX locks disappear when a worker exits, including after a reboot.
    import fcntl

    with path.open("a") as lock:
        started, last_report = time.monotonic(), None
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if last_report is None or time.monotonic() - last_report >= 5:
                    report({"phase": "waiting", "documents": 0, "tokens": 0})
                    last_report = time.monotonic()
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(
                        f"Another process is building this cache: {path}"
                    )
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _publish_progress(path, progress):
    descriptor, temporary = tempfile.mkstemp(prefix=".progress-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(progress, output)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _builder_progress(root, key, dtype):
    """Read shared progress, including approximate counts from older builders."""
    path = root / f".{key}.progress.json"
    try:
        progress = json.loads(path.read_text())
        if progress.get("key") == key and time.time() - progress.get("updated", 0) < 30:
            return {
                **{
                    name: progress[name]
                    for name in ("documents", "tokens", "total_documents")
                    if name in progress
                },
                "builder_phase": progress.get("phase"),
            }
    except (OSError, ValueError, TypeError):
        pass
    candidates = []
    for temporary in root.glob(f".{key}.*.tmp"):
        try:
            tokens = (temporary / "tokens.bin").stat()
            offsets = (temporary / "offsets.bin").stat()
            candidates.append(
                (
                    tokens.st_mtime_ns,
                    max(0, offsets.st_size // 8 - 1),
                    tokens.st_size // np.dtype(dtype).itemsize,
                )
            )
        except OSError:
            continue
    if candidates:
        _, documents, tokens = max(candidates)
        return {"documents": documents, "tokens": tokens, "progress_estimated": True}
    return {}


def cache_documents(
    directory,
    *,
    sources,
    tokenizer,
    read_documents: Callable[[], Iterable],
    policy: Mapping,
    encode: Callable[[object], CachedDocument] | None = None,
    eos_token_id: int | None = None,
    batch_size: int = 512,
    on_progress: Callable[[dict], None] | None = None,
    lock_timeout: float = 3600,
    total_documents: int | None = None,
    existing: TokenCacheImport | None = None,
):
    """Build once, or open a completed cache without calling read_documents.

    Default reader output is text; texts are batch-tokenized without implicit
    special tokens. Optional EOS appending is explicit. For arbitrary records,
    pass encode(record) -> CachedDocument (IDs, optional labels and metadata).
    Custom encoding owns special tokens and cannot use eos_token_id here.

    policy must identify reader/preprocessing revisions (including chat template
    and fixed augmentation settings). It is intentionally supplied by the recipe.
    Random augmentation, packing, splits and batch sampling belong after caching.
    """
    if type(batch_size) is not int or batch_size < 1 or lock_timeout < 0:
        raise ValueError("batch_size must be positive and lock_timeout nonnegative")
    if total_documents is not None and (
        type(total_documents) is not int or total_documents < 1
    ):
        raise ValueError("total_documents must be a positive integer or None")
    if not policy:
        raise ValueError(
            "Supply a preprocessing policy/version to identify your reader and encoding"
        )
    if encode is not None and eos_token_id is not None:
        raise ValueError("Custom encode owns EOS placement; omit eos_token_id")
    sources = [Path(path).expanduser().resolve() for path in sources]
    maximum = max(tokenizer.get_vocab().values(), default=-1)
    if eos_token_id is not None and (
        type(eos_token_id) is not int or not 0 <= eos_token_id <= maximum
    ):
        raise ValueError("EOS must be an existing tokenizer ID")
    if not 0 <= maximum <= np.iinfo(np.uint32).max:
        raise ValueError("Tokenizer IDs must fit in uint32")
    dtype = "<u2" if maximum <= np.iinfo(np.uint16).max else "<u4"
    identity = {
        "cache_format": 1,
        "encoding": "custom" if encode else "text",
        "eos_token_id": eos_token_id,
        "dtype": dtype,
        "recipe": dict(policy),
    }
    key = token_cache_key(sources, tokenizer, policy=identity)
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key

    def report(progress):
        if progress["phase"] == "waiting":
            progress = {**progress, **_builder_progress(root, key, dtype)}
            progress["message"] = (
                "Another process is preparing this cache. Showing its progress."
                + (
                    " Counts are approximate while files are buffered."
                    if progress.get("progress_estimated")
                    else ""
                )
            )
        progress = {
            **progress,
            "key": key,
            "directory": str(destination),
            "total_documents": progress.get("total_documents") or total_documents,
        }
        if progress["phase"] in ("building", "importing", "complete", "failed"):
            _publish_progress(
                root / f".{key}.progress.json", {**progress, "updated": time.time()}
            )
        sink = _progress_sink.get()
        if sink:
            sink(progress)
        if on_progress:
            on_progress(progress)
        else:
            print(
                f"Token cache {key}: {progress['phase']} · {progress['documents']:,} documents · {progress['tokens']:,} tokens",
                flush=True,
            )

    def open_cache():
        cache = TokenCache(destination)
        if cache.manifest["key"] != key:
            raise ValueError("Token cache has a different identity")
        report(
            {
                "phase": "reused",
                "documents": len(cache),
                "tokens": cache.manifest["tokens"],
            }
        )
        return cache

    if destination.exists():
        return open_cache()
    with _builder_lock(root / f".{key}.lock", lock_timeout, report):
        if destination.exists():
            return open_cache()
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", suffix=".tmp", dir=root))
        try:
            report({"phase": "building", "documents": 0, "tokens": 0})
            manifest = (
                _write_existing(temporary, existing, dtype, maximum, report)
                if existing
                else _write_documents(
                    temporary,
                    read_documents,
                    encode,
                    tokenizer,
                    eos_token_id,
                    batch_size,
                    dtype,
                    maximum,
                    report,
                )
            )
            if token_cache_key(sources, tokenizer, policy=identity) != key:
                raise ValueError(
                    "Sources or tokenizer changed during cache preparation; cache was not published"
                )
            manifest.update(
                format=1,
                complete=True,
                key=key,
                policy=identity,
                sources=[str(path) for path in sources],
            )
            with (temporary / "manifest.json").open("w") as target:
                json.dump(manifest, target, allow_nan=False, indent=2)
                target.flush()
                os.fsync(target.fileno())
            TokenCache(temporary)  # Validate before making it visible to readers.
            os.replace(temporary, destination)
            report(
                {
                    "phase": "complete",
                    "documents": manifest["documents"],
                    "tokens": manifest["tokens"],
                }
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return TokenCache(destination)
