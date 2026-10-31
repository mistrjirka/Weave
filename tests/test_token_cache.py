"""Real tokenizer/cache I/O; custom records, reuse, failure and reader independence."""

from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from backend.training.token_cache import cache_documents, CachedDocument, TokenCache


def tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2, "end": 3}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="end"
    )


def setup_cache(tmp_path, **kwargs):
    source = tmp_path / "source.jsonl"
    if not source.exists():
        source.write_text("reader-defined content")
    defaults = dict(
        sources=[source],
        tokenizer=tokenizer(),
        read_documents=lambda: iter(["hello world", "hello"]),
        policy={"version": "test-v1"},
    )
    defaults.update(kwargs)
    return cache_documents(tmp_path / "cache", **defaults)


def test_cache_preserves_document_boundaries_and_reuses_without_reading(tmp_path):
    progress = []
    cache = setup_cache(
        tmp_path, eos_token_id=3, batch_size=1, on_progress=progress.append
    )
    assert len(cache) == 2 and cache.manifest["tokens"] == 5
    assert list(cache[0].tokens) == [1, 2, 3] and list(cache[-1].tokens) == [1, 3]
    assert cache[0].labels is None and cache[0].metadata == {}
    assert isinstance(cache.tokens, np.memmap) and not cache.tokens.flags.writeable
    assert [p["phase"] for p in progress] == ["building", "complete"]

    def must_not_run():
        raise AssertionError("Cache hit called source reader")

    tok = tokenizer()
    tok.name_or_path = "/a/different/frozen/run"
    again = setup_cache(
        tmp_path,
        eos_token_id=3,
        read_documents=must_not_run,
        tokenizer=tok,
        on_progress=progress.append,
    )
    assert again.directory == cache.directory
    assert progress[-1]["phase"] == "reused"
    reopened = pickle.loads(pickle.dumps(cache))
    assert reopened._maps == {} and list(reopened[1].tokens) == [1, 3]
    with pytest.raises(IndexError):
        cache[2]
    assert all(Path(p).is_file() for p in cache.source_files)


def test_custom_records_labels_and_metadata_and_policy_versions(tmp_path):
    rows = [
        {"ids": [1, 2, 3], "labels": [-100, 2, 3], "file": "one.parquet", "row": 7},
        {"ids": [2, 1], "labels": [1, -100], "file": "two.jsonl", "row": 2},
    ]

    def encode(row):
        return CachedDocument(
            row["ids"], row["labels"], {"file": row["file"], "row": row["row"]}
        )

    cache = setup_cache(tmp_path, read_documents=lambda: iter(rows), encode=encode)
    assert list(cache[0].labels) == [-100, 2, 3]
    assert cache[1].metadata == {"file": "two.jsonl", "row": 2}
    changed = setup_cache(
        tmp_path,
        read_documents=lambda: iter(rows),
        encode=encode,
        policy={"version": "new-label-policy"},
    )
    assert changed.directory != cache.directory
    with pytest.raises(ValueError, match="owns EOS"):
        setup_cache(
            tmp_path, read_documents=lambda: iter(rows), encode=encode, eos_token_id=3
        )


def test_interrupted_builder_does_not_publish_or_leave_reusable_partial_cache(tmp_path):
    def broken():
        yield "hello"
        raise RuntimeError("reader interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        setup_cache(tmp_path, read_documents=broken, batch_size=1)
    assert not list((tmp_path / "cache").glob("*/manifest.json"))
    assert not list((tmp_path / "cache").glob("*.tmp"))
    cache = setup_cache(tmp_path)
    assert len(cache) == 2
    with (cache.directory / "tokens.bin").open("wb") as target:
        target.write(b"x")
    with pytest.raises(ValueError, match="unexpected length"):
        setup_cache(tmp_path)


def test_sources_changing_midbuild_are_not_published(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("original")

    def changed():
        yield "hello"
        source.write_text("a different source")

    with pytest.raises(ValueError, match="changed during"):
        setup_cache(tmp_path, read_documents=changed, batch_size=1)
    assert not list((tmp_path / "cache").glob("*/manifest.json"))


def test_concurrent_builders_share_one_published_cache(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("same data")
    readers = []
    entered = threading.Event()
    release = threading.Event()

    def read():
        readers.append(1)
        entered.set()
        assert release.wait(5)
        yield "hello world"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(setup_cache, tmp_path, read_documents=read)
        assert entered.wait(5)
        second = executor.submit(setup_cache, tmp_path, read_documents=read)
        time.sleep(0.1)
        release.set()
        assert first.result(timeout=5).directory == second.result(timeout=5).directory
    assert len(readers) == 1


@pytest.mark.parametrize(
    "document",
    [
        CachedDocument([1, 2], [-100]),
        CachedDocument([1.5]),
        CachedDocument([1, 999]),
        CachedDocument([1], [-3]),
    ],
)
def test_custom_encoding_rejects_invalid_arrays(tmp_path, document):
    with pytest.raises(ValueError):
        setup_cache(
            tmp_path, read_documents=lambda: iter([document]), encode=lambda row: row
        )
    assert not list((tmp_path / "cache").glob("*/manifest.json"))


def test_large_added_token_ids_use_uint32_and_progress_has_default_logs(
    tmp_path, capsys
):
    tok = tokenizer()
    # A tokenizer may have sparse/high IDs; selecting dtype by len(vocab) truncates them.
    vocab = tok.get_vocab()
    vocab["large"] = 70000
    tok.get_vocab = lambda: vocab
    cache = setup_cache(
        tmp_path,
        tokenizer=tok,
        read_documents=lambda: iter([70000]),
        encode=lambda row: CachedDocument([row]),
    )
    assert cache.manifest["dtype"] == "<u4" and int(cache[0].tokens[0]) == 70000
    assert "building" in capsys.readouterr().out


def test_import_existing_tokens_without_encoding(tmp_path):
    from backend.training.token_cache import TokenCacheImport

    original = tmp_path / "existing.u16"
    np.asarray([1, 2, 3, 2, 3], dtype="<u2").tofile(original)

    def forbidden():
        raise AssertionError("Reader must not run when adopting existing tokens")

    cache = setup_cache(
        tmp_path,
        read_documents=forbidden,
        existing=TokenCacheImport(original, [0, 3, 5]),
    )
    assert cache[0].tokens.tolist() == [1, 2, 3]
    assert cache[1].tokens.tolist() == [2, 3]
    assert (cache.directory / "tokens.bin").stat().st_ino == original.stat().st_ino
    with pytest.raises(ValueError, match="monotonic"):
        setup_cache(
            tmp_path,
            policy={"changed": 1},
            existing=TokenCacheImport(original, [0, 5, 3]),
        )


def test_automatic_progress_sink(tmp_path):
    from backend.training.token_cache import cache_progress_reporting

    events = []
    with cache_progress_reporting(events.append):
        setup_cache(tmp_path)
        setup_cache(tmp_path)
    assert events[0]["phase"] == "building"
    assert events[-1]["phase"] == "reused"
    assert events[-1]["documents"] > 0
