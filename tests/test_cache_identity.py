"""Token caches survive export relocation but invalidate when encoding/data changes."""

from pathlib import Path
from types import SimpleNamespace
import json
import shutil

from tokenizers import Tokenizer, models, pre_tokenizers

from backend.training.cache_identity import token_cache_key, tokenizer_fingerprint
from backend.training.layout import write_helpers


def tokenizer(path, vocab=None):
    backend = Tokenizer(
        models.WordLevel(
            vocab or {"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return SimpleNamespace(
        backend_tokenizer=backend,
        name_or_path=str(path),
        special_tokens_map={"unk_token": "[UNK]", "eos_token": "world"},
        eos_token_id=2,
    )


def test_cache_identity_ignores_export_folder_and_tracks_tokenizer_contents(tmp_path):
    source = tmp_path / "corpus.parquet"
    source.write_bytes(b"unchanged corpus")
    first = tokenizer(tmp_path / "run-1/tokenizer")
    second = tokenizer(tmp_path / "run-2/tokenizer")
    policy = {"format": "vsl-v2", "buckets": [64, 128], "validation_every": 100}
    assert first.name_or_path != second.name_or_path
    expected = token_cache_key([source], first, policy=policy)
    assert token_cache_key([source], second, policy=policy) == expected
    changed = tokenizer(
        tmp_path / "run-3/tokenizer", {"[UNK]": 0, "hello": 2, "world": 1}
    )
    assert token_cache_key([source], changed, policy=policy) != expected
    second.eos_token_id = 1
    assert token_cache_key([source], second, policy=policy) != expected
    second.eos_token_id = 2
    second.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    assert token_cache_key([source], second, policy=policy) != expected
    assert (
        token_cache_key([source], first, policy={**policy, "validation_every": 50})
        != expected
    )
    source.write_bytes(b"changed corpus")
    assert token_cache_key([source], first, policy=policy) != expected


def test_cache_identity_tracks_ordered_sources_and_paths_without_hashing_corpus(
    tmp_path,
):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    tok = tokenizer("anywhere")
    key = token_cache_key([first, second], tok, policy={})
    assert token_cache_key([second, first], tok, policy={}) != key
    another = tmp_path / "same-name/a"
    another.parent.mkdir()
    shutil.copy2(first, another)
    assert (
        another.stat().st_size == first.stat().st_size
        and another.stat().st_mtime_ns == first.stat().st_mtime_ns
    )
    assert token_cache_key([another, second], tok, policy={}) != key
    # Tokenizer hashes are independent of the process and serialization key order.
    payload = json.loads(tok.backend_tokenizer.to_str())
    tok.backend_tokenizer = SimpleNamespace(
        to_str=lambda: json.dumps(payload, sort_keys=True)
    )
    assert token_cache_key([first, second], tok, policy={}) == key


def test_cache_helper_is_included_in_generated_portable_support(tmp_path):
    generated = write_helpers(tmp_path / "weave_support")
    assert "weave_support/cache_identity.py" in generated
    assert (
        "def token_cache_key("
        in (tmp_path / "weave_support/cache_identity.py").read_text()
    )
