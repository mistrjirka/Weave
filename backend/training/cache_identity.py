"""Optional cache identities for recipes using fast tokenizers.

Tokenization behavior, data identity and the recipe policy define the cache.
Tokenizer export locations do not: frozen Weave runs have different directories.
"""

import hashlib
import json
from pathlib import Path


def tokenizer_fingerprint(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Token-cache fingerprints require a fast tokenizer backend")
    payload = {
        "backend": json.loads(backend.to_str()),
        "special_tokens": {
            key: (
                [str(token) for token in value]
                if isinstance(value, list)
                else str(value)
            )
            for key, value in tokenizer.special_tokens_map.items()
        },
        "eos_token_id": tokenizer.eos_token_id,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def token_cache_key(files, tokenizer, *, policy):
    """Fingerprint ordered input paths/stats, tokenizer semantics and caller policy.

    Source files use path/size/mtime to avoid rereading an entire corpus. Include
    chunking, splitting and a recipe-format version in policy. This is a cache
    invalidation key, not a cryptographic proof of dataset contents.
    """
    sources = []
    for value in files:
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        sources.append(
            {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    identity = {
        "schema": 1,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "sources": sources,
        "policy": policy,
    }
    return hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()[:16]
