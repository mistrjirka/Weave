"""Portable Weave reporting API. Uses only the Python standard library.

Training remains ordinary PyTorch. Events are durable local JSONL records; a
missing browser or server never blocks an optimizer step. Only rank zero writes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
import uuid

PROTOCOL_VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Run:
    def __init__(self, directory=None, *, rank=None):
        self.directory = Path(
            directory or os.environ.get("WEAVE_RUN_DIR", "weave-run")
        ).resolve()
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.tokens = 0
        self.commands_seen = set()
        self._stream = None

    def __enter__(self):
        if self.rank == 0:
            self._stream = (self.directory / "events.jsonl").open("a", buffering=1)
            environment_path = self.directory / "environment.json"
            if not environment_path.exists():
                import importlib.metadata
                import platform
                import sys

                packages = {
                    distribution.metadata["Name"]: distribution.version
                    for distribution in importlib.metadata.distributions()
                    if distribution.metadata["Name"]
                }
                atomic_json(
                    environment_path,
                    {
                        "python": sys.version,
                        "platform": platform.platform(),
                        "packages": packages,
                    },
                )
        self.emit("started", {"protocol": PROTOCOL_VERSION, "pid": os.getpid()})
        return self

    def __exit__(self, kind, exception, tb):
        if exception:
            self.emit(
                "error",
                {
                    "message": str(exception),
                    "traceback": "".join(
                        traceback.format_exception(kind, exception, tb)
                    ),
                },
            )
        self.emit("finished", {"success": exception is None})
        if self._stream:
            self._stream.close()
        return False

    def emit(self, kind, payload):
        if self.rank != 0:
            return
        record = {
            "version": PROTOCOL_VERSION,
            "id": uuid.uuid4().hex,
            "time": time.time(),
            "kind": kind,
            "step": self.step,
            "tokens": self.tokens,
            "payload": payload,
        }
        line = json.dumps(record, allow_nan=False)
        if len(line.encode()) > 1024 * 1024:
            raise ValueError(
                "A reporting event must be smaller than 1 MB; store large values as artifacts."
            )
        if self._stream is None:
            raise RuntimeError("Use reporting inside `with Run() as run:`")
        self._stream.write(line + "\n")
        self._stream.flush()

    def log(self, metrics, *, step, tokens=None):
        self.step = int(step)
        if tokens is not None:
            self.tokens = int(tokens)
        clean = {}
        for key, value in metrics.items():
            number = float(value)
            if math.isfinite(number):
                clean[str(key)] = number
            else:
                self.warning(f"Nonfinite metric: {key}={number}", code="nonfinite")
        self.emit("metrics", clean)

    def progress(self, *, completed, total=None, unit="steps"):
        self.emit("progress", {"completed": completed, "total": total, "unit": unit})

    def sample(self, *, prompt, output, checkpoint=None, metadata=None):
        self.emit(
            "sample",
            {
                "prompt": prompt,
                "output": output,
                "checkpoint": checkpoint,
                "metadata": metadata or {},
            },
        )

    def checkpoint(self, path, *, metadata=None):
        path = Path(path).resolve()
        relative = path.relative_to(self.directory)
        if not path.is_file():
            raise ValueError(
                "Publish a completed checkpoint file inside the run directory."
            )
        self.emit(
            "checkpoint",
            {
                "path": str(relative),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "metadata": metadata or {},
            },
        )

    def warning(self, message, *, code="custom", **details):
        self.emit("warning", {"message": message, "code": code, **details})

    def poll_commands(self):
        """Poll only at safe optimizer-step boundaries. Never interrupt backward."""
        commands = []
        for path in sorted((self.directory / "commands").glob("*.json")):
            if path.name in self.commands_seen:
                continue
            command = json.loads(path.read_text())
            self.commands_seen.add(path.name)
            commands.append(command)
        return commands

    def capabilities(self, **capabilities):
        self.emit("capabilities", capabilities)


def instrumentation_warnings(source):
    """Advisory syntax inspection, not a claim that callbacks actually execute."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            {
                "code": "syntax",
                "message": f"Python syntax error at line {exc.lineno}: {exc.msg}",
            }
        ]
    methods = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    expected = {
        "log": "No explicit metric logging found.",
        "sample": "No generated-sample reporting found.",
        "checkpoint": "No checkpoint reporting found.",
        "poll_commands": "No cooperative stop/snapshot polling found.",
    }
    return [
        {"code": "missing-" + key, "message": message}
        for key, message in expected.items()
        if key not in methods
    ]


@contextlib.contextmanager
def atomic_checkpoint_path(destination):
    """torch.save(state, temporary) inside this context; publish after fsync."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("pending-" + uuid.uuid4().hex + ".pt")
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
