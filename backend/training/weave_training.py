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
import threading
import urllib.request
import subprocess
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
    def __init__(
        self,
        directory=None,
        *,
        rank=None,
        optimizer=None,
        total_steps=None,
        start_step=0,
        log_every=1,
    ):
        self.directory = Path(
            directory or os.environ.get("WEAVE_RUN_DIR", "weave-run")
        ).resolve()
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.directory.mkdir(parents=True, exist_ok=True)
        if (
            start_step < 0
            or log_every < 1
            or (total_steps is not None and total_steps < start_step)
        ):
            raise ValueError(
                "Use nonnegative start_step, log_every >= 1, and total_steps >= start_step"
            )
        self.step = int(start_step)
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.log_every = int(log_every)
        self._optimizer_updates = 0
        self._loss_reported = False
        self._missing_loss_warned = False
        self._optimizer_handles = {}
        self.tokens = 0
        self.commands_seen = set()
        self._stream = None
        self._write_lock = threading.RLock()
        self._reporter = None
        self._hooks = []

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
        if self.optimizer is not None:
            self.watch_optimizer(self.optimizer, every=self.log_every)
            self.capabilities(
                metrics=True,
                samples=False,
                checkpoint=False,
                commands=False,
                optimizer_tracking=True,
            )
        config_path = Path(os.environ.get("WEAVE_REPORT_CONFIG", "reporting.json"))
        if (
            self.rank == 0
            and config_path.is_file()
            and not (self.directory / ".report-token").exists()
        ):
            self._reporter = _Reporter(self, json.loads(config_path.read_text()))
            self._reporter.start()
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
        if (
            self.optimizer is not None
            and self._optimizer_updates
            and not self._loss_reported
        ):
            self._warn_missing_loss()
        self.emit("finished", {"success": exception is None})
        for handle in self._hooks:
            handle.remove()
        if self._reporter:
            self._reporter.close()
        with self._write_lock:
            if self._stream:
                self._stream.close()
                self._stream = None
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
        with self._write_lock:
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
        if any("loss" in name.lower() for name in clean):
            self._loss_reported = True
        self.emit("metrics", clean)

    def log_loss(self, loss, *, tokens=None, metrics=None):
        """Report the user's chosen scalar objective, using tracked optimizer steps.

        Call after optimizer.step() (and scheduler.step(), if any). No HTTP or
        tensor/backward patching lives in the user's training loop. Pass a reduced
        loss for DDP; this helper never guesses cross-rank or token weighting.
        """
        if not self._optimizer_handles:
            raise RuntimeError(
                "Use Run(optimizer=optimizer) or run.watch_optimizer(optimizer) before log_loss()"
            )
        self._loss_reported = True
        if tokens is not None:
            self.tokens = int(tokens)
        if self.rank != 0:
            return
        if (
            self._optimizer_updates != 1
            and self.step % self.log_every
            and self.step != self.total_steps
        ):
            return
        # Detach only the reporting value; retain the caller's original tensor.
        value = loss.detach() if hasattr(loss, "detach") else loss
        self.log(
            {**(metrics or {}), "train/loss": value}, step=self.step, tokens=tokens
        )

    def _warn_missing_loss(self):
        if not self._missing_loss_warned:
            self._missing_loss_warned = True
            self.warning(
                "Optimizer updates were detected, but no loss has been reported. Add run.log_loss(loss) after the optimizer update, or report your reduced objective with run.log().",
                code="missing-loss",
            )

    def watch_optimizer(self, optimizer, *, every=1):
        """Read LR through PyTorch's public hook; does not patch optimizer code.

        Loss remains explicit: run.log({'train/loss': loss.item()}, step=step).
        Call once inside the Run context, after constructing your optimizer.
        """
        if every < 1:
            raise ValueError("every must be at least 1")
        if id(optimizer) in self._optimizer_handles:
            return self._optimizer_handles[id(optimizer)]
        updates = self.step

        def report(current, args, kwargs):
            nonlocal updates
            updates += 1
            self._optimizer_updates += 1
            self.step = updates
            if (
                self.optimizer is not None
                and not self._loss_reported
                and self._optimizer_updates > max(2, self.log_every)
            ):
                self._warn_missing_loss()
            if updates % every == 0 or (
                self.optimizer is not None
                and (self._optimizer_updates == 1 or updates == self.total_steps)
            ):
                metrics = {
                    f"lr/group_{i}": group["lr"]
                    for i, group in enumerate(current.param_groups)
                }
                self.log(metrics, step=updates)
                if self.total_steps is not None:
                    self.progress(
                        completed=updates,
                        total=self.total_steps,
                        unit="optimizer steps",
                    )

        handle = optimizer.register_step_post_hook(report)
        self._hooks.append(handle)
        self._optimizer_handles[id(optimizer)] = handle
        return handle

    def console(self, text):
        """Explicit log output for independent scripts (also retain normal stdout)."""
        self.emit("console", {"text": str(text)})

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
        "log": "No loss or metric reporting found. Use run.log_loss(loss) with Run(optimizer=...), or run.log().",
        "sample": "No generated-sample reporting found.",
        "checkpoint": "No checkpoint reporting found.",
        "poll_commands": "No cooperative stop/snapshot polling found.",
    }
    return [
        {"code": "missing-" + key, "message": message}
        for key, message in expected.items()
        if key not in methods and not (key == "log" and "log_loss" in methods)
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


class _Reporter:
    """Durable JSONL outbox with a background HTTP sender and bounded retries."""

    def __init__(self, run, config):
        self.run = run
        self.url = config["url"]
        self.token = config["token"]
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self.work, daemon=True, name="weave-reports"
        )
        self.cursor_path = run.directory / "report-cursor.json"
        saved = (
            json.loads(self.cursor_path.read_text())
            if self.cursor_path.exists()
            else {}
        )
        self.cursor = saved.get("offset", 0) if saved.get("url") == self.url else 0
        self.last_hardware = 0
        self.provenance_sent = False

    def start(self):
        # Dataset hashing can be slow; do not delay metric delivery behind it.
        threading.Thread(
            target=self.provenance, daemon=True, name="weave-provenance"
        ).start()
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)

    def hardware(self):
        """Device-wide GPU readings; unavailable NVIDIA tools simply omit them."""
        if time.monotonic() - self.last_hardware < 10:
            return
        self.last_hardware = time.monotonic()
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            ).stdout
            import csv

            gpus = []
            keys = (
                "memory_total_mb",
                "memory_used_mb",
                "utilization",
                "temperature",
                "power_watts",
            )
            for row in csv.reader(output.splitlines()):
                values = [value.strip() for value in row]

                def number(value):
                    try:
                        result = float(value)
                        return result if math.isfinite(result) else None
                    except ValueError:
                        return None

                gpus.append(
                    {
                        "index": int(values[0]),
                        "uuid": values[1],
                        "name": values[2],
                        **dict(zip(keys, map(number, values[3:]))),
                    }
                )
            self.run.emit("hardware", {"time": time.time(), "gpus": gpus})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    def send(self):
        path = self.run.directory / "events.jsonl"
        if self.cursor > path.stat().st_size:
            self.cursor = 0
        batch = []
        size = 0
        with path.open("rb") as stream:
            stream.seek(self.cursor)
            while len(batch) < 100:
                begin = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    stream.seek(begin)
                    break
                if batch and size + len(line) > 1500 * 1024:
                    stream.seek(begin)
                    break
                batch.append(json.loads(line))
                size += len(line)
            end = stream.tell()
        # Empty batches poll cooperative commands and confirm connectivity.
        request = urllib.request.Request(
            self.url,
            json.dumps({"events": batch}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Weave-Report-Token": self.token,
            },
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.load(response)
        for command in result.get("commands", []):
            identifier = command["id"]
            if not isinstance(identifier, str) or not identifier.isalnum():
                continue
            atomic_json(
                self.run.directory / "commands" / (identifier + ".json"), command
            )
        self.cursor = end
        atomic_json(self.cursor_path, {"url": self.url, "offset": end})
        atomic_json(
            self.run.directory / "reporting-status.json",
            {"connected": True, "last_sent": time.time()},
        )
        return end < path.stat().st_size

    def provenance(self):
        try:
            self.collect_provenance()
        except (OSError, ValueError, RuntimeError) as exc:
            with self.run._write_lock:
                if self.run._stream is not None:
                    self.run.warning(
                        f"Could not record independent-run provenance: {exc}",
                        code="provenance",
                    )

    def collect_provenance(self):
        if self.provenance_sent:
            return
        self.provenance_sent = True
        files = {}
        for name in ("train.py", "model.py", "config.json", "environment.json"):
            path = (
                Path(name) if name != "environment.json" else self.run.directory / name
            )
            if path.is_file():
                files[name] = {
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                }
        datasets = {}
        config_path = Path("config.json")
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            for name in ("dataset", "eval_dataset"):
                if config.get(name):
                    path = Path(config[name]).expanduser().resolve()
                    if path.is_file():
                        datasets[name] = {
                            "path": str(path),
                            "sha256": file_hash(path),
                            "bytes": path.stat().st_size,
                        }
        with self.run._write_lock:
            if self.run._stream is not None:
                self.run.emit("provenance", {"files": files, "datasets": datasets})

    def work(self):
        while True:
            try:
                if not self.stop.is_set():
                    self.hardware()
                pending = self.send()
            except Exception as exc:
                # Reporting failure never aborts the user's optimizer loop.
                atomic_json(
                    self.run.directory / "reporting-status.json",
                    {"connected": False, "error": str(exc), "time": time.time()},
                )
                pending = False
            if self.stop.is_set():
                # Flush all queued batches on a clean exit, within close's timeout.
                if pending:
                    continue
                break
            self.stop.wait(0.1 if pending else 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Resend durable reports after a server outage"
    )
    parser.add_argument("--sync", metavar="RUN_DIRECTORY", required=True)
    parser.add_argument("--connection", default="reporting.json")
    args = parser.parse_args()
    reporter = _Reporter(Run(args.sync), json.loads(Path(args.connection).read_text()))
    while reporter.send():
        pass
    print("All queued reports delivered")
