"""Portable Weave reporting API. Uses only the Python standard library.

Training remains ordinary PyTorch. Events are durable local JSONL records; a
missing browser or server never blocks an optimizer step. Only rank zero writes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import traceback
import threading
import uuid

if __package__:
    from .weave_training_files import atomic_json, file_hash, atomic_checkpoint_path
    from .weave_training_commands import _CommandCheckpointMixin
    from .weave_training_reporter import _Reporter
    from .weave_training_inspection import instrumentation_warnings
else:
    from weave_training_files import atomic_json, file_hash, atomic_checkpoint_path
    from weave_training_commands import _CommandCheckpointMixin
    from weave_training_reporter import _Reporter
    from weave_training_inspection import instrumentation_warnings

PROTOCOL_VERSION = 1


def available_cpu_threads():
    """Use all CPUs available to this process, respecting Linux CPU affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
            if affinity:
                return len(affinity)
        except OSError:
            pass  # Some platforms expose the API without supporting it.
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, count or 1)


class Run(_CommandCheckpointMixin):
    def __init__(
        self,
        directory=None,
        *,
        rank=None,
        optimizer=None,
        total_steps=None,
        start_step=0,
        log_every=1,
        source_directory=None,
        channel=None,
    ):
        self.directory = Path(
            directory or os.environ.get("WEAVE_RUN_DIR", "weave-run")
        ).resolve()
        self.source_directory = (
            Path(source_directory).resolve()
            if source_directory
            else (
                self.directory
                if (self.directory / "train.py").is_file()
                else Path.cwd()
            )
        )
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        if channel not in (None, "training", "evaluation", "system"):
            raise ValueError("Unknown reporting channel")
        self.channel = channel
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
        self.counters = {}
        self._capabilities = {}

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
        self.emit(
            "started",
            {
                "protocol": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "directory": str(self.directory),
                "source_directory": str(self.source_directory),
            },
        )
        if self.optimizer is not None:
            self.watch_optimizer(self.optimizer, every=self.log_every)
            self.capabilities(
                metrics=True,
                samples=False,
                checkpoint=False,
                commands=False,
                optimizer_tracking=True,
            )
        configured_connection = os.environ.get("WEAVE_REPORT_CONFIG")
        config_path = (
            Path(configured_connection)
            if configured_connection
            else self.source_directory / "reporting.json"
        )
        if not configured_connection and not config_path.is_file():
            config_path = Path("reporting.json")
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

    def emit(
        self,
        kind,
        payload,
        *,
        step=None,
        tokens=None,
        counters=None,
        channel=None,
        checkpoint=None,
        policy_version=None,
    ):
        if self.rank != 0:
            return
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("Event steps must be nonnegative integers")
        if channel not in (None, "training", "evaluation", "system"):
            raise ValueError("Unknown reporting channel")
        record = {
            "version": PROTOCOL_VERSION,
            "id": uuid.uuid4().hex,
            "time": time.time(),
            "kind": kind,
            "step": self.step if step is None else step,
            "tokens": self.tokens if tokens is None else tokens,
            "payload": payload,
        }
        if channel is not None or self.channel is not None:
            record["channel"] = channel if channel is not None else self.channel
        if counters is not None or self.counters:
            record["counters"] = dict(self.counters if counters is None else counters)
        if checkpoint is not None:
            record["checkpoint"] = str(checkpoint)
        if policy_version is not None:
            record["policy_version"] = str(policy_version)
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

    def log(self, metrics, *, step, tokens=None, counters=None):
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        self.step = int(step)
        if counters is not None:
            self.set_counters(**counters)
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
            updates = self.step + 1
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

    def warning(self, message, *, code="custom", **details):
        self.emit("warning", {"message": message, "code": code, **details})

    def capabilities(self, **capabilities):
        if any(type(value) is not bool for value in capabilities.values()):
            raise ValueError("Capabilities must be explicit booleans")
        self._capabilities.update(capabilities)
        self.emit("capabilities", dict(self._capabilities))

    def configure(
        self,
        values,
        *,
        metrics=None,
        primary_metric=None,
        step_unit="optimizer updates",
        headline_metrics=None,
    ):
        """Report applied settings and presentation; never applies training settings."""
        from .run_contract import reporting_contract

        contract = reporting_contract(
            metrics or {}, primary_metric, step_unit, headline_metrics
        )
        configuration = {**values, "reporting": contract}
        json.dumps(configuration, allow_nan=False)
        self.emit("configuration", configuration)

    def set_counters(self, **counters):
        """Named cumulative counters, independent of optimizer hook tracking."""
        if len(self.counters.keys() | counters.keys()) > 32:
            raise ValueError("Report at most 32 named counters")
        for name, value in counters.items():
            if not isinstance(name, str) or not name or len(name) > 80:
                raise ValueError("Counter names need 1–80 characters")
            if type(value) is not int or value < 0:
                raise ValueError("Counters must be nonnegative integers")
        self.counters.update(counters)

    def artifact(self, path, *, name=None, metadata=None):
        """Register a completed local file; remote reporting does not upload it."""
        path = Path(path).resolve()
        relative = path.relative_to(self.directory)
        if not path.is_file():
            raise ValueError("Publish a completed file inside the run directory")
        payload = {
            "path": str(relative),
            "name": name or path.name,
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
            "metadata": metadata or {},
        }
        self.emit("artifact", payload)
        return path

    def table(self, name, rows, *, columns=None, metadata=None):
        """Stream JSON rows to an artifact; embed only a small preview in events."""
        from .run_contract import write_table

        if self.rank != 0:
            return None
        payload = write_table(self.directory, name, rows, columns, metadata)
        self.emit("table", payload)
        return self.directory / payload["path"]


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
