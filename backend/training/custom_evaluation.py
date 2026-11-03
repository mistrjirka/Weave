"""Optional persistent evaluator for user-owned loops, isolated from training.

The callback must be a top-level, spawn-importable function. It receives an
immutable checkpoint path and a device, and returns EvaluationResult or metrics.
"""

from dataclasses import dataclass, field, asdict
import json
import math
import multiprocessing
from pathlib import Path
import time
import traceback

from .weave_training import file_hash


@dataclass
class EvaluationResult:
    metrics: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)


def _worker(connection, evaluate_fn, device):
    try:
        while True:
            job = connection.recv()
            if job is None:
                return
            try:
                if file_hash(job["checkpoint"]) != job["sha256"]:
                    raise ValueError("Evaluation checkpoint changed before loading")
                result = evaluate_fn(Path(job["checkpoint"]), device)
                if isinstance(result, dict):
                    result = EvaluationResult(metrics=result)
                if not isinstance(result, EvaluationResult):
                    raise TypeError(
                        "Evaluation must return numeric metrics or EvaluationResult"
                    )
                result.metrics = {
                    key: float(value) for key, value in result.metrics.items()
                }
                if (
                    len(json.dumps(asdict(result), allow_nan=False).encode())
                    > 1024 * 1024
                ):
                    raise ValueError(
                        "Evaluation reports must be smaller than 1 MB; store large trajectories as artifacts in the worker"
                    )
                if file_hash(job["checkpoint"]) != job["sha256"]:
                    raise ValueError("Evaluation checkpoint changed while evaluating")
                connection.send({"result": result})
            except Exception:
                connection.send({"error": traceback.format_exc()})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class AsyncEvaluator:
    """One reusable worker, one outstanding snapshot, no unbounded job queue.

    submit returns False while busy. Call poll at safe boundaries to report
    completed results at their original policy step, never the current step.
    Rollout collection for training remains controlled by the RL algorithm.
    """

    def __init__(self, run, evaluate_fn, *, device="cpu", timeout=600):
        if run.rank != 0:
            raise ValueError("Create the evaluator only on rank zero")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Evaluation timeout must be positive")
        self.run, self.evaluate_fn, self.device = run, evaluate_fn, device
        self.timeout = timeout
        self.process = self.connection = self.job = None

    def __enter__(self):
        return self

    @property
    def busy(self):
        return self.job is not None

    def _start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.process = context.Process(
            target=_worker, args=(child, self.evaluate_fn, self.device), daemon=True
        )
        try:
            self.process.start()
        except BaseException:
            parent.close()
            child.close()
            self.process = None
            raise
        child.close()
        self.connection = parent

    def submit(self, checkpoint, *, step=None, counters=None):
        self.poll()
        if self.job is not None:
            return False
        path = Path(checkpoint).resolve()
        path.relative_to(self.run.directory)
        if not path.is_file():
            raise ValueError("Submit a completed checkpoint inside the run directory")
        if self.process is None:
            self._start()
        step = self.run.step if step is None else step
        if type(step) is not int or step < 0:
            raise ValueError("Evaluation step must be a nonnegative integer")
        self.job = {
            "checkpoint": str(path),
            "sha256": file_hash(path),
            "step": step,
            "counters": dict(self.run.counters if counters is None else counters),
            "submitted": time.monotonic(),
        }
        self.connection.send(self.job)
        self.run.emit(
            "evaluation_status",
            self._status("running"),
            step=step,
            channel="evaluation",
        )
        return True

    def _status(self, state, **values):
        return {
            "state": state,
            "checkpoint": str(
                Path(self.job["checkpoint"]).relative_to(self.run.directory)
            ),
            "step": self.job["step"],
            "device": str(self.device),
            "policy_version": self.job["sha256"],
            **values,
        }

    def poll(self, *, raise_on_error=False):
        if self.job is None:
            return None
        failure, response = None, None
        if self.connection.poll():
            try:
                response = self.connection.recv()
            except (EOFError, OSError):
                failure = "Evaluation worker disconnected"
        elif not self.process.is_alive():
            failure = "Evaluation worker exited without a result"
        elif time.monotonic() - self.job["submitted"] >= self.timeout:
            failure = f"Evaluation exceeded {self.timeout:g} seconds"
        else:
            return None
        failure = failure or (response or {}).get("error")
        job = self.job
        if not failure:
            try:
                result = response["result"]
                metrics = {key: float(value) for key, value in result.metrics.items()}
                if any(not math.isfinite(value) for value in metrics.values()):
                    raise ValueError("Evaluation metrics must be finite")
                self.run.emit(
                    "metrics",
                    metrics,
                    step=job["step"],
                    counters=job["counters"],
                    channel="evaluation",
                    checkpoint=self._status("completed")["checkpoint"],
                    policy_version=job["sha256"],
                )
                for sample in result.samples:
                    self.run.emit(
                        "sample",
                        {
                            **sample,
                            "checkpoint": self._status("completed")["checkpoint"],
                            "metadata": {
                                **sample.get("metadata", {}),
                                "policy_version": job["sha256"],
                            },
                        },
                        step=job["step"],
                        counters=job["counters"],
                        channel="evaluation",
                    )
                for name, rows in result.tables.items():
                    from .run_contract import write_table

                    payload = write_table(
                        self.run.directory,
                        name,
                        rows,
                        None,
                        {"policy_version": job["sha256"]},
                    )
                    self.run.emit(
                        "table",
                        payload,
                        step=job["step"],
                        counters=job["counters"],
                        channel="evaluation",
                    )
            except Exception:
                failure = traceback.format_exc()
        self.run.emit(
            "evaluation_status",
            self._status(
                "failed" if failure else "completed",
                **({"error": failure} if failure else {}),
            ),
            step=job["step"],
            channel="evaluation",
        )
        self.job = None
        if failure:
            self._terminate()
            self.run.warning(failure, code="custom-evaluation")
            if raise_on_error:
                raise RuntimeError(failure)
        return response.get("result") if response and not failure else None

    def _terminate(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            self.process = None
        if self.connection:
            self.connection.close()
            self.connection = None

    def close(self, *, wait=True):
        if wait:
            while self.job is not None:
                self.poll()
                if self.job is not None:
                    time.sleep(0.05)
        elif self.job is not None:
            self.run.emit(
                "evaluation_status",
                self._status("cancelled"),
                step=self.job["step"],
                channel="evaluation",
            )
            self.job = None
        self._terminate()

    def __exit__(self, kind, exception, tb):
        self.close(wait=exception is None)
