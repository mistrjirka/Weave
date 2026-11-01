"""Private local IPC for reusable evaluators. No PyTorch or web dependencies."""

import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
import uuid

from weave_training import atomic_json


def memory_snapshot_directory(required_bytes):
    """Use Linux tmpfs only when active, pending and replacement weights fit."""
    root = Path("/dev/shm")
    try:
        if (
            root.is_dir()
            and shutil.disk_usage(root).free > 3 * required_bytes + 16 * 1024**2
        ):
            return tempfile.TemporaryDirectory(
                prefix=f"weave-evaluation-{os.getpid()}-", dir=root
            )
    except OSError:
        pass
    return None


def read_request(directory):
    path = Path(directory) / "request.json"
    return json.loads(path.read_text()) if path.is_file() else None


def next_request(directory, previous, parent):
    while os.getppid() == parent:
        request = read_request(directory)
        if request and request["id"] != previous:
            return request
        time.sleep(0.05)
    return None


class EvaluationSession:
    """One process, one outstanding immutable checkpoint at a time.

    poll/wait describe the current job, so the trainer can use the same lifecycle
    for one-shot subprocesses and reusable workers. close owns process teardown.
    """

    def __init__(self, directory, device):
        self.directory = Path(directory)
        self.device = device
        self.process = None
        self.request_id = None
        self.result = None

    def submit(self, checkpoint, settings):
        self.request_id = uuid.uuid4().hex
        self.result = None
        atomic_json(
            self.directory / "request.json",
            {
                "id": self.request_id,
                "checkpoint": str(checkpoint),
                "settings": settings,
                "submitted": time.monotonic(),
            },
        )

    def poll(self):
        path = self.directory / "result.json"
        if path.is_file():
            result = json.loads(path.read_text())
            if result["id"] == self.request_id:
                self.result = result
                return 0 if result["success"] else 1
        code = self.process.poll()
        return None if code is None else code or 1

    @property
    def returncode(self):
        return self.poll()

    def wait(self, timeout=None):
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired("validation job", timeout)
            time.sleep(0.05)
        return self.poll()

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
