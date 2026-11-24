"""Durable reporting outbox for standalone runs."""

import json
import math
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

if __package__:
    from .weave_training_files import atomic_json, file_hash
else:
    from weave_training_files import atomic_json, file_hash


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
        source_root = self.run.source_directory
        for name in (
            "train.py",
            "weave_support/model.py",
            "config.json",
            "environment.json",
        ):
            path = (
                source_root / name
                if name != "environment.json"
                else self.run.directory / name
            )
            if path.is_file():
                files[name] = {
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                }
        datasets = {}
        config_path = source_root / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            for name in ("dataset", "eval_dataset"):
                if config.get(name):
                    path = Path(config[name]).expanduser()
                    if not path.is_absolute():
                        path = config_path.parent / path
                    path = path.resolve()
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
