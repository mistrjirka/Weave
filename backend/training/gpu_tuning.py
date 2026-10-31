"""Persistent host-wide GPU benchmark jobs and compatible training profiles."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from .layout import write_support
from .service import model_metadata, valid_id
from .tuning_math import graph_signature
from .weave_training import atomic_json
from ..compiler import compile_project, export_zip
from ..core import params, validate_project, ordered
from ..tokenizer_bindings import bind_input_tokenizers


class TuningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: dict
    python: str
    devices: list[int] = Field(min_length=1, max_length=8)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    sequence_length: int = Field(default=128, ge=2, le=32768)
    global_batch_size: int = Field(default=8, ge=1, le=1024)
    warmup_steps: int = Field(default=2, ge=1, le=20)
    measure_steps: int = Field(default=5, ge=3, le=50)
    max_candidates: int = Field(default=4, ge=1, le=12)
    candidate_timeout: int = Field(default=120, ge=15, le=600)


def profile_summary(profile):
    return {key: value for key, value in profile.items() if key != "results"}


class GPUTuning:
    def __init__(self, training_root):
        self.root = Path(training_root).expanduser().resolve() / "gpu-tuning"
        self.root.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.lock = threading.Lock()

    def settings(self):
        path = self.root / "settings.json"
        return json.loads(path.read_text()) if path.exists() else {"auto_apply": True}

    def configure(self, auto_apply):
        atomic_json(self.root / "settings.json", {"auto_apply": bool(auto_apply)})
        return self.settings()

    def directory(self, identifier):
        directory = self.root / valid_id(identifier)
        if not (directory / "request.json").is_file():
            raise ValueError("Unknown GPU benchmark")
        return directory

    def owned_pid(self, directory):
        path = directory / "process.json"
        if not path.exists():
            return None
        pid = json.loads(path.read_text())["pid"]
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            return (
                pid if str(directory / "tuning_worker.py").encode() in command else None
            )
        except OSError:
            return None

    def status(self, identifier):
        directory = self.directory(identifier)
        path = directory / "status.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else {"state": "starting", "results": [], "message": "Starting benchmark"}
        )
        process = self.processes.get(identifier)
        if (
            process
            and process.poll() is not None
            and result["state"] in ("running", "starting")
        ):
            result.update(
                state="failed",
                message="Benchmark exited unexpectedly; see worker logs.",
            )
        if (
            not process
            and result["state"] in ("running", "starting")
            and not self.owned_pid(directory)
        ):
            result.update(
                state="interrupted",
                message="Benchmark process is no longer running; partial results are not active.",
            )
        return {
            "id": identifier,
            **result,
            "logs": (
                (directory / "worker.log").read_text()[-6000:]
                if (directory / "worker.log").exists()
                else ""
            ),
        }

    def list(self):
        profiles, jobs = [], []
        for path in sorted(
            self.root.glob("tune-*/request.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            directory = path.parent
            status = self.status(directory.name)
            jobs.append(
                {
                    key: value
                    for key, value in status.items()
                    if key not in ("results", "logs")
                }
            )
            if (directory / "profile.json").exists() and status["state"] == "completed":
                profiles.append(
                    profile_summary(
                        json.loads((directory / "profile.json").read_text())
                    )
                )
        return {"settings": self.settings(), "profiles": profiles, "jobs": jobs}

    def start(self, request):
        if len(set(request.devices)) != len(request.devices) or any(
            index < 0 for index in request.devices
        ):
            raise ValueError("Select unique nonnegative GPU indices")
        if request.global_batch_size < len(request.devices):
            raise ValueError(
                "Global microbatch must contain at least one sequence per GPU"
            )
        interpreter = shutil.which(request.python)
        if not interpreter:
            raise ValueError("Benchmark Python executable not found")
        request = request.model_copy(update={"python": os.path.abspath(interpreter)})
        report = validate_project(request.project)
        if (
            not report["valid"]
            or len(report["outputs"]) != 1
            or not report["outputs"][0]
            or report["outputs"][0]["dtype"] != "float32"
        ):
            raise ValueError(
                "Benchmark requires a valid graph with one floating-point prediction output"
            )
        # Dynamic dimensions follow the compiler contract; optional inputs keep defaults.
        project, _, _ = bind_input_tokenizers(request.project, ordered)
        config = {**project["config"], **project["graph"].get("variables", {})}
        nodes = {
            params(node, config).get("index", 0): params(node, config)
            for node in project["graph"]["nodes"]
            if node["data"]["kind"] == "input"
        }
        last_required = max(
            (index for index, options in nodes.items() if not options.get("optional")),
            default=-1,
        )
        inputs = []
        for index in range(last_required + 1):
            options = nodes.get(index)
            if not options or options.get("optional"):
                inputs.append(None)
                continue
            shape = options.get("shape", ["B", "T"])
            if any(isinstance(d, str) and d not in ("B", "T", "S") for d in shape):
                raise ValueError(
                    "Benchmark inputs support B/T/S symbols and numeric dimensions; resolve other dimensions first"
                )
            inputs.append({"shape": shape, "dtype": options.get("dtype", "int64")})
        if not any(
            item and item["shape"] and item["shape"][0] == "B" for item in inputs
        ):
            raise ValueError(
                "GPU allocation tuning requires a dynamic B batch dimension on at least one required input"
            )
        files = compile_project(request.project)
        metadata = model_metadata(files, request.project)
        with self.lock, (self.root / ".start.lock").open("a") as start_lock:
            import fcntl

            fcntl.flock(start_lock, fcntl.LOCK_EX)
            if any(
                process.poll() is None for process in self.processes.values()
            ) or any(
                self.owned_pid(path.parent)
                for path in self.root.glob("tune-*/process.json")
            ):
                raise ValueError(
                    "A GPU benchmark is already running; stop it or wait for completion"
                )
            identifier = "tune-" + uuid.uuid4().hex
            directory = self.root / identifier
            directory.mkdir()
            write_support(directory, export_zip(request.project))
            shutil.copyfile(
                Path(__file__).with_name("tuning_worker.py"),
                directory / "tuning_worker.py",
            )
            atomic_json(
                directory / "request.json",
                {
                    **request.model_dump(),
                    "started_at": time.time(),
                    **metadata,
                    "inputs": inputs,
                    "devices": sorted(request.devices),
                },
            )
            atomic_json(
                directory / "status.json",
                {
                    "state": "starting",
                    "results": [],
                    "message": "Starting isolated synthetic model benchmark",
                },
            )
            environment = {
                **os.environ,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "USE_TORCH": "1",
            }
            with (directory / "worker.log").open("w") as logs:
                process = subprocess.Popen(
                    [
                        request.python,
                        str(directory / "tuning_worker.py"),
                        "--root",
                        str(directory),
                    ],
                    cwd=directory,
                    env=environment,
                    stdout=logs,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            atomic_json(directory / "process.json", {"pid": process.pid})
            self.processes[identifier] = process
        return self.status(identifier)

    def stop(self, identifier):
        directory = self.directory(identifier)
        process = self.processes.get(identifier)
        pid = (
            process.pid
            if process and process.poll() is None
            else self.owned_pid(directory)
        )
        if pid:
            os.killpg(pid, signal.SIGTERM)
            if process:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL)
                    process.wait()
            else:
                for _ in range(100):
                    if not self.owned_pid(directory):
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(pid, signal.SIGKILL)
            result = self.status(identifier)
            atomic_json(
                directory / "status.json",
                {
                    **result,
                    "state": "cancelled",
                    "message": "Benchmark stopped; partial results were not activated",
                },
            )
        return self.status(identifier)

    def close(self):
        for identifier in list(self.processes):
            self.stop(identifier)


def matching_profile(training_root, project, config):
    """Runtime validates physical GPUs/library versions again before applying."""
    if (
        not config.use_gpu_tuning
        or config.strategy not in ("ddp", "split")
        or len(config.devices) < 2
    ):
        return None
    tuning = GPUTuning(training_root)
    if not tuning.settings()["auto_apply"]:
        return None
    signature = graph_signature(project)
    for profile in tuning.list()["profiles"]:
        candidate = profile["best"].get(config.strategy)
        if (
            profile["model_signature"] == signature
            and profile["devices"] == sorted(config.devices)
            and os.path.abspath(profile["python"]) == os.path.abspath(config.python)
            and candidate
            and candidate["precision"] == config.precision
            and profile["global_batch_size"]
            == config.batch_size
            * (len(config.devices) if config.strategy == "ddp" else 1)
        ):
            return {
                **{
                    key: profile[key]
                    for key in (
                        "id",
                        "model_signature",
                        "hardware",
                        "devices",
                        "sequence_length",
                        "global_batch_size",
                    )
                },
                "strategy": config.strategy,
                "precision": config.precision,
                "batch_sizes": {
                    str(device): size
                    for device, size in zip(
                        profile["devices"], candidate["batch_sizes"]
                    )
                },
                "placement": candidate["placement"],
                "step_seconds": candidate["step_seconds"],
            }
    return None
