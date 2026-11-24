"""Independent run supervisor: survives browser and API server restarts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from .service import build_snapshot, events, hardware, index_busy
from .weave_training import Run, atomic_json
from .recipe import is_code_first


def index_events(directory, **options):
    """Read the run's event index, or return None while another reader holds it.

    Every index read takes SQLite's writer lock to advance the index, and the API,
    SSE streams and MCP clients take the same lock. A busy dashboard or client
    must never stop healthy training, so a lock timeout here only means "no new
    information this tick"; the supervisor retries on its next tick. Other
    database errors still fail the run.
    """
    try:
        return events(directory, timeout=0.2, **options)
    except sqlite3.OperationalError as exc:
        if index_busy(exc):
            return None
        raise


def main():
    directory = Path(sys.argv[1]).resolve()
    python = sys.argv[2]
    config = json.loads((directory / "config.json").read_text())
    environment = dict(
        os.environ,
        WEAVE_RUN_DIR=str(directory),
        PYTHONUNBUFFERED="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        # The API disables Transformers model imports for tokenizer inspection.
        # Training/evaluation workers must discover their selected PyTorch runtime.
        USE_TORCH="1",
    )
    # Operator credentials must never enter an editable training script.
    environment.pop("WEAVE_TRAINING_TOKEN", None)
    code_first = is_code_first((directory / "train.py").read_text())
    command = (
        [python, "train.py"]
        if code_first
        else [python, "train.py", "--config", "config.json"]
    )
    if config["strategy"] == "ddp" and not code_first:
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(config['devices'])}",
            "train.py",
            "--config",
            "config.json",
        ]
    process = None
    evaluation = None
    completed_commands = set()
    evaluated = set()
    checkpoints = []
    metrics_received = False
    started = time.time()
    terminal_wait_started = None

    def status(state, **extra):
        atomic_json(
            directory / "status.json",
            {
                "state": state,
                "started": started,
                "updated": time.time(),
                "pid": process.pid if process else None,
                **extra,
            },
        )

    def terminate(child):
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    try:
        status("starting")
        # Capture actual selected environment without importing editable code.
        probe = subprocess.run(
            [
                python,
                "-c",
                "import json,platform,torch,importlib.metadata as m; print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'packages':{d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']},'devices':[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]}))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
            env=environment,
        )
        environment_info = json.loads(probe.stdout)
        atomic_json(directory / "environment.json", environment_info)
        package_lines = [
            f"{name}=={version}"
            for name, version in sorted(environment_info["packages"].items())
        ]
        if environment_info["cuda"]:
            package_lines.insert(
                0,
                "--extra-index-url https://download.pytorch.org/whl/cu"
                + environment_info["cuda"].replace(".", ""),
            )
        (directory / "requirements-lock.txt").write_text(
            "\n".join(package_lines) + "\n"
        )
        if config["devices"] and not json.loads(probe.stdout)["cuda_available"]:
            raise ValueError(
                "Selected Python environment has no usable CUDA. Select a CUDA-enabled Python interpreter or CPU."
            )
        with (directory / "output.log").open("a", buffering=1) as output:
            process = subprocess.Popen(
                command,
                cwd=directory,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        status("running")
        with Run(directory / "system") as reporter:
            while True:
                record = {"time": time.time(), "gpus": hardware()}
                try:
                    import psutil

                    host = psutil.virtual_memory()
                    cpu = psutil.Process(process.pid)
                    record.update(
                        cpu_percent=psutil.cpu_percent(),
                        ram_used_bytes=host.used,
                        ram_total_bytes=host.total,
                        process_rss_bytes=cpu.memory_info().rss,
                    )
                except (ImportError, ProcessLookupError):
                    pass
                except Exception:
                    pass
                with (directory / "hardware.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                # Force stop must not wait behind any event-index access.
                commands = list((directory / "commands").glob("*.json"))
                for path in commands:
                    request = json.loads(path.read_text())
                    if request["kind"] == "kill" and request["id"] not in completed_commands:
                        terminate(process)
                        terminate(evaluation)
                        status("killed", message="Force stopped; only previously completed checkpoints are available.")
                        return
                found = index_events(directory, kind="checkpoint")
                if found is not None:
                    checkpoints = [e for e in found if e.get("channel") == "training"]
                for path in commands:
                    request = json.loads(path.read_text())
                    if request["id"] in completed_commands:
                        continue
                    if request["kind"] == "snapshot" and found is not None:
                        checkpoint = next(
                            (
                                e
                                for e in reversed(checkpoints)
                                if request["id"]
                                in e["payload"]
                                .get("metadata", {})
                                .get("snapshot_requests", [])
                            ),
                            None,
                        )
                        if checkpoint or process.poll() is not None:
                            try:
                                build_snapshot(
                                    directory,
                                    checkpoint or (checkpoints[-1] if checkpoints else None),
                                    request["include_data"],
                                    request["id"],
                                    index_timeout=0.2,
                                )
                            except sqlite3.OperationalError as exc:
                                if not index_busy(exc):
                                    raise
                                found = None
                                continue
                            completed_commands.add(request["id"])
                if (
                    config["strategy"] == "train_eval"
                    and not code_first
                    and config["recipe"] != "custom"
                ):
                    if evaluation and evaluation.poll() is not None:
                        if evaluation.returncode:
                            reporter.warning(
                                "Evaluation process failed; see evaluation.log",
                                code="evaluation-failed",
                            )
                        evaluation = None
                    pending = [
                        e for e in checkpoints if e["payload"]["path"] not in evaluated
                    ] if found is not None else []
                    if pending and evaluation is None:
                        # Evaluate latest snapshot; older superseded jobs do not pile up.
                        selected = pending[-1]["payload"]["path"]
                        evaluated.update(e["payload"]["path"] for e in pending)
                        with (directory / "evaluation.log").open("a") as output:
                            evaluation = subprocess.Popen(
                                [
                                    python,
                                    "train.py",
                                    "--config",
                                    "config.json",
                                    "--evaluate",
                                    str(directory / selected),
                                ],
                                cwd=directory,
                                env=environment,
                                stdout=output,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                if process.poll() is not None and evaluation is None:
                    stopped = index_events(directory, kind="stopped", limit=1) if found is not None else None
                    if stopped is None:
                        if terminal_wait_started is None:
                            terminal_wait_started = time.monotonic()
                        if time.monotonic() - terminal_wait_started >= 10:
                            status("unknown" if process.returncode == 0 else "failed", exit_code=process.returncode, message="Training exited, but its final state could not be determined because the event index remained busy.")
                            break
                        time.sleep(0.2)
                        continue
                    state = "completed" if process.returncode == 0 else "failed"
                    if stopped:
                        state = "stopped"
                    status(
                        state,
                        exit_code=process.returncode,
                        message=(
                            None
                            if state != "failed"
                            else "Training failed. Open Logs for the Python traceback."
                        ),
                    )
                    break
                if not metrics_received:
                    found = index_events(directory, kind="metrics", limit=1)
                    metrics_received = bool(found)
                status(
                    "running" if process.poll() is None else "evaluating",
                    metrics_received=metrics_received,
                )
                time.sleep(2)
    except BaseException as exc:
        terminate(process)
        terminate(evaluation)
        status("failed", message=str(exc))
        with (directory / "output.log").open("a") as output:
            output.write(str(exc) + "\n")
        raise


if __name__ == "__main__":
    main()
