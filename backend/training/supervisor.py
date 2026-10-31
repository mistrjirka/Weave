"""Independent run supervisor: survives browser and API server restarts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .service import build_snapshot, events, hardware
from .weave_training import Run, atomic_json
from .recipe import is_code_first


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
    started = time.time()

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
                checkpoints = [
                    e
                    for e in events(directory)
                    if e["kind"] == "checkpoint" and e.get("channel") == "training"
                ]
                for path in (directory / "commands").glob("*.json"):
                    request = json.loads(path.read_text())
                    if request["id"] in completed_commands:
                        continue
                    if request["kind"] == "kill":
                        terminate(process)
                        terminate(evaluation)
                        status(
                            "killed",
                            message="Force stopped; only previously completed checkpoints are available.",
                        )
                        return
                    if request["kind"] == "snapshot":
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
                            build_snapshot(
                                directory,
                                checkpoint
                                or (checkpoints[-1] if checkpoints else None),
                                request["include_data"],
                                request["id"],
                            )
                            completed_commands.add(request["id"])
                if config["strategy"] == "train_eval" and not code_first:
                    if evaluation and evaluation.poll() is not None:
                        if evaluation.returncode:
                            reporter.warning(
                                "Evaluation process failed; see evaluation.log",
                                code="evaluation-failed",
                            )
                        evaluation = None
                    pending = [
                        e for e in checkpoints if e["payload"]["path"] not in evaluated
                    ]
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
                    state = "completed" if process.returncode == 0 else "failed"
                    if any(e["kind"] == "stopped" for e in events(directory)):
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
                status(
                    "running" if process.poll() is None else "evaluating",
                    metrics_received=any(
                        e["kind"] == "metrics" for e in events(directory)
                    ),
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
