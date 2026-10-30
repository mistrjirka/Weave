"""Training documents, immutable run exports, and verified portable archives."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from pydantic import BaseModel, Field, ConfigDict
from ..compiler import export_zip
from ..core import validate_project
from .weave_training import atomic_json, file_hash, instrumentation_warnings
from .layout import PACKAGE, normalize_script, write_support

HERE = Path(__file__).parent


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: str = Field(default="causal", pattern="^(causal|sft|custom)$")
    strategy: str = Field(default="single", pattern="^(single|ddp|split|train_eval)$")
    execution: str = Field(default="local", pattern="^(local|monitor)$")
    python: str = sys.executable
    devices: list[int] = Field(default_factory=list, max_length=32)
    evaluation_device: int | None = None
    placement: dict[str, int] = Field(default_factory=dict)
    dataset: str = ""
    eval_dataset: str = ""
    sequence_length: int = Field(default=128, ge=2, le=1048576)
    batch_size: int = Field(default=2, ge=1, le=65536)
    accumulation: int = Field(default=1, ge=1, le=65536)
    max_steps: int = Field(default=100, ge=1, le=10**12)
    learning_rate: float = Field(default=0.0003, gt=0, le=10)
    weight_decay: float = Field(default=0.01, ge=0, le=100)
    gradient_clip: float = Field(default=1, gt=0, le=1e6)
    warmup_steps: int = Field(default=10, ge=0)
    log_every: int = Field(default=1, ge=1)
    checkpoint_every: int = Field(default=50, ge=1)
    eval_batches: int = Field(default=5, ge=1)
    generation_tokens: int = Field(default=32, ge=1, le=65536)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    precision: str = Field(default="fp32", pattern="^(fp32|fp16|bf16)$")
    start_mode: str = Field(default="random", pattern="^(random|weights|resume)$")
    parent_run: str | None = None
    parent_checkpoint: str | None = None
    chat_template: str = ""
    prompts: list[str | dict] = Field(default_factory=list, max_length=100)
    cpu_threads: int = Field(default=2, ge=1, le=256)
    inspect_blocks: bool = False
    summary_every: int = Field(default=50, ge=1)
    custom: dict = Field(default_factory=dict)

    def validate_execution(self):
        if len(set(self.devices)) != len(self.devices) or any(
            i < 0 for i in self.devices
        ):
            raise ValueError("Select unique nonnegative GPU indices.")
        if self.strategy in ("ddp", "split") and len(self.devices) < 2:
            raise ValueError("This strategy requires at least two selected GPUs.")
        if self.strategy == "single" and len(self.devices) > 1:
            raise ValueError("Single-device training accepts at most one GPU.")
        if self.strategy == "train_eval" and (
            len(self.devices) != 1
            or self.evaluation_device is None
            or self.evaluation_device in self.devices
        ):
            raise ValueError("Select one training GPU and a different evaluation GPU.")
        if self.start_mode != "random" and not (
            self.parent_run and self.parent_checkpoint
        ):
            raise ValueError("Choose a parent run and checkpoint.")
        return self


class Stage(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    script: str = Field(max_length=1024 * 1024)
    script_path: str | None = None
    config_path: str | None = None
    script_revision: str | None = None
    config_revision: str | None = None
    workspace_warning: str | None = None
    project_path: str | None = None
    model_path: str | None = None
    support_path: str | None = None
    previous_script: str | None = None


class Plan(BaseModel):
    version: int = 1
    project_id: str
    stages: list[Stage] = Field(default_factory=list, max_length=100)


class Launch(BaseModel):
    project: dict
    stage: Stage


def valid_id(value):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,150}", value):
        raise ValueError("Invalid resource identifier")
    return value


def inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact path escapes its run directory")
    return path


def events(directory, *, limit=None, kind=None, after=None):
    """Incremental SQLite index over durable script-owned JSONL event streams."""
    import sqlite3

    database = directory / "event-index.sqlite"
    with sqlite3.connect(database, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS offsets (path TEXT PRIMARY KEY, offset INTEGER)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, kind TEXT, document TEXT)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS kind_index ON events(kind)")
        sources = [
            directory / "events.jsonl",
            directory / "system/events.jsonl",
            *sorted(directory.glob("evaluation-*/events.jsonl")),
        ]
        for path in sources:
            if not path.exists():
                continue
            name = str(path.relative_to(directory))
            row = connection.execute(
                "SELECT offset FROM offsets WHERE path=?", (name,)
            ).fetchone()
            offset = row[0] if row and row[0] <= path.stat().st_size else 0
            with path.open("rb") as stream:
                stream.seek(offset)
                while True:
                    begin = stream.tell()
                    line = stream.readline(1024 * 1024 + 2)
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        if len(line) > 1024 * 1024:
                            while line and not line.endswith(b"\n"):
                                line = stream.readline(1024 * 1024)
                            continue
                        stream.seek(begin)
                        break
                    try:
                        event = json.loads(line)
                        if (
                            event.get("version") != 1
                            or not isinstance(event.get("kind"), str)
                            or not isinstance(event.get("payload"), dict)
                        ):
                            continue
                        import math

                        if (
                            not isinstance(event.get("id"), str)
                            or not isinstance(event.get("time"), (int, float))
                            or not math.isfinite(event["time"])
                            or type(event.get("step")) is not int
                            or type(event.get("tokens")) is not int
                        ):
                            continue
                        event["channel"] = (
                            event.get("channel")
                            if event.get("channel")
                            in ("training", "system", "evaluation")
                            else (
                                "training"
                                if path.parent == directory
                                else (
                                    "system"
                                    if path.parent.name == "system"
                                    else "evaluation"
                                )
                            )
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO events(id,kind,document) VALUES(?,?,?)",
                            (event["id"], event["kind"], json.dumps(event)),
                        )
                    except (ValueError, KeyError, AttributeError):
                        continue
                connection.execute(
                    "INSERT OR REPLACE INTO offsets(path,offset) VALUES(?,?)",
                    (name, stream.tell()),
                )
        query = "SELECT document FROM events"
        values, filters = [], []
        if kind:
            filters.append("kind=?")
            values.append(kind)
        if after is not None:
            previous = connection.execute(
                "SELECT seq FROM events WHERE id=?", (after,)
            ).fetchone()
            filters.append("seq>?")
            values.append(previous[0] if previous else 0)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY seq " + ("ASC" if after is not None else "DESC")
        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)
        rows = connection.execute(query, values).fetchall()
        return [
            json.loads(row[0])
            for row in (rows if after is not None else reversed(rows))
        ]


def hardware():
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        rows = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)

            def optional(call):
                try:
                    return call()
                except nvml.NVMLError:
                    return None

            power = optional(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
            rows.append(
                {
                    "index": index,
                    "uuid": nvml.nvmlDeviceGetUUID(handle),
                    "name": nvml.nvmlDeviceGetName(handle),
                    "memory_total_mb": memory.total / 1024**2,
                    "memory_used_mb": memory.used / 1024**2,
                    "utilization": optional(
                        lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    ),
                    "temperature": optional(
                        lambda: nvml.nvmlDeviceGetTemperature(
                            handle, nvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "power_watts": power / 1000 if power is not None else None,
                }
            )
        return rows
    except Exception:
        # CLI is a compatibility fallback when NVML bindings are unavailable.
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rows = []
        for line in result.stdout.splitlines():
            values = [v.strip() for v in line.split(",")]

            def numeric(value):
                try:
                    return float(value)
                except ValueError:
                    return None

            rows.append(
                dict(
                    zip(
                        (
                            "index",
                            "uuid",
                            "name",
                            "memory_total_mb",
                            "memory_used_mb",
                            "utilization",
                            "temperature",
                            "power_watts",
                        ),
                        [
                            int(values[0]),
                            values[1],
                            values[2],
                            *[numeric(v) for v in values[3:]],
                        ],
                    )
                )
            )
        return rows
    except (OSError, subprocess.SubprocessError):
        return []


def model_metadata(files, project):
    tree = ast.parse(files["model.py"])
    root = [node for node in tree.body if isinstance(node, ast.ClassDef)][-1]
    init = next(
        node
        for node in root.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    names = []
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.append(target.attr)
    # Support a single path of modules; all residual/attention operations inside
    # each container stay on that module's device.
    from ..codegen import OPERATIONS

    graph = project["graph"]
    computational = [
        n for n in graph["nodes"] if n["data"]["kind"] not in ("input", "output")
    ]
    reasons = []
    for node in computational:
        kind = node["data"]["kind"]
        if kind not in ("block", "loop", "moe") and (
            kind not in OPERATIONS or OPERATIONS[kind].expression is not None
        ):
            reasons.append(
                "Root graph contains functional/custom operations; wrap each stage in a container before using automatic split placement."
            )
        if (
            len([e for e in graph["edges"] if e["target"] == node["id"]]) != 1
            or len([e for e in graph["edges"] if e["source"] == node["id"]]) != 1
        ):
            reasons.append(
                "Automatic split placement requires a single chain of top-level modules."
            )
    if (
        len([n for n in graph["nodes"] if n["data"]["kind"] == "input"]) != 1
        or len([n for n in graph["nodes"] if n["data"]["kind"] == "output"]) != 1
    ):
        reasons.append(
            "The built-in text recipes require a single token input and logits output."
        )
    return {
        "modules": list(dict.fromkeys(names)),
        "split_supported": not reasons,
        "split_reasons": list(dict.fromkeys(reasons)),
    }


class TrainingService:
    def __init__(self, root, tokenizer_root):
        self.root = Path(root)
        self.tokenizer_root = Path(tokenizer_root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "plans").mkdir(exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        from .workspaces import WorkspaceStore

        self.workspaces = WorkspaceStore(self.root)
        self.token_path = self.root / "operator.token"

    @property
    def token(self):
        return os.environ.get("WEAVE_TRAINING_TOKEN", "")

    def run_dir(self, run_id):
        directory = self.root / "runs" / valid_id(run_id)
        if not (directory / "run.json").is_file():
            raise FileNotFoundError("Training run not found")
        return directory

    def list_runs(self, project_id):
        result = []
        for path in (self.root / "runs").glob("*/run.json"):
            info = json.loads(path.read_text())
            if info["project_id"] == project_id:
                info["status"] = self.status(path.parent)
                result.append(info)
        return sorted(result, key=lambda item: item["created"], reverse=True)

    def status(self, directory):
        path = directory / "status.json"
        if path.exists():
            return json.loads(path.read_text())
        info = json.loads((directory / "run.json").read_text())
        if info.get("execution") == "monitor":
            records = [e for e in events(directory) if e.get("channel") == "training"]
            last_start = max(
                (
                    index
                    for index, event in enumerate(records)
                    if event["kind"] == "started"
                ),
                default=0,
            )
            records = records[last_start:]
            if any(e["kind"] == "stopped" for e in records):
                return {"state": "stopped"}
            finished = [e for e in records if e["kind"] == "finished"]
            if finished:
                return {
                    "state": (
                        "completed"
                        if finished[-1]["payload"].get("success")
                        else "failed"
                    )
                }
            if any(e["kind"] == "started" for e in records):
                return {
                    "state": "running",
                    "message": "Terminal-launched process; liveness is based on reported events.",
                }
        return {"state": "prepared"}

    def launch(self, request):
        project, stage = request.project, request.stage
        stage = stage.model_copy(
            update={"script": normalize_script(stage.script, stage.config.recipe)}
        )
        stage.config.validate_execution()
        report = validate_project(project)
        if not report["valid"]:
            raise ValueError(
                "Resolve graph errors before training: "
                + "; ".join(e["message"] for e in report["errors"][:3])
            )
        ast.parse(stage.script)
        run_id = "run-" + uuid.uuid4().hex
        directory = self.root / "runs" / run_id
        directory.mkdir()
        try:
            from ..tokenizer_bindings import input_tokenizers

            directories = {
                info["id"]: self.tokenizer_root / valid_id(info["id"])
                for info in input_tokenizers(project).values()
            }
            bundle = export_zip(project, tokenizer_dirs=directories)
            write_support(directory, bundle)
            extras = self.workspaces.freeze_extras(project["id"], stage.id, directory)
            atomic_json(directory / ".workspace-files.json", extras)
            files = {"model.py": (directory / PACKAGE / "model.py").read_text()}
            metadata = model_metadata(files, project)
            if stage.config.strategy == "split" and not metadata["split_supported"]:
                raise ValueError(" ".join(metadata["split_reasons"]))
            config = stage.config.model_dump()
            code_digest = hashlib.sha256()
            for artifact in sorted(directory.rglob("*")):
                if artifact.is_file() and (
                    artifact.suffix == ".py" or "tokenizers" in artifact.parts
                ):
                    code_digest.update(str(artifact.relative_to(directory)).encode())
                    code_digest.update(file_hash(artifact).encode())
            config["model_sha256"] = code_digest.hexdigest()
            config["script_sha256"] = hashlib.sha256(stage.script.encode()).hexdigest()
            from ..codegen import has_moe

            config["has_moe"] = has_moe(project["graph"])
            if config["recipe"] != "custom":
                if (
                    len(report["outputs"]) != 1
                    or len(report["outputs"][0]["shape"]) != 3
                    or report["outputs"][0]["dtype"] != "float32"
                ):
                    raise ValueError(
                        "The text recipes require one floating-point [batch, sequence, vocabulary] output. Use a custom script for this graph."
                    )
                inputs = [
                    n for n in project["graph"]["nodes"] if n["data"]["kind"] == "input"
                ]
                if len(inputs) != 1:
                    raise ValueError(
                        "The text recipes require one token-ID input. Use a custom script for encoder-decoder or multiple-input graphs."
                    )
                config["vocabulary_size"] = report["outputs"][0]["shape"][-1]
                input_contract = inputs[0]["data"].get("params", {})
                if (
                    input_contract.get("dtype", "int64") != "int64"
                    or len(input_contract.get("shape", ["B", "T"])) != 2
                ):
                    raise ValueError(
                        "The text recipes require an int64 [batch, sequence] input."
                    )
                from ..core import params

                def check_causal(graph, scope):
                    scope = {**scope, **graph.get("variables", {})}
                    for item in graph["nodes"]:
                        p = params(item, scope)
                        if item["data"]["kind"] in ("attention", "sdpa") and not p.get(
                            "causal", True
                        ):
                            raise ValueError(
                                "Autoregressive text training requires causal attention; bidirectional attention can leak future target tokens. Use a custom objective for diffusion or masked prediction."
                            )
                        if item["data"].get("body"):
                            check_causal(item["data"]["body"], {**scope, **p})

                check_causal(project["graph"], project["config"])
            if config["strategy"] == "split" and config["precision"] != "fp32":
                raise ValueError("The split-model template currently requires FP32.")
            config["prepared_sha256"] = None
            datasets = {}
            for key in ("dataset", "eval_dataset"):
                value = config[key]
                if value:
                    path = Path(value).expanduser().resolve()
                    if not path.is_file():
                        raise ValueError(
                            f"{key}: file does not exist on the Weave host: {path}"
                        )
                    config[key] = str(path)
                    datasets[key] = {
                        "path": str(path),
                        "sha256": file_hash(path),
                        "bytes": path.stat().st_size,
                    }
            if stage.config.recipe != "custom" and not config["dataset"]:
                raise ValueError("Select a training dataset file.")
            config["dataset_sha256"] = datasets.get("dataset", {}).get("sha256")
            config["eval_dataset_sha256"] = datasets.get("eval_dataset", {}).get(
                "sha256"
            )
            if config["strategy"] == "split" and set(config["placement"]) != set(
                metadata["modules"]
            ):
                raise ValueError(
                    "Assign every top-level module in split placement: "
                    + ", ".join(metadata["modules"])
                )
            parent = None
            if config["start_mode"] != "random":
                parent_directory = self.run_dir(config["parent_run"])
                checkpoint = inside(parent_directory, config["parent_checkpoint"])
                published = next(
                    (
                        e
                        for e in events(parent_directory)
                        if e["kind"] == "checkpoint"
                        and e["payload"]["path"] == config["parent_checkpoint"]
                    ),
                    None,
                )
                if (
                    not published
                    or published["payload"].get("remote")
                    or not checkpoint.is_file()
                    or file_hash(checkpoint) != published["payload"]["sha256"]
                ):
                    raise ValueError(
                        "Parent checkpoint is missing, unpublished, or has changed."
                    )
                (directory / "starting-weights").mkdir()
                target = directory / "starting-weights" / "parent.pt"
                shutil.copyfile(checkpoint, target)
                config["checkpoint"] = str(target)
                parent = {
                    "run_id": config["parent_run"],
                    "checkpoint": config["parent_checkpoint"],
                    "sha256": file_hash(target),
                }
                provenance = directory / "lineage" / config["parent_run"]
                provenance.mkdir(parents=True)
                for artifact in parent_directory.iterdir():
                    if (
                        artifact.is_file()
                        and artifact.name != "events.jsonl"
                        and artifact.suffix in (".py", ".json", ".md", ".txt", ".log")
                        and not artifact.name.startswith("prepared-")
                        and artifact.name != "reporting.json"
                    ):
                        shutil.copyfile(artifact, provenance / artifact.name)
                atomic_json(provenance / "selected-checkpoint.json", published)
                (provenance / "events.jsonl").write_text(
                    "\n".join(
                        json.dumps(event)
                        for event in events(parent_directory)
                        if event.get("step", 0) <= published["step"]
                    )
                    + "\n"
                )
                for trace in parent_directory.glob("consumed-rank-*.jsonl"):
                    with trace.open() as source_trace, (provenance / trace.name).open(
                        "w"
                    ) as target_trace:
                        for line in source_trace:
                            try:
                                if json.loads(line).get("step", 0) <= published["step"]:
                                    target_trace.write(line)
                            except ValueError:
                                continue
                for folder in ("tokenizers", "licenses", PACKAGE):
                    if (parent_directory / folder).exists():
                        shutil.copytree(
                            parent_directory / folder,
                            provenance / folder,
                            dirs_exist_ok=True,
                        )
                if (parent_directory / "lineage").exists():
                    shutil.copytree(
                        parent_directory / "lineage",
                        directory / "lineage",
                        dirs_exist_ok=True,
                    )
                lineage = json.loads((parent_directory / "run.json").read_text())
                atomic_json(
                    directory / "ancestor.json",
                    {
                        "run": lineage,
                        "dataset_manifest": json.loads(
                            (parent_directory / "datasets.json").read_text()
                        ),
                        "ancestors": (
                            json.loads((parent_directory / "ancestor.json").read_text())
                            if (parent_directory / "ancestor.json").exists()
                            else None
                        ),
                    },
                )
            config.setdefault("checkpoint", None)
            (directory / "train.py").write_text(stage.script)

            atomic_json(directory / "config.json", config)
            atomic_json(directory / "stage.json", stage.model_dump())
            atomic_json(directory / "datasets.json", datasets)
            warnings = instrumentation_warnings(stage.script)
            info = {
                "id": run_id,
                "project_id": project["id"],
                "stage_id": stage.id,
                "name": stage.name,
                "created": datetime.now(timezone.utc).isoformat(),
                "parent": parent,
                "recipe": config["recipe"],
                "strategy": config["strategy"],
                "execution": config["execution"],
                "warnings": warnings,
                "model": metadata,
            }
            atomic_json(directory / "run.json", info)
            if config["execution"] == "local":
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "backend.training.supervisor",
                        str(directory),
                        config["python"],
                    ],
                    cwd=HERE.parent.parent,
                    stdout=(directory / "supervisor.log").open("a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                info["supervisor_pid"] = process.pid
                atomic_json(directory / "run.json", info)
            return info
        except BaseException:
            shutil.rmtree(directory)
            raise

    def complete_snapshots(self, directory):
        checkpoints = events(directory, kind="checkpoint")
        for path in (directory / "commands").glob("*.json"):
            request = json.loads(path.read_text())
            if (
                request["kind"] != "snapshot"
                or (directory / "snapshots" / f"snapshot-{request['id']}.json").exists()
            ):
                continue
            matching = next(
                (
                    e
                    for e in reversed(checkpoints)
                    if request["id"]
                    in e["payload"].get("metadata", {}).get("snapshot_requests", [])
                ),
                None,
            )
            if matching and not matching["payload"].get("remote"):
                build_snapshot(
                    directory, matching, request["include_data"], request["id"]
                )

    def command(self, run_id, kind, include_data=False):
        directory = self.run_dir(run_id)
        identifier = uuid.uuid4().hex
        command = {
            "id": identifier,
            "kind": kind,
            "include_data": include_data,
            "time": time.time(),
        }
        atomic_json(directory / "commands" / (identifier + ".json"), command)
        remote = (directory / "reporting-status.json").exists() and json.loads(
            (directory / "reporting-status.json").read_text()
        ).get("remote")
        if kind == "snapshot" and (
            remote
            or self.status(directory).get("state")
            in (
                "completed",
                "failed",
                "stopped",
                "killed",
                "prepared",
            )
        ):
            checkpoints = [
                e
                for e in events(directory)
                if e["kind"] == "checkpoint" and not e["payload"].get("remote")
            ]
            artifact = build_snapshot(
                directory,
                checkpoints[-1] if checkpoints else None,
                include_data and not remote,
                identifier,
            )
            return {**command, "artifact": artifact}
        return command


def build_snapshot(
    directory, checkpoint_event=None, include_data=False, identifier=None
):
    if checkpoint_event and checkpoint_event["payload"].get("remote"):
        raise ValueError(
            "Remote weights remain on the training computer; copy them to the server before backing them up"
        )
    identifier = identifier or uuid.uuid4().hex
    relative = f"snapshots/snapshot-{identifier}.zip"
    target = directory / relative
    target.parent.mkdir(exist_ok=True)
    lock = target.with_suffix(".lock")
    if (target.with_suffix(".json")).exists():
        return relative
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(lock_fd)
    except FileExistsError:
        return relative
    temporary = target.with_suffix(".partial")
    cutoff = time.time()
    checkpoint = checkpoint_event["payload"]["path"] if checkpoint_event else None
    manifest = {
        "version": 1,
        "created": time.time(),
        "checkpoint": checkpoint,
        "resumable_weights": checkpoint is not None,
        "includes_source_data": include_data,
        "files": {},
        "missing_data": [],
        "note": "Dataset hashes identify data; only included files preserve it. Custom loader resume guarantees depend on the script.",
    }

    def add(archive, name, content):
        archive.writestr(name, content)
        manifest["files"][name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    def add_path(archive, name, path):
        digest = hashlib.sha256()
        written = 0
        # Bound to the size observed at opening; logs may continue growing.
        remaining = path.stat().st_size
        with path.open("rb") as source, archive.open(
            name, "w", force_zip64=True
        ) as output:
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                remaining -= len(chunk)
        manifest["files"][name] = {"sha256": digest.hexdigest(), "bytes": written}

    try:
        if checkpoint_event:
            source = inside(directory, checkpoint)
            if file_hash(source) != checkpoint_event["payload"]["sha256"]:
                raise ValueError("Published checkpoint checksum changed")
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for path in sorted(directory.rglob("*")):
                name = path.relative_to(directory).as_posix()
                if not include_data and path.name.startswith("prepared-"):
                    continue
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or name.startswith(
                        (
                            "snapshots/",
                            "commands/",
                            "starting-weights/",
                            "source-data/",
                            "packages/",
                        )
                    )
                    or path.name
                    in (
                        "operator.token",
                        "manifest.json",
                        "reporting.json",
                        ".report-token",
                    )
                    or path.name.endswith((".pending", ".partial"))
                    or "event-index.sqlite" in path.name
                    or "__pycache__" in path.parts
                ):
                    continue
                if name.startswith("checkpoints/") and name != checkpoint:
                    continue
                if path.name == "events.jsonl":
                    continue
                if (
                    path.parent == directory
                    and path.name.startswith("consumed-rank-")
                    and checkpoint_event
                ):
                    with archive.open(name, "w", force_zip64=True) as output, path.open(
                        "rb"
                    ) as source:
                        digest = hashlib.sha256()
                        length = 0
                        for line in source:
                            try:
                                if (
                                    json.loads(line).get("step", 0)
                                    > checkpoint_event["step"]
                                ):
                                    continue
                            except ValueError:
                                continue
                            output.write(line)
                            digest.update(line)
                            length += len(line)
                        manifest["files"][name] = {
                            "sha256": digest.hexdigest(),
                            "bytes": length,
                        }
                else:
                    add_path(archive, name, path)
            selected = [
                event
                for event in events(directory)
                if event["time"] <= cutoff
                and (
                    not checkpoint_event
                    or event.get("step", 0) <= checkpoint_event["step"]
                )
            ]
            add(
                archive,
                "events.jsonl",
                ("\n".join(json.dumps(e) for e in selected) + "\n").encode(),
            )
            if include_data:
                sources = [
                    ("", directory / "datasets.json"),
                    *[
                        (f"lineage/{path.parent.name}/", path)
                        for path in (directory / "lineage").glob("*/datasets.json")
                    ],
                ]
                for scope, dataset_manifest in sources:
                    for key, info in json.loads(dataset_manifest.read_text()).items():
                        path = Path(info["path"])
                        if not path.is_file() or file_hash(path) != info["sha256"]:
                            manifest["missing_data"].append(
                                {
                                    "source": scope + key,
                                    "path": info["path"],
                                    "reason": "missing or changed",
                                }
                            )
                        else:
                            name = f"source-data/{scope}{key}/{path.name}"
                            add_path(archive, name, path)
                            if manifest["files"][name]["sha256"] != info["sha256"]:
                                raise ValueError(
                                    "Source data changed while making the snapshot"
                                )
            add(
                archive,
                "RESTORE.md",
                b"Import this archive into Weave, select its checkpoint, then choose Resume or Start new stage. Verify/reconnect dataset paths. Scripts are not executed during import. No checkpoint means this is a run record, not a weight backup. GPU kernels/custom loaders may prevent exact continuation.\n",
            )
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        os.replace(temporary, target)
    except BaseException:
        lock.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(
        directory / "snapshots" / f"snapshot-{identifier}.json",
        {
            "path": relative,
            "checkpoint": checkpoint,
            "sha256": file_hash(target),
            "bytes": target.stat().st_size,
            "missing_data": manifest["missing_data"],
        },
    )
    lock.unlink(missing_ok=True)
    return relative


def import_snapshot(service, file):
    with zipfile.ZipFile(file) as archive:
        infos = archive.infolist()
        if len(infos) > 50000 or sum(info.file_size for info in infos) > 100 * 1024**3:
            raise ValueError("Snapshot exceeds import limits (100 GB, 50,000 files).")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("version") != 1:
            raise ValueError("Unsupported snapshot version")
        if len({entry.filename for entry in infos}) != len(infos):
            raise ValueError("Duplicate archive paths")
        expected = set(manifest["files"]) | {"manifest.json"}
        if set(archive.namelist()) != expected:
            raise ValueError("Snapshot manifest does not match archive contents")
        run_id = "import-" + uuid.uuid4().hex
        directory = service.root / "runs" / run_id
        directory.mkdir()
        try:
            for name, info in manifest["files"].items():
                path = inside(directory, name)
                if name.startswith("/") or "\\" in name:
                    raise ValueError("Unsafe archive path")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if (
                    path.stat().st_size != info["bytes"]
                    or file_hash(path) != info["sha256"]
                ):
                    raise ValueError("Snapshot checksum failed: " + name)
            info = json.loads((directory / "run.json").read_text())
            info.update(
                id=run_id, imported=True, created=datetime.now(timezone.utc).isoformat()
            )
            atomic_json(directory / "run.json", info)
            atomic_json(
                directory / "status.json",
                {
                    "state": "completed" if manifest["checkpoint"] else "prepared",
                    "imported": True,
                },
            )
            # Reconnect bundled datasets without trusting paths from another host.
            datasets = json.loads((directory / "datasets.json").read_text())
            config = json.loads((directory / "config.json").read_text())
            for key, dataset in datasets.items():
                bundled = directory / "source-data" / key / Path(dataset["path"]).name
                if bundled.is_file():
                    dataset["path"] = str(bundled)
                    config[key] = str(bundled)
            atomic_json(directory / "datasets.json", datasets)
            atomic_json(directory / "config.json", config)
            return info
        except BaseException:
            shutil.rmtree(directory)
            raise
