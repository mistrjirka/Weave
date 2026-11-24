"""Training documents, immutable run exports, and verified portable archives."""

from __future__ import annotations

import ast
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import sys
import time
import uuid
import zipfile
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict
from ..compiler import export_zip
from ..core import validate_project
from .weave_training import (
    atomic_json,
    file_hash,
    instrumentation_warnings,
    available_cpu_threads,
)
from .layout import PACKAGE, normalize_script, write_support
from .recipe import is_code_first, literal_settings

HERE = Path(__file__).parent


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: str = Field(default="causal", pattern="^(causal|sft|custom)$")
    strategy: str = Field(default="single", pattern="^(single|ddp|split|train_eval)$")
    execution: str = Field(default="local", pattern="^(local|monitor)$")
    python: str = sys.executable
    devices: list[int] = Field(default_factory=list, max_length=32)
    evaluation_device: int | Literal["cpu"] | None = None
    evaluation_mode: Literal["script", "inline", "async"] = "script"
    compute_mode: Literal["script", "override"] = "script"
    performance_mode: Literal["script", "override"] = "script"
    compile_mode: Literal["off", "default", "reduce-overhead", "max-autotune"] = "off"
    fused_optimizer: bool = False
    profile_steps: int = Field(default=0, ge=0, le=100, strict=True)
    placement: dict[str, int] = Field(default_factory=dict)
    use_gpu_tuning: bool = True
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
    cpu_threads: int = Field(default_factory=available_cpu_threads, ge=1)
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
            or type(self.evaluation_device) is not int
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


def index_busy(exc: sqlite3.OperationalError) -> bool:
    return (getattr(exc, "sqlite_errorcode", 0) & 0xFF) in (
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    )


from .service_index import events, hardware, model_metadata
from .service_launch import TrainingLaunch
from .service_commands import TrainingCommands
from .service_snapshots import build_snapshot, import_snapshot

class TrainingService(TrainingLaunch, TrainingCommands):
    def __init__(self, root, tokenizer_root):
        self.root = Path(root)
        self.tokenizer_root = Path(tokenizer_root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "plans").mkdir(exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        from .workspaces import WorkspaceStore

        self.workspaces = WorkspaceStore(self.root)
        self.token_path = self.root / "operator.token"
        self.controls_lock = threading.RLock()

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

    def active_runs(self):
        """Small cross-project summaries; no metrics, hardware or Python probes."""
        result = []
        active = {"starting", "preparing", "running", "evaluating"}
        for path in (self.root / "runs").glob("*/run.json"):
            try:
                status_path = path.parent / "status.json"
                # The supervisor's state is enough for this navbar summary.
                # Avoid indexing every validation log just to count local jobs.
                status = (
                    json.loads(status_path.read_text())
                    if status_path.is_file()
                    else self.status(path.parent)
                )
                if status.get("state") not in active:
                    continue
                info = json.loads(path.read_text())
                project_path = path.parent / PACKAGE / "project.json"
                project = (
                    json.loads(project_path.read_text())
                    if project_path.is_file()
                    else {}
                )
                result.append(
                    {
                        "id": info["id"],
                        "project_id": info["project_id"],
                        "project_name": project.get("name", "Project"),
                        "stage_id": info["stage_id"],
                        "name": info["name"],
                        "created": info["created"],
                        "state": status["state"],
                    }
                )
            except FileNotFoundError:
                continue  # A run can be removed while the list is being read.
        return sorted(result, key=lambda item: item["created"], reverse=True)

    def status(self, directory, *, index_timeout=30):
        path = directory / "status.json"
        status = json.loads(path.read_text()) if path.exists() else None
        if status and status["state"] not in ("starting", "running"):
            return status
        preparation = events(
            directory, kind="preparation", channel="system", limit=1,
            timeout=index_timeout,
        )
        progress = (
            preparation[-1]["payload"]
            if preparation and preparation[-1].get("channel") == "system"
            else None
        )

        def preparing():
            if progress and progress.get("phase") == "failed":
                return {
                    "state": "failed",
                    "message": progress.get("message", "Data preparation failed"),
                }
            return {
                "state": "preparing",
                "message": (
                    "Data ready; initializing training."
                    if progress and progress.get("phase") in ("complete", "reused")
                    else "Preparing tokenizer and data; optimizer updates have not started yet."
                ),
            }

        if status is not None:
            if (
                status["state"] in ("starting", "running")
                and progress
                and not status.get("metrics_received")
            ):
                status.update(preparing())
            return status
        info = json.loads((directory / "run.json").read_text())
        if info.get("execution") == "monitor":
            starts = events(
                directory, kind="started", channel="training", limit=1,
                timeout=index_timeout,
            )
            start_id = starts[-1]["id"] if starts else None
            stopped = events(
                directory, kind="stopped", channel="training", after=start_id,
                limit=1, timeout=index_timeout,
            )
            if stopped:
                return {"state": "stopped"}
            finished = events(
                directory, kind="finished", channel="training", after=start_id,
                limit=1, latest=True, timeout=index_timeout,
            )
            if finished:
                return {
                    "state": (
                        "completed"
                        if finished[-1]["payload"].get("success")
                        else "failed"
                    )
                }
            if starts:
                return {
                    "state": "running",
                    "message": "Terminal-launched process; liveness is based on reported events.",
                }
        return preparing() if progress else {"state": "prepared"}
