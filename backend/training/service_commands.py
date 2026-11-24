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

from . import service as service_module
from .service import Launch, Stage, inside

class TrainingCommands:
    def complete_snapshots(self, directory):
        checkpoints = service_module.events(directory, kind="checkpoint")
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
                service_module.build_snapshot(
                    directory, matching, request["include_data"], request["id"]
                )

    def restart_context(self, run_id):
        """Find the last usable checkpoint, including before a failed resume."""
        visited = set()
        directory = self.run_dir(run_id)
        while directory.name not in visited:
            visited.add(directory.name)
            info = json.loads((directory / "run.json").read_text())
            checkpoint = next(
                (
                    event
                    for event in reversed(service_module.events(directory, kind="checkpoint"))
                    if not event["payload"].get("remote")
                ),
                None,
            )
            if checkpoint is not None:
                return {
                    "source_run": directory.name,
                    "directory": str(directory.resolve()),
                    "checkpoint": checkpoint["payload"]["path"],
                    "step": checkpoint["step"],
                    "exact_resume": checkpoint["payload"]
                    .get("metadata", {})
                    .get("exact_resume", False),
                    "edited_script_path": str(
                        self.workspaces.directory(info["project_id"], info["stage_id"])
                        / "train.py"
                    ),
                }
            parent = info.get("parent")
            if not parent:
                return None
            try:
                directory = self.run_dir(parent["run_id"])
            except FileNotFoundError:
                return None
        return None

    def restart(
        self,
        run_id,
        validation_settings=None,
        *,
        script_source="previous",
        refresh_support=False,
    ):
        """Snapshot each attempt. Failed restores can retry the last usable state."""
        if script_source not in ("previous", "edited"):
            raise ValueError("Choose the previous or edited script")
        with self.controls_lock:
            requested_source = self.run_dir(run_id)
            info = json.loads((requested_source / "run.json").read_text())
            context = self.restart_context(run_id)
            capability_source = (
                self.run_dir(context["source_run"]) if context else requested_source
            )
            script_caps = next(
                (
                    event["payload"]
                    for event in reversed(
                        service_module.events(capability_source, kind="capabilities")
                    )
                    if event.get("channel") == "training"
                ),
                {},
            )
            custom_resume = (
                info.get("recipe") == "custom" and script_caps.get("resume") is True
            )
            if info["execution"] != "local" or not (
                info.get("code_first") or custom_resume
            ):
                raise ValueError(
                    "Checkpoint restart requires a local Trainer run or a custom script declaring resume support."
                )
            if self.status(requested_source)["state"] not in (
                "stopped",
                "failed",
                "killed",
                "completed",
            ):
                raise ValueError(
                    "Stop training safely before restarting from its checkpoint."
                )
            receipt = requested_source / "restart.json"
            if receipt.is_file():
                previous = json.loads(receipt.read_text())
                child = self.run_dir(previous["run_id"])
                # A repeated click must not launch a second worker. A failed
                # attempt must never prevent fixing the script and trying again.
                if self.status(child)["state"] not in ("failed", "killed", "stopped"):
                    return json.loads((child / "run.json").read_text())
            if not context or not context["exact_resume"]:
                raise ValueError(
                    "No checkpoint with restorable optimizer and data-loader state is available. Create a new stage from weights instead."
                )
            source = self.run_dir(context["source_run"])
            project = json.loads((source / PACKAGE / "project.json").read_text())
            stage = Stage(**json.loads((source / "stage.json").read_text()))
            stage.script = (source / "train.py").read_text()
            if script_source == "edited":
                with self.workspaces.lock:
                    edited = self.workspaces.get_stage(info["project_id"], stage.id)
                    stage.script = edited.script
            stage.config = stage.config.model_copy(
                update={
                    "execution": "local",
                    "start_mode": "resume",
                    "parent_run": source.name,
                    "parent_checkpoint": context["checkpoint"],
                }
            )
            if validation_settings is None and not custom_resume:
                from .run_controls import validation_state

                validation_settings = validation_state(
                    requested_source,
                    service_module.events(requested_source),
                    self.status(requested_source),
                )["current"]
            attempt_id = uuid.uuid4().hex
            attempt = requested_source / "resume-attempts" / attempt_id
            attempt.mkdir(parents=True)
            (attempt / "previous.py").write_text((source / "train.py").read_text())
            (attempt / "requested.py").write_text(stage.script)
            atomic_json(attempt / "stage.json", stage.model_dump())
            user_files = []
            if script_source == "edited":
                user_files = self.workspaces.freeze_extras(
                    info["project_id"], stage.id, attempt / "user-files"
                )
            record = {
                "attempt_id": attempt_id,
                "requested_from": run_id,
                "source_run": source.name,
                "script_source": script_source,
                "refresh_support": refresh_support,
                "script_sha256": hashlib.sha256(stage.script.encode()).hexdigest(),
                "user_files": {
                    name: file_hash(inside(attempt / "user-files", name))
                    for name in user_files
                },

                "validation": validation_settings,
                "checkpoint": context["checkpoint"],
                "created": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(attempt / "attempt.json", record)
            try:
                restarted = self.launch(
                    Launch(project=project, stage=stage),
                    frozen_source=source,
                    validation_settings=validation_settings,
                    restart_record=record,
                    edited_files_source=attempt / "user-files",
                )
            except Exception as exc:
                atomic_json(
                    attempt / "attempt.json",
                    {**record, "state": "rejected", "error": str(exc)},
                )
                if isinstance(exc, SyntaxError):
                    raise ValueError(
                        f"Edited script has a syntax error at line {exc.lineno}: {exc.msg}. Fix the file and retry, or select the previous script."
                    ) from exc
                raise
            atomic_json(
                attempt / "attempt.json",
                {**record, "state": "launched", "run_id": restarted["id"]},
            )
            atomic_json(
                receipt,
                {
                    "run_id": restarted["id"],
                    "checkpoint": context["checkpoint"],
                    "attempt_id": attempt_id,
                },
            )
            return restarted

    def set_validation(self, run_id, settings):
        from .run_controls import validation_state

        with self.controls_lock:
            directory = self.run_dir(run_id)
            state = validation_state(
                directory, service_module.events(directory), self.status(directory)
            )
            if not state["supported"]:
                raise ValueError(
                    "Restart from a checkpoint to enable live validation controls for this frozen trainer."
                )
            if not state["active"]:
                raise ValueError(
                    "Training is not running; choose validation settings when restarting."
                )
            if state["pending"]:
                raise ValueError(
                    "Wait for the trainer to acknowledge the pending validation change."
                )
            command = self.command(run_id, "validation", settings=settings)
            atomic_json(directory / "validation-request.json", command)
            return command

    def command(self, run_id, kind, include_data=False, *, settings=None):
        directory = self.run_dir(run_id)
        remote = (directory / "reporting-status.json").exists() and json.loads(
            (directory / "reporting-status.json").read_text()
        ).get("remote")
        active = self.status(directory)["state"] in (
            "starting",
            "running",
            "preparing",
            "evaluating",
        )
        capability = next(
            iter(service_module.events(directory, kind="capabilities", channel="training", limit=1)),
            {},
        ).get("payload", {})
        if active and kind == "stop" and capability.get("commands") is False:
            raise ValueError(
                "This script has not declared cooperative stop support. Use Force stop for a locally launched process."
            )
        if (
            active
            and kind == "snapshot"
            and not remote
            and (
                capability.get("commands") is False
                or capability.get("checkpoint") is False
            )
        ):
            raise ValueError(
                "This script has not declared checkpoint/snapshot handling. Stop the run before exporting its recorded files."
            )
        identifier = uuid.uuid4().hex
        command = {
            "id": identifier,
            "kind": kind,
            "include_data": include_data,
            "time": time.time(),
        }
        if settings is not None:
            command["settings"] = settings
        atomic_json(directory / "commands" / (identifier + ".json"), command)
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
                for e in service_module.events(directory)
                if e["kind"] == "checkpoint" and not e["payload"].get("remote")
            ]
            artifact = service_module.build_snapshot(
                directory,
                checkpoints[-1] if checkpoints else None,
                include_data and not remote,
                identifier,
            )
            return {**command, "artifact": artifact}
        return command
