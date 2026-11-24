"""Authenticated, same-origin local training API."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sqlite3
import stat
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
import zipfile
from contextlib import asynccontextmanager

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    File,
    Query,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from .service import (
    TrainingService,
    TrainingConfig,
    Stage,
    Plan,
    Launch,
    inside,
    valid_id,
    events,
    hardware,
    index_busy,
    model_metadata,
    import_snapshot,
)
from .workspaces import WorkspaceConflict
from .environment import inspect_python
from .reporting import ReportStore, ReportBatch
from .layout import PACKAGE, normalize_script
from .recipe import is_code_first, starter_script
from .stage_settings import preview_stage_settings
from .chat import (
    ChatSessions,
    ChatPreview,
    ChatTestSuite,
    ChatToolResult,
    ChatLoad,
    ChatMessage,
    ChatOptions,
    checkpoint_catalog,
    decoder_catalog,
    option_diagnostics,
    configured_decoders,
)
from .chat_protocol import presets
from .gpu_tuning import GPUTuning, TuningRequest
from .run_controls import ValidationSettings, validation_state
from ..paths import projects_directory
from ..compiler import compile_project


from .api import Command, RestartRequest


def register(api, service, authenticated, run_directory):

    def validate_validation_device(directory, settings):
        if settings.mode == "inline" or settings.device == "cpu":
            return
        environment = directory / "environment.json"
        if not environment.is_file():
            raise ValueError(
                "GPU information is not available yet; wait for trainer startup or choose CPU."
            )
        info = json.loads(environment.read_text())
        if not info.get("cuda_available") or settings.device >= len(
            info.get("devices", [])
        ):
            raise ValueError(
                "The selected validation GPU is unavailable in this run's Python environment."
            )

    @api.put("/runs/{run_id}/validation", dependencies=[Depends(authenticated)])
    def update_validation(run_id: str, body: ValidationSettings):
        try:
            validate_validation_device(run_directory(run_id), body)
            return service.set_validation(run_id, body.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.post("/runs/{run_id}/restart", dependencies=[Depends(authenticated)])
    def restart_training(run_id: str, body: RestartRequest):
        if os.environ.get("WEAVE_TRAINING_ENABLED") != "1":
            raise HTTPException(
                403, "Enable local training execution before restarting."
            )
        try:
            if body.validation:
                validate_validation_device(run_directory(run_id), body.validation)
            return service.restart(
                run_id,
                body.validation.model_dump() if body.validation else None,
                script_source=body.script_source,
                refresh_support=body.refresh_support,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))

    @api.post("/runs/{run_id}/commands", dependencies=[Depends(authenticated)])
    def command(run_id: str, body: Command):
        if body.kind not in ("stop", "kill", "snapshot"):
            raise HTTPException(422, "Unsupported command")
        directory = run_directory(run_id)
        if (
            body.kind == "kill"
            and json.loads((directory / "run.json").read_text()).get("execution")
            == "monitor"
        ):
            raise HTTPException(
                422,
                "Force-stop terminal jobs from their terminal. Weave does not own that process.",
            )
        try:
            return service.command(run_id, body.kind, body.include_data)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.get("/runs/{run_id}/artifact", dependencies=[Depends(authenticated)])
    def artifact(run_id: str, path: str):
        directory = run_directory(run_id)
        try:
            target = inside(directory, path)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        if not target.is_file() or target.is_symlink():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(target, filename=target.name)

    @api.get("/runs/{run_id}/package", dependencies=[Depends(authenticated)])
    def package(run_id: str):
        directory = run_directory(run_id)
        import uuid

        destination = directory / "packages" / ("training-" + uuid.uuid4().hex + ".zip")
        destination.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            extras_path = directory / ".workspace-files.json"
            code_first = is_code_first((directory / "train.py").read_text())
            editable = ["train.py", "reporting.json"] + (
                [] if code_first else ["config.json"]
            )
            if extras_path.exists():
                editable.extend(json.loads(extras_path.read_text()))
            for name in editable:
                path = inside(directory, name)
                if not path.is_file() or path.is_symlink():
                    continue
                if name == "config.json":
                    config = json.loads(path.read_text())
                    if (directory / "starting-weights/parent.pt").is_file():
                        config["checkpoint"] = "starting-weights/parent.pt"
                    archive.writestr(name, json.dumps(config, indent=2))
                else:
                    archive.write(path, name)
            for folder in (PACKAGE, "starting-weights"):
                for path in (directory / folder).rglob("*"):
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and "__pycache__" not in path.parts
                        and "migrations" not in path.parts
                    ):
                        name = str(path.relative_to(directory))
                        if name == PACKAGE + "/launch.json":
                            launch = json.loads(path.read_text())
                            if (directory / "starting-weights/parent.pt").is_file():
                                launch["checkpoint"] = "starting-weights/parent.pt"
                            archive.writestr(name, json.dumps(launch, indent=2))
                        else:
                            archive.write(path, name)
            lock = directory / "requirements-lock.txt"
            if lock.is_file():
                archive.write(lock, PACKAGE + "/requirements-lock.txt")
            archive.writestr(
                "README.md",
                "# Training project\n\nEdit train.py and your own helpers. Code-owned recipes define TrainingConfig and hooks directly in Python; run python train.py. Legacy recipes use config.json and python train.py --config config.json. weave_support/ contains generated network code, tokenizers, launch context, reporting and the optional trainer; it is refreshed by Weave.\n\nInstall dependencies in your chosen Python environment: pip install -r weave_support/requirements.txt (select a suitable PyTorch CUDA wheel for your GPU). reporting.json supplies the optional reporting connection. Source paths/size/mtime are recorded by default; SHA256 fingerprints are opt-in. Consumed batch traces identify source records when the loader supplies _weave_source metadata.\n",
            )
        return FileResponse(
            destination, filename=run_id + "-training.zip", media_type="application/zip"
        )

    @api.post("/import", dependencies=[Depends(authenticated)])
    def import_archive(file: UploadFile = File(...)):
        try:
            return import_snapshot(service, file.file)
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise HTTPException(422, str(exc))
