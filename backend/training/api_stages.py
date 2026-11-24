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


from . import api as api_module


def register(api, service, authenticated, safe_id):
    @api.get("/templates", dependencies=[Depends(authenticated)])
    def templates():
        custom = '''"""Custom training: keep your own data, loss and optimizer logic.

For your real training loop, the minimal reporting integration is:
    with Run(optimizer=optimizer, total_steps=total_steps) as run:
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)  # Your objective.
            loss.backward()
            optimizer.step()
            run.log_loss(loss)

Steps, learning rates and progress are automatic. This helper has no HTTP
code for you to maintain and needs no pip-installed Weave package.
The executable demonstration below only reports events; replace it with
actual training code when your model/data/optimizer are ready.
"""\nfrom weave_training import Run\n\nWEAVE_SUPPORTED_SETTINGS = ()  # Add fields only after your loop consumes launch_settings().\n\nwith Run() as run:\n    run.capabilities(metrics=True, samples=False, checkpoint=False, commands=True, resume=False)\n    run.configure({}, metrics={"example/progress": {"label": "Example progress"}}, step_unit="updates")\n    # Replace this demonstration with your own PyTorch training loop.\n    for step in range(10):\n        commands = run.poll_commands()\n        if any(command["kind"] == "stop" for command in commands):\n            break\n        run.log({"example/progress": step}, step=step)\n        run.progress(completed=step + 1, total=10)\n    run.warning("This starter reported events only. Add your model, loss, optimizer, and checkpoint save logic.")\n'''
        return {
            "config": TrainingConfig().model_dump(),
            "scripts": {
                "causal": starter_script("causal"),
                "sft": starter_script("sft"),
                "custom": normalize_script(custom, "custom"),
            },
        }

    @api.post("/datasets", dependencies=[Depends(authenticated)])
    def upload_dataset(file: UploadFile = File(...)):
        from .weave_training import file_hash
        import uuid

        name = Path(file.filename or "data.jsonl").name
        if Path(name).suffix not in (".txt", ".json", ".jsonl", ".parquet"):
            raise HTTPException(422, "Upload TXT, JSON, JSONL or Parquet data")
        target = service.root / "datasets" / uuid.uuid4().hex / name
        target.parent.mkdir(parents=True)
        total = 0
        try:
            with target.open("wb") as output:
                for chunk in iter(lambda: file.file.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise HTTPException(
                            413,
                            "Browser uploads are limited to 512 MB; use a host path for larger datasets",
                        )
                    output.write(chunk)
            return {"path": str(target), "sha256": file_hash(target), "bytes": total}
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    @api.post("/preview", dependencies=[Depends(authenticated)])
    def preview(body: Launch):
        from .template_data import preview_records

        try:
            return preview_records(
                body.stage.config, body.project, service.tokenizer_root
            )
        except (ValueError, KeyError, OSError) as exc:
            raise HTTPException(422, str(exc))

    @api.get("/hardware", dependencies=[Depends(authenticated)])
    def get_hardware():
        return {"gpus": hardware(), "scope": "device-wide"}

    @api.post("/model", dependencies=[Depends(authenticated)])
    def get_model(project: dict):
        try:

            return model_metadata(compile_project(project), project)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc))

    @api.post("/settings/preview", dependencies=[Depends(authenticated)])
    def settings_preview(body: Stage):
        """Preview the next launch's device settings without executing the script."""
        return preview_stage_settings(body)

    @api.get("/plans/{project_id}", dependencies=[Depends(authenticated)])
    def get_plan(project_id: str):
        try:
            return service.workspaces.load_plan(safe_id(project_id))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.put("/plans/{project_id}", dependencies=[Depends(authenticated)])
    def put_plan(project_id: str, plan: Plan):
        if plan.project_id != project_id or plan.version != 1:
            raise HTTPException(422, "Training plan identity/version mismatch")
        if len({stage.id for stage in plan.stages}) != len(plan.stages):
            raise HTTPException(422, "Stage IDs must be unique")
        try:
            return service.workspaces.save_plan(plan)
        except WorkspaceConflict as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.get("/environment", dependencies=[Depends(authenticated)])
    def environment(python: str):
        if (
            os.environ.get("WEAVE_TRAINING_ENABLED") != "1"
            and python != os.sys.executable
        ):
            raise HTTPException(
                403,
                "Enable local execution before inspecting another Python environment",
            )
        try:
            return api_module.inspect_python(python)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.get(
        "/workspaces/{project_id}/stages/{stage_id}",
        dependencies=[Depends(authenticated)],
    )
    def get_stage_files(project_id: str, stage_id: str):
        try:
            return service.workspaces.get_stage(safe_id(project_id), safe_id(stage_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.put(
        "/workspaces/{project_id}/stages/{stage_id}",
        dependencies=[Depends(authenticated)],
    )
    def save_stage_files(project_id: str, stage_id: str, body: Launch):
        if body.project.get("id") != project_id or body.stage.id != stage_id:
            raise HTTPException(
                422, "Stage and project identities must match their server paths"
            )
        try:
            saved = service.workspaces.save_stage(project_id, body.stage)
            try:
                service.workspaces.prepare_model(
                    body.project, saved, service.tokenizer_root
                )
            except (ValueError, OSError) as exc:
                saved.workspace_warning = (
                    "Script saved, but the stage model/tokenizer could not be synchronized: "
                    + str(exc)
                )
            # Generating a previously unsaved graph can assign its readable folder.
            # Return paths after that move rather than the earlier ID-based paths.
            refreshed = service.workspaces.read(body.project["id"], saved)
            refreshed.workspace_warning = saved.workspace_warning
            saved = refreshed
            return saved
        except WorkspaceConflict as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
