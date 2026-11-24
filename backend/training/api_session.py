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


from .api import Login


def register(api, service, tuning, authenticated, same_origin):
    @api.get("/gpu-tuning", dependencies=[Depends(authenticated)])
    def tuning_list():
        return tuning.list()

    @api.put("/gpu-tuning/settings", dependencies=[Depends(authenticated)])
    def tuning_settings(body: dict):
        if type(body.get("auto_apply")) is not bool:
            raise HTTPException(422, "auto_apply must be true or false")
        return tuning.configure(body["auto_apply"])

    @api.post("/gpu-tuning", dependencies=[Depends(authenticated)])
    def tuning_start(body: TuningRequest):
        if os.environ.get("WEAVE_TRAINING_ENABLED") != "1":
            raise HTTPException(
                403, "Enable local training before benchmarking server GPUs"
            )
        try:
            return tuning.start(body)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc))

    @api.get("/gpu-tuning/{identifier}", dependencies=[Depends(authenticated)])
    def tuning_status(identifier: str):
        try:
            return tuning.status(identifier)
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @api.post("/gpu-tuning/{identifier}/stop", dependencies=[Depends(authenticated)])
    def tuning_stop(identifier: str):
        try:
            return tuning.stop(identifier)
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @api.get("/capabilities")
    def capabilities(request: Request):
        credential = request.cookies.get("weave_training", "")
        return {
            "enabled": os.environ.get("WEAVE_TRAINING_ENABLED") == "1",
            "authenticated": not service.token
            or hmac.compare_digest(credential, service.token),
            "auth_required": bool(service.token),
            "python": os.environ.get("WEAVE_TRAINING_PYTHON", os.sys.executable),
            "token_file": None,
            "remote_launch": False,
            "projects_directory": str(projects_directory()),
        }

    @api.post("/session")
    def login(body: Login, request: Request, response: Response):
        same_origin(request)
        if service.token and not hmac.compare_digest(body.token, service.token):
            raise HTTPException(401, "Invalid training access code")
        response.set_cookie(
            "weave_training",
            service.token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=86400,
        )
        return {"authenticated": True}

    @api.delete("/session")
    def logout(response: Response):
        response.delete_cookie("weave_training")
        return {"authenticated": False}
