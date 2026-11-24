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


MONITOR_INDEX_TIMEOUT = 0.5


def monitor_index_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="The training event index is busy. Retry shortly.",
        headers={"Retry-After": "3"},
    )


class Login(BaseModel):
    token: str


class Command(BaseModel):
    kind: str
    include_data: bool = False


class RestartRequest(BaseModel):
    validation: ValidationSettings | None = None
    script_source: Literal["previous", "edited"] = "previous"
    refresh_support: bool = False


def directory_size(directory):
    """A live run may atomically replace/delete files while it is inspected."""
    total = 0
    for path in directory.rglob("*"):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def router(data):
    service = TrainingService(Path(data) / "training", Path(data) / "tokenizers")
    chats = ChatSessions()
    tuning = GPUTuning(Path(data) / "training")

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            tuning.close()
            chats.close()

    api = APIRouter(prefix="/api/training", lifespan=lifespan)
    reports = ReportStore()

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Training requests must come from this Weave host")

    def authenticated(request: Request):
        same_origin(request)
        if not service.token:
            return
        credential = request.cookies.get("weave_training") or request.headers.get(
            "x-weave-training-token", ""
        )
        if not hmac.compare_digest(credential, service.token):
            raise HTTPException(401, "Unlock training with the operator access code")

    def run_directory(run_id):
        try:
            return service.run_dir(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    def chat_session(run_id):
        directory = run_directory(run_id)
        project_id = json.loads((directory / "run.json").read_text())["project_id"]
        preferences_path = (
            service.workspaces.project_directory(project_id) / "chat-settings.json"
        )
        return chats.for_run(run_id, preferences_path)

    def safe_id(value):
        try:
            return valid_id(value)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    from .api_session import register as register_session
    from .api_stages import register as register_stages
    from .api_runs import register as register_runs
    from .api_chat import register as register_chat
    from .api_run_actions import register as register_run_actions

    register_session(api, service, tuning, authenticated, same_origin)
    register_stages(api, service, authenticated, safe_id)
    register_runs(api, service, reports, authenticated, run_directory)
    register_chat(api, authenticated, chat_session, run_directory)
    register_run_actions(api, service, authenticated, run_directory)
    return api
