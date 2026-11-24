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




def register(api, authenticated, chat_session, run_directory):

    @api.get("/runs/{run_id}/chat", dependencies=[Depends(authenticated)])
    def get_chat(run_id: str):
        directory = run_directory(run_id)
        chat = chat_session(run_id)
        environment_path = directory / "environment.json"
        environment = (
            json.loads(environment_path.read_text())
            if environment_path.is_file()
            else {}
        )
        return {
            **checkpoint_catalog(directory),
            "session": chat.status(run_id),
            "decoders": decoder_catalog(),
            "presets": presets(),
            "preferences": chat.preferences(),
            "supports_cuda": bool(environment.get("cuda_available")),
        }

    @api.post(
        "/runs/{run_id}/chat/validate-options", dependencies=[Depends(authenticated)]
    )
    def validate_chat_options(run_id: str, body: ChatOptions):
        run_directory(run_id)
        try:
            return option_diagnostics(body.model_dump(), configured_decoders())
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.post("/runs/{run_id}/chat/preview", dependencies=[Depends(authenticated)])
    def preview_chat(run_id: str, body: ChatPreview):
        try:
            return chat_session(run_id).preview(run_id, body)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.post("/runs/{run_id}/chat/tests", dependencies=[Depends(authenticated)])
    def test_chat(run_id: str, body: ChatTestSuite):
        try:
            return chat_session(run_id).run_tests(run_id, body)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.post("/runs/{run_id}/chat/tool-result", dependencies=[Depends(authenticated)])
    def tool_result_chat(run_id: str, body: ChatToolResult):
        try:
            return chat_session(run_id).inject_result(run_id, body)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.get("/runs/{run_id}/chat/export", dependencies=[Depends(authenticated)])
    def export_chat(run_id: str):
        session = chat_session(run_id).status(run_id)
        return Response(
            content=json.dumps({"version": 1, "run_id": run_id, **session}) + "\n",
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": 'attachment; filename="chat-transcript.jsonl"'
            },
        )

    @api.put("/runs/{run_id}/chat/settings", dependencies=[Depends(authenticated)])
    def configure_chat(run_id: str, body: ChatOptions):
        run_directory(run_id)
        try:
            return chat_session(run_id).configure(run_id, body)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.post("/runs/{run_id}/chat/load", dependencies=[Depends(authenticated)])
    def load_chat(run_id: str, body: ChatLoad):
        if os.environ.get("WEAVE_TRAINING_ENABLED") != "1":
            raise HTTPException(
                403, "Enable local execution to load checkpoint chat on this server."
            )
        try:
            return chat_session(run_id).load(run_directory(run_id), body)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.get("/runs/{run_id}/chat/session", dependencies=[Depends(authenticated)])
    def get_chat_session(run_id: str):
        run_directory(run_id)
        return chat_session(run_id).status(run_id)

    @api.post("/runs/{run_id}/chat/messages", dependencies=[Depends(authenticated)])
    def send_chat(run_id: str, body: ChatMessage):
        run_directory(run_id)
        try:
            return chat_session(run_id).send(run_id, body)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.post("/runs/{run_id}/chat/stop", dependencies=[Depends(authenticated)])
    def stop_chat(run_id: str):
        run_directory(run_id)
        return chat_session(run_id).stop(run_id)

    @api.post("/runs/{run_id}/chat/clear", dependencies=[Depends(authenticated)])
    def clear_chat(run_id: str):
        run_directory(run_id)
        try:
            return chat_session(run_id).clear(run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @api.delete("/runs/{run_id}/chat", dependencies=[Depends(authenticated)])
    def unload_chat(run_id: str):
        chat = chat_session(run_id)
        chat.close(run_id, preserve=True)
        return chat.status(run_id)

    @api.get("/runs/{run_id}/events", dependencies=[Depends(authenticated)])
    async def stream(run_id: str, request: Request):
        directory = run_directory(run_id)
        last = request.headers.get("last-event-id") or request.query_params.get("after")

        async def generate():
            cursor = last or ""
            while not await request.is_disconnected():
                batch = await asyncio.to_thread(
                    events, directory, after=cursor, limit=1000
                )
                for event in batch:
                    cursor = event["id"]
                    yield "id: " + event["id"] + "\ndata: " + json.dumps(event) + "\n\n"
                yield ": heartbeat\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
