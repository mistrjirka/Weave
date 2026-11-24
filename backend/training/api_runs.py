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


from .api import MONITOR_INDEX_TIMEOUT, directory_size, monitor_index_unavailable


def register(api, service, reports, authenticated, run_directory):

    @api.get("/runs", dependencies=[Depends(authenticated)])
    def get_runs(project_id: str):
        return service.list_runs(project_id)

    @api.get("/active-runs", dependencies=[Depends(authenticated)])
    def get_active_runs():
        return service.active_runs()

    @api.post("/runs", dependencies=[Depends(authenticated)])
    def launch(body: Launch, request: Request):
        if (
            body.stage.config.execution == "local"
            and os.environ.get("WEAVE_TRAINING_ENABLED") != "1"
        ):
            raise HTTPException(
                403,
                "Local execution is disabled. Run bash scripts/install.sh --training cpu, or configure an existing training Python, then restart Weave.",
            )
        try:
            service.workspaces.ensure_model(
                body.project, body.stage, service.tokenizer_root
            )
            try:
                disk_stage = service.workspaces.get_stage(
                    body.project["id"], body.stage.id
                )
                if disk_stage.workspace_warning:
                    raise ValueError(disk_stage.workspace_warning)
            except FileNotFoundError:
                pass
            body.stage = service.workspaces.save_stage(body.project["id"], body.stage)
            if body.stage.workspace_warning:
                raise ValueError(body.stage.workspace_warning)
            result = service.launch(body)
            if body.stage.config.execution == "monitor":
                reports.connection(
                    service.run_dir(result["id"]),
                    f"http://localhost:{request.url.port or 8173}",
                )
            return result
        except WorkspaceConflict as exc:
            raise HTTPException(409, str(exc))
        except (ValueError, FileNotFoundError, SyntaxError) as exc:
            raise HTTPException(422, str(exc))

    @api.post("/runs/{run_id}/reporting", dependencies=[Depends(authenticated)])
    def reporting_connection(run_id: str, request: Request, body: dict):
        directory = run_directory(run_id)
        if json.loads((directory / "run.json").read_text())["execution"] != "monitor":
            raise HTTPException(
                422,
                "Prepare a terminal / external run to connect an independent script",
            )
        base_url = (
            body.get("base_url") or f"http://localhost:{request.url.port or 8173}"
        )
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise HTTPException(422, "Use an HTTP or HTTPS reporting server URL")
        return reports.connection(directory, base_url)

    @api.post("/reports/{run_id}")
    async def ingest_reports(run_id: str, request: Request):
        # Scoped key permits reporting only, never execution or project editing.
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > 2 * 1024 * 1024:
                raise HTTPException(413, "Report batch exceeds 2 MB")
        try:
            batch = await asyncio.to_thread(ReportBatch.model_validate_json, content)
            return await asyncio.to_thread(
                reports.ingest,
                run_directory(run_id),
                request.headers.get("x-weave-report-token", ""),
                batch,
            )
        except PermissionError as exc:
            raise HTTPException(401, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.get("/runs/{run_id}/event-records", dependencies=[Depends(authenticated)])
    def event_records(
        run_id: str,
        after: str = Query(default="", max_length=128),
        kind: str | None = Query(default=None, max_length=64),
        limit: int = Query(default=100, ge=1, le=500),
        latest: bool = False,
    ):
        """Read chronological pages, or the newest bounded page after the cursor."""
        try:
            records = events(
                run_directory(run_id),
                after=after,
                kind=kind,
                limit=limit,
                latest=latest,
                # JSON re-encoding can expand a 1 MiB source line substantially.
                # Leave room for one event so a large record cannot wedge a cursor.
                max_bytes=8_000_000,
                timeout=MONITOR_INDEX_TIMEOUT,
            )
        except sqlite3.OperationalError as exc:
            if index_busy(exc):
                raise monitor_index_unavailable() from exc
            raise
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {
            "events": records,
            "next_after": records[-1]["id"] if records else after,
        }

    @api.get("/runs/{run_id}/status", dependencies=[Depends(authenticated)])
    def run_status(run_id: str, compact: bool = False):
        """Lightweight progress without loading full logs, samples or snapshots."""
        directory = run_directory(run_id)
        if compact:
            try:
                status = service.status(directory, index_timeout=MONITOR_INDEX_TIMEOUT)
            except sqlite3.OperationalError as exc:
                if index_busy(exc):
                    raise monitor_index_unavailable() from exc
                raise
            return {"run_id": run_id, "status": status}
        latest = events(directory, limit=1)
        try:
            metrics = events(directory, kind="metrics", limit=1, max_bytes=8_000_000)
            progress = events(directory, kind="progress", limit=1, max_bytes=8_000_000)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {
            "run_id": run_id,
            "status": service.status(directory),
            "latest_metrics": metrics[-1] if metrics else None,
            "progress": progress[-1] if progress else None,
            "last_event_id": latest[-1]["id"] if latest else "",
        }

    @api.get("/runs/{run_id}", dependencies=[Depends(authenticated)])
    def get_run(run_id: str):
        directory = run_directory(run_id)
        service.complete_snapshots(directory)
        records = events(directory, limit=10000)
        first_event = events(directory, after="", limit=1)
        retained_ids = {event["id"] for event in records}
        metadata = [
            event
            for kind in ("configuration", "capabilities")
            for event in events(directory, kind=kind, channel="training", limit=1)
            if event["id"] not in retained_ids
        ]
        records = metadata + records
        metrics = [event for event in records if event["kind"] == "metrics"]
        capabilities = [
            event["payload"]
            for event in records
            if event["kind"] == "capabilities" and event.get("channel") == "training"
        ]
        status = service.status(directory)
        diagnostics = []
        if not metrics:
            diagnostics.append(
                "No training metrics received yet. The script may be preparing data or missing a logging hook."
            )
        if not any(e["kind"] == "sample" for e in records) and (
            not capabilities or capabilities[-1].get("samples") is not False
        ):
            configuration = next(
                (
                    e["payload"]
                    for e in reversed(records)
                    if e["kind"] == "configuration"
                ),
                {},
            )
            interval = configuration.get("eval_every")
            steps = max((e.get("step", 0) for e in records), default=0)
            if type(interval) is int and interval > steps:
                diagnostics.append(
                    f"No generated examples yet: this run reached update {steps}; "
                    f"the first scheduled evaluation is at update {interval}. "
                    "This does not indicate a tokenizer error."
                )
            else:
                diagnostics.append(
                    "No generated examples received. Samples may be disabled or pending; "
                    "check sample_prompts and validation settings in train.py. "
                    "For a custom training loop, report examples with run.sample(). "
                    "This message does not diagnose a tokenizer error."
                )
        if not capabilities or not capabilities[-1].get("commands"):
            diagnostics.append(
                "Cooperative stop/snapshot support has not been reported by this script."
            )
        snapshots = [
            json.loads(path.read_text())
            for path in sorted((directory / "snapshots").glob("*.json"))
        ]
        hw_path = directory / "hardware.jsonl"
        hw_lines = []
        if hw_path.is_file():
            with hw_path.open("rb") as stream:
                size = hw_path.stat().st_size
                stream.seek(max(0, size - 2_000_000))

                if size > 2_000_000:
                    stream.readline()  # discard a partial first record
                hw_lines = stream.read().decode(errors="replace").splitlines()[-1000:]
        hw = []
        for line in hw_lines:
            try:
                hw.append(json.loads(line))
            except ValueError:
                continue
        reporting_path = directory / "reporting-status.json"
        reporting = (
            json.loads(reporting_path.read_text()) if reporting_path.exists() else None
        )
        if reporting and reporting.get("remote"):
            # Render only readings reported by the training computer.
            hw = [event["payload"] for event in records if event["kind"] == "hardware"]
            diagnostics.append(
                "Independent training computer: checkpoint files stay there. Server snapshots contain the report history, not remote weights or datasets."
            )
        output = directory / "output.log"

        def tail(path, limit=65536):
            if not path.is_file():
                return ""
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - limit))
                return stream.read().decode(errors="replace")

        return {
            "run": json.loads((directory / "run.json").read_text()),
            "script_capabilities": {
                key: value
                for key, value in (capabilities[-1] if capabilities else {}).items()
                if type(value) is bool
            },
            "validation_control": validation_state(directory, records, status),
            "restart_context": service.restart_context(run_id),
            "revision": (
                json.loads((directory / "revision.json").read_text())
                if (directory / "revision.json").is_file()
                else None
            ),
            "resume_problem": (
                json.loads((directory / "resume-problem.json").read_text())
                if (directory / "resume-problem.json").is_file()
                else None
            ),
            "status": status,
            "events": records,
            "first_event_time": first_event[0]["time"] if first_event else None,
            "hardware": hw,
            "logs": tail(output)
            + "\n".join(
                str(event["payload"].get("text", ""))
                for event in records
                if event["kind"] == "console"
            ),
            "reporting": reporting,
            "reporting_url": (
                json.loads((directory / "reporting.json").read_text())["url"]
                if (directory / "reporting.json").exists()
                else None
            ),
            "evaluation_logs": tail(directory / "evaluation.log"),
            "diagnostics": diagnostics,
            "snapshots": snapshots,
            "source_data_bytes": sum(
                info["bytes"]
                for manifest in [
                    directory / "datasets.json",
                    *sorted((directory / "lineage").glob("*/datasets.json")),
                ]
                for info in json.loads(manifest.read_text()).values()
            ),
            "disk_bytes": directory_size(directory),
            "config": json.loads((directory / "config.json").read_text()),
            "datasets": json.loads((directory / "datasets.json").read_text()),
            "module_nodes": (
                json.loads((directory / "graph_modules.json").read_text())
                if (directory / "graph_modules.json").exists()
                else {}
            ),
            "directory": str(directory),
            "script": (directory / "train.py").read_text(),
        }
