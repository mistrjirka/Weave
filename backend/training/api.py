"""Authenticated, same-origin local training API."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    File,
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
    model_metadata,
    import_snapshot,
    HERE,
)
from .workspaces import WorkspaceConflict
from .environment import inspect_python
from .reporting import ReportStore, ReportBatch
from .layout import PACKAGE, normalize_script
from .recipe import is_code_first, starter_script
from ..paths import projects_directory
from ..compiler import compile_project


class Login(BaseModel):
    token: str


class Command(BaseModel):
    kind: str
    include_data: bool = False


def router(data):
    service = TrainingService(Path(data) / "training", Path(data) / "tokenizers")
    api = APIRouter(prefix="/api/training")
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

    def safe_id(value):
        try:
            return valid_id(value)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

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
"""\nfrom weave_training import Run\n\nwith Run() as run:\n    run.capabilities(metrics=True, samples=False, checkpoint=False, commands=True)\n    # Replace this demonstration with your own PyTorch training loop.\n    for step in range(10):\n        commands = run.poll_commands()\n        if any(command["kind"] == "stop" for command in commands):\n            break\n        run.log({"example/progress": step}, step=step)\n        run.progress(completed=step + 1, total=10)\n    run.warning("This starter reported events only. Add your model, loss, optimizer, and checkpoint save logic.")\n'''
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
            return inspect_python(python)
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
            except ValueError:
                saved.workspace_warning = "Script saved. Resolve architecture errors, then save again to generate weave_support/model.py."
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

    @api.get("/runs", dependencies=[Depends(authenticated)])
    def get_runs(project_id: str):
        return service.list_runs(project_id)

    @api.post("/runs", dependencies=[Depends(authenticated)])
    def launch(body: Launch, request: Request):
        if (
            body.stage.config.execution == "local"
            and os.environ.get("WEAVE_TRAINING_ENABLED") != "1"
        ):
            raise HTTPException(
                403,
                "Local execution is disabled. Start the server with WEAVE_TRAINING_ENABLED=1.",
            )
        try:
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
            batch = ReportBatch.model_validate_json(content)
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

    @api.get("/runs/{run_id}", dependencies=[Depends(authenticated)])
    def get_run(run_id: str):
        directory = run_directory(run_id)
        service.complete_snapshots(directory)
        records = events(directory, limit=10000)
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
        if not any(e["kind"] == "sample" for e in records):
            diagnostics.append(
                "No generated examples received. Configure evaluation prompts and a tokenizer, or add run.sample()."
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
        hw_lines = hw_path.read_text().splitlines()[-1000:] if hw_path.exists() else []
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
            "status": status,
            "events": records[-10000:],
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
            "disk_bytes": sum(
                p.stat().st_size for p in directory.rglob("*") if p.is_file()
            ),
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

    @api.get("/runs/{run_id}/events", dependencies=[Depends(authenticated)])
    async def stream(run_id: str, request: Request):
        directory = run_directory(run_id)
        last = request.headers.get("last-event-id") or request.query_params.get("after")

        async def generate():
            cursor = last or ""
            while not await request.is_disconnected():
                for event in events(directory, after=cursor, limit=1000):
                    cursor = event["id"]
                    yield "id: " + event["id"] + "\ndata: " + json.dumps(event) + "\n\n"
                yield ": heartbeat\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
        return service.command(run_id, body.kind, body.include_data)

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

    return api
