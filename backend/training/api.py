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
from .weave_training import atomic_json
from ..compiler import compile_project


class Login(BaseModel):
    token: str


class Command(BaseModel):
    kind: str
    include_data: bool = False


def router(data):
    service = TrainingService(Path(data) / "training", Path(data) / "tokenizers")
    api = APIRouter(prefix="/api/training")

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Training requests must come from this Weave host")

    def authenticated(request: Request):
        same_origin(request)
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
            "authenticated": hmac.compare_digest(credential, service.token),
            "python": os.environ.get("WEAVE_TRAINING_PYTHON", os.sys.executable),
            "token_file": str(service.token_path),
            "remote_launch": False,
        }

    @api.post("/session")
    def login(body: Login, request: Request, response: Response):
        same_origin(request)
        if not hmac.compare_digest(body.token, service.token):
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
        custom = '''"""Custom training: keep your own data, loss and optimizer logic."""\nfrom weave_training import Run\n\nwith Run() as run:\n    run.capabilities(metrics=True, samples=False, checkpoint=False, commands=True)\n    # Replace this demonstration with your own PyTorch training loop.\n    for step in range(10):\n        commands = run.poll_commands()\n        if any(command["kind"] == "stop" for command in commands):\n            break\n        run.log({"example/progress": step}, step=step)\n        run.progress(completed=step + 1, total=10)\n    run.warning("This starter reported events only. Add your model, loss, optimizer, and checkpoint save logic.")\n'''
        return {
            "config": TrainingConfig().model_dump(),
            "scripts": {
                "causal": (HERE / "template.py").read_text(),
                "sft": (HERE / "template.py").read_text(),
                "custom": custom,
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
        path = service.root / "plans" / (safe_id(project_id) + ".json")
        return (
            json.loads(path.read_text())
            if path.exists()
            else {"version": 1, "project_id": project_id, "stages": []}
        )

    @api.put("/plans/{project_id}", dependencies=[Depends(authenticated)])
    def put_plan(project_id: str, plan: Plan):
        if plan.project_id != project_id or plan.version != 1:
            raise HTTPException(422, "Training plan identity/version mismatch")
        if len({stage.id for stage in plan.stages}) != len(plan.stages):
            raise HTTPException(422, "Stage IDs must be unique")
        atomic_json(
            service.root / "plans" / (safe_id(project_id) + ".json"), plan.model_dump()
        )
        return plan

    @api.get("/runs", dependencies=[Depends(authenticated)])
    def get_runs(project_id: str):
        return service.list_runs(project_id)

    @api.post("/runs", dependencies=[Depends(authenticated)])
    def launch(body: Launch):
        if (
            body.stage.config.execution == "local"
            and os.environ.get("WEAVE_TRAINING_ENABLED") != "1"
        ):
            raise HTTPException(
                403,
                "Local execution is disabled. Start the server with WEAVE_TRAINING_ENABLED=1.",
            )
        try:
            return service.launch(body)
        except (ValueError, FileNotFoundError, SyntaxError) as exc:
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
            "logs": tail(output),
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
            for path in directory.iterdir():
                if (
                    path.is_file()
                    and path.suffix in (".py", ".json", ".txt", ".md")
                    and not path.name.startswith("prepared-")
                ):
                    if path.name == "config.json":
                        config = json.loads(path.read_text())
                        if (directory / "starting-weights/parent.pt").is_file():
                            config["checkpoint"] = "starting-weights/parent.pt"
                        archive.writestr("config.json", json.dumps(config, indent=2))
                    else:
                        archive.write(path, path.name)
            for folder in ("tokenizers", "licenses", "starting-weights"):
                for path in (directory / folder).rglob("*"):
                    if path.is_file():
                        archive.write(path, str(path.relative_to(directory)))
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
