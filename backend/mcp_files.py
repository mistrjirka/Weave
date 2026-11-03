"""Bounded MCP file adapters over existing APIs, with no server-path access."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import asyncio
from pathlib import PurePosixPath
from urllib.parse import quote

import httpx
from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents
from pydantic import BaseModel, Field

MAX_FILE_BYTES = 4 * 1024 * 1024
UPLOADS = {
    "tokenizer": ("/api/tokenizers/upload", "files", True),
    "dataset": ("/api/training/datasets", "file", False),
    "snapshot": ("/api/training/import", "file", False),
}
TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".log",
    ".toml",
    ".yaml",
    ".yml",
    ".csv",
}


class UploadFileContent(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(max_length=4 * ((MAX_FILE_BYTES + 2) // 3))


def segment(value):
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("Supply a single run identifier")
    return quote(value, safe="")


async def upload_files(bridge, kind, files, ctx):
    path, field, multiple = UPLOADS[kind]
    if not files or len(files) > 32 or (not multiple and len(files) != 1):
        return {
            "error": "Tokenizer accepts 1–32 files; dataset/snapshot accepts exactly one"
        }
    multipart, total, names = [], 0, set()
    try:
        for item in files:
            name = item.filename
            if (
                name in {".", ".."}
                or "/" in name
                or "\\" in name
                or "\0" in name
                or name in names
            ):
                raise ValueError("Use unique basenames without directory components")
            names.add(name)
            data = await asyncio.to_thread(
                base64.b64decode, item.content_base64, validate=True
            )
            total += len(data)
            if total > MAX_FILE_BYTES:
                raise ValueError(
                    "MCP uploads are limited to 4 MiB total; use the UI/HTTP API for larger files"
                )
            multipart.append((field, (name, data, "application/octet-stream")))
        async with bridge.client() as client:
            response = await client.post(
                path, files=multipart, headers=bridge.credentials(ctx)
            )
            if len(response.content) > 1_000_000:
                return {
                    "status": response.status_code,
                    "error": "Upload response exceeds 1 MB",
                }
            return {"status": response.status_code, "data": response.json()}
    except (ValueError, binascii.Error, httpx.HTTPError) as exc:
        return {"error": str(exc)}


async def download_file(bridge, kind, project, run_id, path, ctx):
    try:
        kwargs = {"headers": bridge.credentials(ctx)}
        if kind == "model":
            if project is None:
                raise ValueError("Model export requires the project document")
            method, endpoint, filename = "POST", "/api/export", "model.zip"
            kwargs["json"] = project
        else:
            run = segment(run_id)
            method = "GET"
            if kind == "training_package":
                endpoint, filename = f"/api/training/runs/{run}/package", "training.zip"
            elif kind == "transcript":
                endpoint, filename = (
                    f"/api/training/runs/{run}/chat/export",
                    "transcript.jsonl",
                )
            else:
                if not path or PurePosixPath(path).suffix.lower() not in TEXT_SUFFIXES:
                    raise ValueError(
                        "Artifact downloads accept text/code extensions only; use HTTP for checkpoint weights"
                    )
                endpoint, filename = (
                    f"/api/training/runs/{run}/artifact",
                    PurePosixPath(path).name,
                )
                kwargs["params"] = {"path": path}
        async with bridge.client() as client:
            async with client.stream(method, endpoint, **kwargs) as response:
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_FILE_BYTES:
                        return {
                            "status": response.status_code,
                            "error": "MCP download exceeds 4 MiB; use the UI/HTTP API",
                        }, []
                if response.status_code >= 400:
                    try:
                        detail = json.loads(data)
                    except ValueError:
                        detail = data.decode(errors="replace")[:1000]
                    return {"status": response.status_code, "data": detail}, []
                mime = response.headers.get(
                    "content-type", "application/octet-stream"
                ).split(";", 1)[0]
        digest = hashlib.sha256(data).hexdigest()
        uri = f"weave://exports/{digest}/{quote(filename, safe='')}"
        if kind in {"artifact", "transcript"}:
            resource = TextResourceContents(
                uri=uri, mimeType=mime, text=data.decode("utf-8")
            )
        else:
            resource = BlobResourceContents(
                uri=uri, mimeType=mime, blob=base64.b64encode(data).decode("ascii")
            )
        return {
            "status": 200,
            "data": {
                "filename": filename,
                "mime_type": mime,
                "bytes": len(data),
                "sha256": digest,
                "uri": uri,
            },
        }, [EmbeddedResource(type="resource", resource=resource)]
    except (ValueError, httpx.HTTPError) as exc:
        return {"error": str(exc)}, []
