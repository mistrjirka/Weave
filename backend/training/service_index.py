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

def events(
    directory, *, limit=None, kind=None, kinds=None, after=None, max_bytes=None, channel=None, timeout=30, latest=False, _attempt=0
):
    """Incremental SQLite index over durable script-owned JSONL event streams."""
    database = directory / "event-index.sqlite"
    with closing(sqlite3.connect(database, timeout=timeout)) as connection:
        schema = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE name IN ('offsets', 'events', 'kind_index')"
        )}
        if schema != {"offsets", "events", "kind_index"}:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS offsets (path TEXT PRIMARY KEY, offset INTEGER)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, kind TEXT, document TEXT)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS kind_index ON events(kind)")
            connection.commit()
        sources = [
            directory / "events.jsonl",
            directory / "system/events.jsonl",
            directory / "preparation/events.jsonl",
            *sorted(directory.glob("evaluation-*/events.jsonl")),
            *sorted(directory.glob("evaluation-*/preparation/events.jsonl")),
        ]
        batches = []
        for path in sources:
            if not path.exists():
                continue
            name = str(path.relative_to(directory))
            row = connection.execute(
                "SELECT offset FROM offsets WHERE path=?", (name,)
            ).fetchone()
            offset = row[0] if row and row[0] <= path.stat().st_size else 0
            documents = []
            with path.open("rb") as stream:
                stream.seek(offset)
                while True:
                    begin = stream.tell()
                    line = stream.readline(1024 * 1024 + 2)
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        if len(line) > 1024 * 1024:
                            while line and not line.endswith(b"\n"):
                                line = stream.readline(1024 * 1024)
                            continue
                        stream.seek(begin)
                        break
                    try:
                        event = json.loads(line)
                        if (
                            event.get("version") != 1
                            or not isinstance(event.get("kind"), str)
                            or not isinstance(event.get("payload"), dict)
                        ):
                            continue
                        import math

                        if (
                            not isinstance(event.get("id"), str)
                            or not isinstance(event.get("time"), (int, float))
                            or not math.isfinite(event["time"])
                            or type(event.get("step")) is not int
                            or type(event.get("tokens")) is not int
                        ):
                            continue
                        event["channel"] = (
                            event.get("channel")
                            if event.get("channel")
                            in ("training", "system", "evaluation")
                            else (
                                "training"
                                if path.parent == directory
                                else (
                                    "system"
                                    if path.parent.name == "system"
                                    else "evaluation"
                                )
                            )
                        )
                        documents.append((event["id"], event["kind"], json.dumps(event)))
                    except (ValueError, KeyError, AttributeError):
                        continue
                old_offset = row[0] if row else None
                if stream.tell() != (old_offset if old_offset is not None else 0):
                    batches.append((name, old_offset, stream.tell(), documents))
        if batches:
            connection.execute("BEGIN IMMEDIATE")
        for name, old_offset, new_offset, documents in batches:
            current = connection.execute("SELECT offset FROM offsets WHERE path=?", (name,)).fetchone()
            if (current[0] if current else None) != old_offset:
                connection.rollback()
                if _attempt >= 7:
                    exc = sqlite3.OperationalError("event index changed repeatedly while reading")
                    exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
                    raise exc
                return events(directory, limit=limit, kind=kind, kinds=kinds, after=after, max_bytes=max_bytes, channel=channel, timeout=timeout, latest=latest, _attempt=_attempt + 1)
            connection.executemany(
                "INSERT OR IGNORE INTO events(id,kind,document) VALUES(?,?,?)", documents
            )
            if old_offset != new_offset:
                connection.execute("INSERT OR REPLACE INTO offsets(path,offset) VALUES(?,?)", (name, new_offset))
        if batches:
            connection.commit()
        query = "SELECT document FROM events"
        values, filters = [], []
        if kind:
            filters.append("kind=?")
            values.append(kind)
        if kinds is not None:
            if not kinds:
                return []
            filters.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            values.extend(kinds)
        if channel is not None:
            filters.append("json_extract(document, '$.channel')=?")
            values.append(channel)
        if after is not None:
            previous = connection.execute(
                "SELECT seq FROM events WHERE id=?", (after,)
            ).fetchone()
            filters.append("seq>?")
            values.append(previous[0] if previous else 0)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        ascending = after is not None and not latest
        query += " ORDER BY seq " + ("ASC" if ascending else "DESC")
        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)
        cursor = connection.execute(query, values)
        if max_bytes is None:
            rows = cursor.fetchall()
        else:
            rows, size = [], 0
            for row in cursor:
                size += len(row[0].encode())
                if size > max_bytes:
                    if not rows:
                        raise ValueError(
                            "One event exceeds the response budget; read its run artifact instead"
                        )
                    break
                rows.append(row)
        return [
            json.loads(row[0])
            for row in (rows if ascending else reversed(rows))
        ]


def hardware():
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        rows = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)

            def optional(call):
                try:
                    return call()
                except nvml.NVMLError:
                    return None

            power = optional(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
            rows.append(
                {
                    "index": index,
                    "uuid": nvml.nvmlDeviceGetUUID(handle),
                    "name": nvml.nvmlDeviceGetName(handle),
                    "memory_total_mb": memory.total / 1024**2,
                    "memory_used_mb": memory.used / 1024**2,
                    "utilization": optional(
                        lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    ),
                    "temperature": optional(
                        lambda: nvml.nvmlDeviceGetTemperature(
                            handle, nvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "power_watts": power / 1000 if power is not None else None,
                }
            )
        return rows
    except Exception:
        # CLI is a compatibility fallback when NVML bindings are unavailable.
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rows = []
        for line in result.stdout.splitlines():
            values = [v.strip() for v in line.split(",")]

            def numeric(value):
                try:
                    return float(value)
                except ValueError:
                    return None

            rows.append(
                dict(
                    zip(
                        (
                            "index",
                            "uuid",
                            "name",
                            "memory_total_mb",
                            "memory_used_mb",
                            "utilization",
                            "temperature",
                            "power_watts",
                        ),
                        [
                            int(values[0]),
                            values[1],
                            values[2],
                            *[numeric(v) for v in values[3:]],
                        ],
                    )
                )
            )
        return rows
    except (OSError, subprocess.SubprocessError):
        return []


def model_metadata(files, project):
    tree = ast.parse(files["model.py"])
    root = [node for node in tree.body if isinstance(node, ast.ClassDef)][-1]
    init = next(
        node
        for node in root.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    names = []
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):

            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.append(target.attr)
    # Support a single path of modules; all residual/attention operations inside
    # each container stay on that module's device.
    from ..codegen import OPERATIONS

    graph = project["graph"]
    computational = [
        n for n in graph["nodes"] if n["data"]["kind"] not in ("input", "output")
    ]
    reasons = []
    for node in computational:
        kind = node["data"]["kind"]
        if kind not in ("block", "loop", "moe") and (
            kind not in OPERATIONS or OPERATIONS[kind].expression is not None
        ):
            reasons.append(
                "Root graph contains functional/custom operations; wrap each stage in a container before using automatic split placement."
            )
        if (
            len([e for e in graph["edges"] if e["target"] == node["id"]]) != 1
            or len([e for e in graph["edges"] if e["source"] == node["id"]]) != 1
        ):
            reasons.append(
                "Automatic split placement requires a single chain of top-level modules."
            )
    if (
        len([n for n in graph["nodes"] if n["data"]["kind"] == "input"]) != 1
        or len([n for n in graph["nodes"] if n["data"]["kind"] == "output"]) != 1
    ):
        reasons.append(
            "The built-in text recipes require a single token input and logits output."
        )
    return {
        "modules": list(dict.fromkeys(names)),
        "split_supported": not reasons,
        "split_reasons": list(dict.fromkeys(reasons)),
    }
