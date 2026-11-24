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

from . import service as service_module
from .service import inside

def build_snapshot(
    directory, checkpoint_event=None, include_data=False, identifier=None, *, index_timeout=30
):
    if checkpoint_event and checkpoint_event["payload"].get("remote"):
        raise ValueError(
            "Remote weights remain on the training computer; copy them to the server before backing them up"
        )
    identifier = identifier or uuid.uuid4().hex
    relative = f"snapshots/snapshot-{identifier}.zip"
    target = directory / relative
    target.parent.mkdir(exist_ok=True)
    lock = target.with_suffix(".lock")
    if (target.with_suffix(".json")).exists():
        return relative
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(lock_fd)
    except FileExistsError:
        return relative
    temporary = target.with_suffix(".partial")
    cutoff = time.time()
    checkpoint = checkpoint_event["payload"]["path"] if checkpoint_event else None
    manifest = {
        "version": 1,
        "created": time.time(),
        "checkpoint": checkpoint,
        "resumable_weights": checkpoint is not None,
        "includes_source_data": include_data,
        "files": {},
        "missing_data": [],
        "note": "Dataset hashes identify data; only included files preserve it. Custom loader resume guarantees depend on the script.",
    }

    def add(archive, name, content):
        archive.writestr(name, content)
        manifest["files"][name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    def add_path(archive, name, path):
        digest = hashlib.sha256()
        written = 0
        # Bound to the size observed at opening; logs may continue growing.
        remaining = path.stat().st_size
        with (
            path.open("rb") as source,
            archive.open(name, "w", force_zip64=True) as output,
        ):
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                remaining -= len(chunk)
        manifest["files"][name] = {"sha256": digest.hexdigest(), "bytes": written}

    try:
        if checkpoint_event:
            source = inside(directory, checkpoint)
            if file_hash(source) != checkpoint_event["payload"]["sha256"]:
                raise ValueError("Published checkpoint checksum changed")
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for path in sorted(directory.rglob("*")):
                name = path.relative_to(directory).as_posix()
                if not include_data and path.name.startswith("prepared-"):
                    continue
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or name.startswith(
                        (
                            "snapshots/",
                            "commands/",
                            "starting-weights/",
                            "source-data/",
                            "packages/",
                        )
                    )
                    or path.name
                    in (
                        "operator.token",
                        "manifest.json",
                        "reporting.json",
                        ".report-token",
                    )
                    or path.name.endswith((".pending", ".partial"))
                    or "event-index.sqlite" in path.name
                    or "__pycache__" in path.parts
                ):
                    continue
                if name.startswith("checkpoints/") and name != checkpoint:
                    continue
                if path.name == "events.jsonl":
                    continue
                if (
                    path.parent == directory
                    and path.name.startswith("consumed-rank-")
                    and checkpoint_event
                ):
                    with (
                        archive.open(name, "w", force_zip64=True) as output,
                        path.open("rb") as source,
                    ):
                        digest = hashlib.sha256()
                        length = 0
                        for line in source:
                            try:
                                if (
                                    json.loads(line).get("step", 0)
                                    > checkpoint_event["step"]
                                ):
                                    continue
                            except ValueError:
                                continue
                            output.write(line)
                            digest.update(line)
                            length += len(line)
                        manifest["files"][name] = {
                            "sha256": digest.hexdigest(),
                            "bytes": length,
                        }
                else:
                    add_path(archive, name, path)
            selected = [
                event
                for event in service_module.events(directory, timeout=index_timeout)
                if event["time"] <= cutoff
                and (
                    not checkpoint_event
                    or event.get("step", 0) <= checkpoint_event["step"]
                )
            ]
            add(
                archive,
                "events.jsonl",
                ("\n".join(json.dumps(e) for e in selected) + "\n").encode(),
            )
            if include_data:
                sources = [
                    ("", directory / "datasets.json"),
                    *[
                        (f"lineage/{path.parent.name}/", path)
                        for path in (directory / "lineage").glob("*/datasets.json")
                    ],
                ]
                for scope, dataset_manifest in sources:
                    for key, info in json.loads(dataset_manifest.read_text()).items():
                        path = Path(info["path"])
                        changed = not path.is_file()
                        if not changed:
                            changed = (
                                (file_hash(path) != info["sha256"])
                                if info.get("sha256")
                                else (
                                    path.stat().st_size != info["bytes"]
                                    or path.stat().st_mtime_ns != info.get("mtime_ns")
                                )
                            )
                        if changed:
                            manifest["missing_data"].append(
                                {
                                    "source": scope + key,
                                    "path": info["path"],
                                    "reason": "missing or changed",
                                }
                            )
                        else:
                            name = f"source-data/{scope}{key}/{path.name}"
                            add_path(archive, name, path)
                            if (
                                info.get("sha256")
                                and manifest["files"][name]["sha256"] != info["sha256"]
                            ):
                                raise ValueError(
                                    "Source data changed while making the snapshot"
                                )
            add(
                archive,
                "RESTORE.md",
                b"Import this archive into Weave, select its checkpoint, then choose Resume or Start new stage. Verify/reconnect dataset paths. Scripts are not executed during import. No checkpoint means this is a run record, not a weight backup. GPU kernels/custom loaders may prevent exact continuation.\n",
            )
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        os.replace(temporary, target)
    except BaseException:
        lock.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(
        directory / "snapshots" / f"snapshot-{identifier}.json",
        {
            "path": relative,
            "checkpoint": checkpoint,
            "sha256": file_hash(target),
            "bytes": target.stat().st_size,
            "missing_data": manifest["missing_data"],
        },
    )
    lock.unlink(missing_ok=True)
    return relative


def import_snapshot(service, file):
    with zipfile.ZipFile(file) as archive:
        infos = archive.infolist()
        if len(infos) > 50000 or sum(info.file_size for info in infos) > 100 * 1024**3:
            raise ValueError("Snapshot exceeds import limits (100 GB, 50,000 files).")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("version") != 1:
            raise ValueError("Unsupported snapshot version")
        if len({entry.filename for entry in infos}) != len(infos):
            raise ValueError("Duplicate archive paths")
        expected = set(manifest["files"]) | {"manifest.json"}
        if set(archive.namelist()) != expected:
            raise ValueError("Snapshot manifest does not match archive contents")
        run_id = "import-" + uuid.uuid4().hex
        directory = service.root / "runs" / run_id
        directory.mkdir()
        try:
            for name, info in manifest["files"].items():
                path = inside(directory, name)
                if name.startswith("/") or "\\" in name:
                    raise ValueError("Unsafe archive path")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if (
                    path.stat().st_size != info["bytes"]
                    or file_hash(path) != info["sha256"]
                ):
                    raise ValueError("Snapshot checksum failed: " + name)
            info = json.loads((directory / "run.json").read_text())
            info.update(
                id=run_id, imported=True, created=datetime.now(timezone.utc).isoformat()
            )
            atomic_json(directory / "run.json", info)
            atomic_json(
                directory / "status.json",
                {
                    "state": "completed" if manifest["checkpoint"] else "prepared",
                    "imported": True,
                },
            )
            # Reconnect bundled datasets without trusting paths from another host.
            datasets = json.loads((directory / "datasets.json").read_text())
            config = json.loads((directory / "config.json").read_text())
            for key, dataset in datasets.items():
                bundled = directory / "source-data" / key / Path(dataset["path"]).name
                if bundled.is_file():
                    dataset["path"] = str(bundled)
                    if key in ("dataset", "eval_dataset"):
                        config[key] = str(bundled)
            atomic_json(directory / "datasets.json", datasets)
            atomic_json(directory / "config.json", config)
            return info
        except BaseException:
            shutil.rmtree(directory)
            raise
