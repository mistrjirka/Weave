"""Validated run controls and a narrow migration for the validation-control trainer."""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, StrictInt, model_validator

from .weave_training import file_hash


class ValidationSettings(BaseModel):
    mode: Literal["inline", "async"] = "async"
    device: StrictInt | Literal["cpu"] = "cpu"
    eval_every: int = Field(default=50, ge=1, le=1000000, strict=True)
    eval_batches: int = Field(default=5, ge=1, le=100000, strict=True)

    @model_validator(mode="after")
    def valid_device(self):
        if self.device != "cpu" and (type(self.device) is not int or self.device < 0):
            raise ValueError("Choose CPU or a nonnegative GPU index")
        return self


EXCLUDED_CODE_FOLDERS = {
    "runs",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "checkpoints",
    "snapshots",
    "packages",
    "lineage",
    "source-data",
    "starting-weights",
    "tokenizers",
    "licenses",
    "migrations",
}
# The release before live controls has the same optimizer, scheduler and loader
# checkpoint contract. Other unknown trainer revisions retain strict resume checks.
VALIDATION_UPGRADE_FROM = {
    "469f4a2fb01e81e9ab3f11a755bae691e65b924be68463613d978988987f7560",
    # Initial live-controls rollout, before legacy train_eval alias normalization.
    "9862eeef1d8c4bee5e0da1c1fad43c85aa11255c628910d5480beca6820bebe0",
}


def code_hash(directory):
    directory = Path(directory)
    paths = []
    for folder, children, files in os.walk(directory):
        children[:] = [
            name
            for name in children
            if name not in EXCLUDED_CODE_FOLDERS and not Path(folder, name).is_symlink()
        ]
        paths.extend(
            Path(folder, name)
            for name in files
            if name.endswith(".py") and not Path(folder, name).is_symlink()
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def validation_state(directory, records, status):
    configuration = next(
        (
            event["payload"]
            for event in reversed(records)
            if event["kind"] == "configuration" and event.get("channel") == "training"
        ),
        {},
    )
    capabilities = next(
        (
            event["payload"]
            for event in reversed(records)
            if event["kind"] == "capabilities" and event.get("channel") == "training"
        ),
        {},
    )
    applied = next(
        (
            event["payload"]
            for event in reversed(records)
            if event["kind"] == "validation_settings"
            and event.get("channel") == "training"
        ),
        None,
    )
    current = {
        "mode": (
            "async"
            if configuration.get("async_evaluation")
            or configuration.get("strategy") == "train_eval"
            else "inline"
        ),
        "device": (
            configuration.get("evaluation_device")
            if configuration.get("evaluation_device") is not None
            else "cpu"
        ),
        "eval_every": configuration.get("eval_every", 50),
        "eval_batches": configuration.get("eval_batches", 5),
    }
    if applied:
        current.update({key: applied[key] for key in current})
    request_path = Path(directory) / "validation-request.json"
    requested = json.loads(request_path.read_text()) if request_path.is_file() else None
    pending = (
        requested
        if requested and (not applied or requested["id"] != applied.get("command_id"))
        else None
    )
    return {
        "supported": bool(capabilities.get("live_validation")),
        "active": status["state"] in ("running", "preparing", "starting"),
        "current": current,
        "pending": pending,
        "applied_at_step": applied.get("effective_after_step") if applied else None,
        "applied_command_id": applied.get("command_id") if applied else None,
    }
