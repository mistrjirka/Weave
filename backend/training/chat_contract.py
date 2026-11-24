"""Checkpoint selection and a bounded, isolated inference session. Never imports train.py."""

from __future__ import annotations

from contextlib import suppress
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Literal
import uuid

from pydantic import BaseModel, Field, FiniteFloat, JsonValue, field_validator

from .layout import PACKAGE
from .service import events, inside
from .weave_training import atomic_json
from .run_contract import latest_contract
from ..core import validate_project
from ..installation import read_settings
from .chat_protocol import (
    catalog_issues,
    clean_assistant,
    context_messages,
    inspect_calls,
)


class ChatOptions(BaseModel):
    tools: list[dict[str, JsonValue]] | None = Field(default=None, max_length=128)
    decoder: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    system: str = Field(default="", max_length=16000)
    chat_template: str = Field(default="", max_length=32000)
    mode: Literal["chat", "completion"] = "chat"

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools):
        if tools is None:
            return tools
        if len(json.dumps(tools, allow_nan=False).encode()) > 262144:
            raise ValueError("Tool catalog must be at most 256 KiB of JSON.")
        names = set()
        for tool in tools:
            function = tool.get("function")
            if tool.get("type") != "function" or not isinstance(function, dict):
                raise ValueError(
                    "Each tool needs type 'function' and a function object."
                )
            name = function.get("name")
            if not isinstance(name, str) or not re.fullmatch(
                r"[^\x00-\x1f\x7f]{1,64}", name
            ):
                raise ValueError(
                    "Tool names need 1–64 characters without control characters."
                )
            if name in names:
                raise ValueError(f"Duplicate tool name: {name}")
            names.add(name)
            if "parameters" in function and not isinstance(
                function["parameters"], dict
            ):
                raise ValueError(
                    f"Tool {name}: parameters must be a JSON Schema object."
                )
            if "parameters" in function:
                from jsonschema import Draft202012Validator, SchemaError

                try:
                    Draft202012Validator.check_schema(function["parameters"])
                except SchemaError as exc:
                    raise ValueError(
                        f"Tool {name}: invalid JSON Schema: {exc.message}"
                    ) from exc
                pending = [function["parameters"]]
                while pending:
                    node = pending.pop()
                    if isinstance(node, dict):
                        reference = node.get("$ref")
                        if reference is not None and (
                            not isinstance(reference, str)
                            or not reference.startswith("#")
                        ):
                            raise ValueError(
                                f"Tool {name}: external schema references are unsupported; bundle local schemas instead."
                            )
                        pending.extend(node.values())
                    elif isinstance(node, list):
                        pending.extend(node)
        return tools


class DecoderRegistration(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    target: str = Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    python_path: str | None = None
    schema_profile: Literal["generic", "xml-flat-v1"] = "generic"

    @field_validator("python_path")
    @classmethod
    def local_directory(cls, value):
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute() or not path.is_dir():
                raise ValueError(
                    "Decoder python_path must be an existing absolute local directory."
                )
            return str(path.resolve())
        return value


def configured_decoders():
    """Only the server owner can register code; chat requests select an ID."""
    configured = read_settings().get("chat_decoders", {})
    if not isinstance(configured, dict):
        raise ValueError("chat_decoders in .weave-config.json must be an object.")
    result = {}
    for identifier, value in configured.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier):
            raise ValueError(f"Invalid configured decoder ID: {identifier}")
        result[identifier] = DecoderRegistration.model_validate(value).model_dump()
    return result


def decoder_catalog():
    choices = [
        {
            "id": identifier,
            "label": info["label"],
            "available": True,
            "schema_profile": info["schema_profile"],
        }
        for identifier, info in configured_decoders().items()
    ]
    if not any(item["id"] == "xml-tool-grammar" for item in choices):
        choices.append(
            {"id": "xml-tool-grammar", "label": "XML tool grammar", "available": False}
        )
    return choices


class ChatLoad(ChatOptions):
    checkpoint: str | None = None
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda:[0-9]+)$")


class ChatMessage(ChatOptions):
    text: str = Field(min_length=1, max_length=32000)
    mode: Literal["chat", "completion"] = "chat"
    system: str = Field(default="", max_length=16000)
    chat_template: str = Field(default="", max_length=32000)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    context_length: int = Field(default=2048, ge=2, le=32768)
    temperature: FiniteFloat = Field(default=0.7, ge=0, le=2)
    top_p: FiniteFloat = Field(default=0.9, gt=0, le=1)
    repetition_penalty: FiniteFloat = Field(default=1.0, ge=1, le=2)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class ChatPreview(ChatMessage):
    text: str = Field(default="", max_length=32000)


class ExpectedCall(BaseModel):
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ChatTestCase(ChatMessage):
    name: str = Field(default="", max_length=200)
    expected: list[ExpectedCall] | str | None = None
    temperature: FiniteFloat = Field(default=0, ge=0, le=2)


class ChatTestSuite(BaseModel):
    cases: list[ChatTestCase] = Field(min_length=1, max_length=50)


class ChatToolResult(BaseModel):
    tool_call_id: str = Field(min_length=1, max_length=150)
    content: str = Field(max_length=32000)
    simulated: bool = True


def option_diagnostics(options, decoders):
    decoder = options.get("decoder")
    if decoder is not None and decoder not in decoders:
        raise ValueError(
            "Choose a decoder registered on the server. Reload the checkpoint after new registrations."
        )
    return catalog_issues(
        options.get("tools"),
        decoders[decoder].get("schema_profile", "generic") if decoder else "generic",
    )


def support_directory(directory):
    return (
        directory / PACKAGE
        if (directory / PACKAGE / "model.py").is_file()
        else directory
    )


def checkpoint_catalog(directory):
    records = events(directory)
    contract = latest_contract(records)
    primary = contract["primary_metric"]
    metric_names = list(
        dict.fromkeys(
            [
                "eval/tool_exact_match",
                "eval/exact_match",
                "eval/tool_accuracy",
                *contract["metrics"],
            ]
        )
    )
    losses = {}
    task_metrics = {}
    checkpoint_metrics = {}
    for record in records:
        value = (
            record["payload"].get("eval/loss") if record["kind"] == "metrics" else None
        )
        reference = record.get("checkpoint")
        if type(value) in (int, float) and math.isfinite(value) and not reference:
            losses[record["step"]] = float(value)
        if record["kind"] == "metrics":
            if reference:
                checkpoint_metrics.setdefault(
                    (reference, record.get("policy_version")), {}
                ).update(
                    {
                        key: float(score)
                        for key, score in record["payload"].items()
                        if type(score) in (int, float) and math.isfinite(score)
                    }
                )
                continue
            for key in metric_names:
                score = record["payload"].get(key)
                if type(score) in (int, float) and math.isfinite(score):
                    task_metrics.setdefault(record["step"], {})[key] = float(score)
    checkpoints = {}
    for record in records:
        if record["kind"] != "checkpoint":
            continue
        payload = record["payload"]
        path = payload.get("path", "")
        try:
            available = (
                bool(path)
                and not payload.get("remote")
                and inside(directory, path).is_file()
            )
        except ValueError:
            available = False
        matched = checkpoint_metrics.get((path, payload.get("sha256")), {})
        checkpoints[path] = {
            "path": path,
            "step": record["step"],
            "sha256": payload.get("sha256", ""),
            "validation_loss": matched.get("eval/loss", losses.get(record["step"])),
            "available": available,
            "task_metrics": {
                **task_metrics.get(record["step"], {}),
                **{key: value for key, value in matched.items() if key in metric_names},
            },
        }
    choices = sorted(checkpoints.values(), key=lambda item: item["step"], reverse=True)
    local = [item for item in choices if item["available"]]
    scored = [item for item in local if item["validation_loss"] is not None]
    best = (
        min(scored, key=lambda item: (item["validation_loss"], -item["step"]))
        if scored
        else next(iter(local), None)
    )
    reason = (
        "Lowest recorded validation loss among saved checkpoints; not a tool-accuracy score."
        if scored
        else "Latest saved checkpoint; none has a matching validation-loss score."
    )
    for metric in (
        [primary]
        if primary
        else ["eval/tool_exact_match", "eval/exact_match", "eval/tool_accuracy"]
    ):
        candidates = [item for item in local if metric in item["task_metrics"]]
        if candidates:
            direction = contract["metrics"].get(metric, {}).get("direction", "max")
            best = max(
                candidates,
                key=lambda item: (
                    (1 if direction == "max" else -1) * item["task_metrics"][metric],
                    item["step"],
                ),
            )
            reason = f"{'Highest' if direction == 'max' else 'Lowest'} {metric} measured at this checkpoint. Unscored checkpoints are not ranked by task quality."
            break
        if primary:
            best = next(iter(local), None)
            reason = (
                f"Latest saved checkpoint; none has a matching {primary} score yet."
            )
    problem = None
    support = support_directory(directory)
    try:
        project = json.loads((support / "project.json").read_text())
        report = validate_project(project)
        inputs = [
            node
            for node in project["graph"]["nodes"]
            if node["data"]["kind"] == "input"
        ]
        valid_input = (
            len(inputs) == 1
            and inputs[0]["data"].get("params", {}).get("dtype", "int64") == "int64"
            and len(inputs[0]["data"].get("params", {}).get("shape", ["B", "T"])) == 2
        )
        valid_output = (
            len(report["outputs"]) == 1
            and len(report["outputs"][0]["shape"]) == 3
            and report["outputs"][0]["dtype"] == "float32"
        )
        if not valid_input or not valid_output:
            problem = "Checkpoint chat currently requires one token-ID [B,T] input and one vocabulary-logit [B,T,V] output."
        configured = json.loads((support / "tokenizers.json").read_text())
        if not configured:
            problem = "This run has no saved input tokenizer. Select a tokenizer on the graph input before creating the training run."
    except (OSError, ValueError, KeyError, TypeError):
        problem = "This run is missing its frozen model or tokenizer metadata."
    if not local:
        problem = "No local checkpoint weights are available yet. Remote reports alone do not include checkpoint files."
    return {
        "checkpoints": choices,
        "recommended_checkpoint": best["path"] if best else None,
        "recommendation": reason,
        "problem": problem,
    }
