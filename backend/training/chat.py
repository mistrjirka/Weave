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
    losses = {}
    task_metrics = {}
    for record in records:
        value = (
            record["payload"].get("eval/loss") if record["kind"] == "metrics" else None
        )
        if type(value) in (int, float) and math.isfinite(value):
            losses[record["step"]] = float(value)
        if record["kind"] == "metrics":
            for key in (
                "eval/tool_exact_match",
                "eval/exact_match",
                "eval/tool_accuracy",
            ):
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
        checkpoints[path] = {
            "path": path,
            "step": record["step"],
            "sha256": payload.get("sha256", ""),
            "validation_loss": losses.get(record["step"]),
            "available": available,
            "task_metrics": task_metrics.get(record["step"], {}),
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
    for metric in ("eval/tool_exact_match", "eval/exact_match", "eval/tool_accuracy"):
        candidates = [item for item in local if metric in item["task_metrics"]]
        if candidates:
            best = max(
                candidates,
                key=lambda item: (item["task_metrics"][metric], item["step"]),
            )
            reason = f"Highest {metric} measured at this checkpoint. Unscored checkpoints are not ranked by task quality."
            break
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


class ChatService:
    """One run's isolated model and conversation, with configurable idle unload."""

    def __init__(self, preferences_path=None):
        self.lock = threading.RLock()
        self.process = None
        self.run_id = None
        self.state = self.empty()
        self.timer = None
        self.loading_timer = None
        self.decoders = {}
        self.preferences_path = preferences_path
        self.idle_seconds = max(
            60, min(86400, int(read_settings().get("chat_idle_timeout_seconds", 3600)))
        )

    @staticmethod
    def empty():
        return {
            "state": "unloaded",
            "checkpoint": None,
            "device": "cpu",
            "messages": [],
            "error": None,
            "has_chat_template": False,
            "finish_reason": None,
            "loading_stage": None,
            "loading_started": None,
            "tools": None,
            "decoder": None,
            "decoder_label": None,
            "system": "",
            "chat_template": "",
            "mode": "chat",
            "context_preview": None,
            "conversation_settings": None,
            "test_suite": None,
            "idle_expires": None,
            "notice": None,
            "last_checkpoint": None,
        }

    def status(self, run_id):
        with self.lock:
            return copy.deepcopy(self.state if run_id == self.run_id else self.empty())

    def _idle_timer(self):
        if self.timer:
            self.timer.cancel()
        self.state["idle_expires"] = time.time() + self.idle_seconds
        self.timer = threading.Timer(self.idle_seconds, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        with self.lock:
            if self.state["state"] in ("loading", "generating", "testing"):
                self._idle_timer()
            else:
                self.close(preserve=True)
                self.state["notice"] = (
                    "Model unloaded after inactivity. Messages and tool settings are preserved."
                )

    def close(self, run_id=None, preserve=False):
        with self.lock:
            if run_id is not None and run_id != self.run_id:
                return
            if self.timer:
                self.timer.cancel()
            if self.loading_timer:
                self.loading_timer.cancel()
                self.loading_timer = None
            process, self.process = self.process, None
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process and process.stdin:
                with suppress(OSError):
                    process.stdin.close()
            self.run_id = self.run_id if preserve else None
            previous_checkpoint = self.state["checkpoint"] or self.state.get(
                "last_checkpoint"
            )
            self.state = (
                {
                    **self.state,
                    "state": "unloaded",
                    "checkpoint": None,
                    "idle_expires": None,
                    "last_checkpoint": previous_checkpoint,
                }
                if preserve
                else self.empty()
            )

    def preferences(self):
        if self.preferences_path and self.preferences_path.is_file():
            return ChatOptions.model_validate(
                json.loads(self.preferences_path.read_text())
            ).model_dump()
        return ChatOptions().model_dump()

    def _save_preferences(self):
        if self.preferences_path:
            atomic_json(
                self.preferences_path,
                {key: self.state[key] for key in ChatOptions.model_fields},
            )

    def load(self, directory, request):
        decoders = configured_decoders()
        defaults = self.preferences()
        if self.run_id == directory.name:
            defaults.update({key: self.state[key] for key in ChatOptions.model_fields})
        options = {
            key: (
                getattr(request, key)
                if key in request.model_fields_set
                else defaults[key]
            )
            for key in ChatOptions.model_fields
        }
        diagnostics = option_diagnostics(options, decoders)
        if diagnostics["errors"]:
            raise ValueError(" ".join(diagnostics["errors"]))
        catalog = checkpoint_catalog(directory)
        if catalog["problem"]:
            raise ValueError(catalog["problem"])
        path = request.checkpoint or catalog["recommended_checkpoint"]
        selected = next(
            (
                item
                for item in catalog["checkpoints"]
                if item["path"] == path and item["available"]
            ),
            None,
        )
        if not selected:
            raise ValueError(
                "Choose a published checkpoint whose weights are available on this server."
            )
        with self.lock:
            previous = self.state if self.run_id == directory.name else self.empty()
            reuse_messages = (
                previous["state"] in ("ready", "unloaded")
                and path == (previous["checkpoint"] or previous.get("last_checkpoint"))
                and (
                    not previous.get("conversation_settings")
                    or all(
                        options[key] == previous["conversation_settings"][key]
                        for key in ChatOptions.model_fields
                    )
                )
            )
            self.close()
            self.run_id = directory.name
            self.decoders = decoders
            self.state = {
                **self.empty(),
                "state": "loading",
                "checkpoint": path,
                "device": request.device,
                "loading_stage": "Starting inference worker",
                "loading_started": time.time(),
                **options,
                "messages": previous["messages"] if reuse_messages else [],
                "conversation_settings": (
                    previous.get("conversation_settings") if reuse_messages else None
                ),
                "decoder_label": (
                    decoders[options["decoder"]]["label"]
                    if options["decoder"]
                    else None
                ),
            }
            self.generation_options = None
            self.session = directory / "chat" / uuid.uuid4().hex
            self.session.mkdir(parents=True)
            config = json.loads((directory / "config.json").read_text())
            spec = {
                "checkpoint": str(inside(directory, path)),
                "sha256": selected["sha256"],
                "device": request.device,
                "cancel_path": str(self.session / "cancel"),
                "chat_template": config.get("chat_template", ""),
                "decoders": decoders,
                "decoder": options["decoder"],
            }
            atomic_json(self.session / "worker.json", spec)
            self._save_preferences()
            environment = {
                key: value
                for key, value in os.environ.items()
                if key
                not in (
                    "WEAVE_TRAINING_TOKEN",
                    "WEAVE_REPORT_CONFIG",
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                )
                and not key.startswith("TORCHELASTIC_")
            }
            environment.update(
                USE_TORCH="1",
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1",
            )
            worker = Path(__file__).with_name("chat_worker.py")
            try:
                with (self.session / "worker.log").open("a") as log:
                    self.process = subprocess.Popen(
                        [
                            config["python"],
                            "-u",
                            "-c",
                            "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                            str(worker),
                            str(self.session / "worker.json"),
                        ],
                        cwd=directory,
                        env=environment,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log,
                        text=True,
                        bufsize=1,
                    )
            except OSError as exc:
                self.state.update(state="error", error=str(exc))
                raise ValueError(f"Cannot start checkpoint chat: {exc}") from exc
            self._idle_timer()
            self.loading_timer = threading.Timer(
                300, self._loading_timeout, args=(self.process, self.session)
            )
            self.loading_timer.daemon = True
            self.loading_timer.start()
            threading.Thread(
                target=self._read,
                args=(self.process, self.session),
                daemon=True,
                name="weave-chat",
            ).start()
            return self.status(directory.name)

    def _loading_timeout(self, process, session):
        with self.lock:
            if self.process is not process or self.state["state"] != "loading":
                return
            run_id = self.run_id
            state = {
                **self.state,
                "state": "error",
                "loading_started": None,
                "error": (
                    "Checkpoint loading timed out after 300 seconds during "
                    f"{self.state.get('loading_stage') or 'startup'}. "
                    f"Retry loading or choose another device. Details: {session / 'worker.log'}"
                ),
            }
            self.close()
            self.run_id, self.state = run_id, state
            self._idle_timer()

    def _read(self, process, session):
        try:
            for line in process.stdout:
                event = json.loads(line)
                with self.lock:
                    if self.process is not process:
                        return
                    kind = event["kind"]
                    if kind == "loading":
                        self.state["loading_stage"] = event["stage"]
                    elif kind == "ready":
                        if self.loading_timer:
                            self.loading_timer.cancel()
                            self.loading_timer = None
                        self.state.update(
                            state="ready",
                            has_chat_template=event["has_chat_template"],
                            loading_stage=None,
                            loading_started=None,
                        )
                        self._idle_timer()
                    elif kind == "text":
                        self.state["messages"][-1]["content"] = event["text"]
                    elif kind == "preview":
                        if (self.state.get("context_preview") or {}).get(
                            "request_id"
                        ) == event["request_id"]:
                            self.state["context_preview"] = {**event, "state": "ready"}
                    elif kind == "suite_case":
                        self.state["test_suite"]["results"].append(event["result"])
                        atomic_json(
                            session / "tests" / self.state["test_suite"]["id"],
                            self.state["test_suite"],
                        )
                    elif kind == "suite_done":
                        self.state["test_suite"].update(
                            state="stopped" if event.get("stopped") else "complete"
                        )
                        self.state["state"] = "ready"
                        atomic_json(
                            session / "tests" / self.state["test_suite"]["id"],
                            self.state["test_suite"],
                        )
                        self._idle_timer()
                    elif kind == "done":
                        self._finish_turn(event["finish_reason"])
                        self.state.update(
                            state="ready", finish_reason=event["finish_reason"]
                        )
                    elif kind == "error":
                        if event.get("operation") == "preview":
                            if (self.state.get("context_preview") or {}).get(
                                "request_id"
                            ) == event.get("request_id"):
                                self.state["context_preview"] = {
                                    "state": "error",
                                    "error": event["message"],
                                    "request_id": event.get("request_id"),
                                }
                            continue
                        if self.state["state"] == "generating":
                            self._finish_turn("error")
                        if self.loading_timer:
                            self.loading_timer.cancel()
                            self.loading_timer = None
                        self.state.update(
                            state=(
                                "error" if self.state["state"] == "loading" else "ready"
                            ),
                            error=event["message"],
                            loading_stage=None,
                            loading_started=None,
                        )
                    if kind in ("done", "error"):
                        self._idle_timer()
                        atomic_json(session / "transcript.json", self.state)
                        with (session / "conversations.jsonl").open("a") as archive:
                            archive.write(
                                json.dumps(
                                    {
                                        "time": time.time(),
                                        "generation": self.generation_options,
                                        **self.state,
                                    }
                                )
                                + "\n"
                            )
            code = process.wait()
            with self.lock:
                if self.process is process and self.state["state"] != "error":
                    log = session / "worker.log"
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 2000))
                        lines = (
                            stream.read().decode(errors="replace").strip().splitlines()
                        )
                    detail = lines[-1] if lines else "No additional worker output."
                    self.state.update(
                        state="error",
                        error=f"Chat worker exited ({code}): {detail} Details: {log}",
                        loading_stage=None,
                        loading_started=None,
                    )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            with self.lock:
                if self.process is process:
                    self.state.update(state="error", error=str(exc))
        finally:
            with self.lock:
                if (
                    self.process is process
                    and self.state["state"] == "error"
                    and self.loading_timer
                ):
                    self.loading_timer.cancel()
                    self.loading_timer = None
            process.stdout.close()

    def _options(self, request):
        options = {
            name: (
                getattr(request, name)
                if name in request.model_fields_set
                else self.state[name]
            )
            for name in ChatOptions.model_fields
        }
        decoder = options["decoder"]
        if decoder is not None and decoder not in self.decoders:
            raise ValueError(
                "Decoder is not registered for this session. Configure it on the server and reload the checkpoint."
            )
        issues = option_diagnostics(options, self.decoders)
        if issues["errors"]:
            raise ValueError(" ".join(issues["errors"]))
        return {
            **options,
            "decoder_label": self.decoders[decoder]["label"] if decoder else None,
        }

    def _check_pinned(self, options):
        pinned = self.state.get("conversation_settings")
        if pinned and any(
            message.get("included_in_context") for message in self.state["messages"]
        ):
            if any(options[key] != pinned[key] for key in ChatOptions.model_fields):
                raise ValueError(
                    "Conversation settings are pinned. Clear messages before changing the system prompt, tools, decoder, format or template."
                )

    def _finish_turn(self, reason):
        if (
            not self.state["messages"]
            or self.state["messages"][-1]["role"] != "assistant"
        ):
            return
        message = self.state["messages"][-1]
        message["raw_content"] = message["content"]
        message["content"] = clean_assistant(message["content"])
        parsed = inspect_calls(message["content"], self.state["tools"])
        message["parsed"] = parsed
        message["tool_calls"] = [
            {
                "id": f"call_{message.get('turn_id', 'legacy')}_{index}",
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for index, call in enumerate(parsed["calls"])
            if call["valid"]
        ]
        included = bool(message["content"]) and reason not in ("error", "stopped")
        xml_profile = (
            self.decoders.get(self.state["decoder"], {}).get("schema_profile")
            == "xml-flat-v1"
        )
        if parsed["errors"] or (
            xml_profile and parsed["kind"] not in ("calls", "no_call")
        ):
            included = False
        message["included_in_context"] = included
        message["finish_reason"] = reason
        message["context_reason"] = (
            None
            if included
            else "Failed, empty, stopped or invalid output; this turn is excluded from the model's next prompt."
        )
        if (
            len(self.state["messages"]) > 1
            and self.state["messages"][-2]["role"] == "user"
        ):
            self.state["messages"][-2]["included_in_context"] = included
            self.state["messages"][-2]["context_reason"] = message["context_reason"]

    def _write(self, payload):
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            self.state.update(state="error", error=str(exc))
            raise ValueError(
                "The chat worker disconnected. Reload the checkpoint."
            ) from exc

    def preview(self, run_id, request):
        with self.lock:
            if (
                run_id != self.run_id
                or self.state["state"] != "ready"
                or not self.process
                or self.process.poll() is not None
            ):
                raise ValueError(
                    "Load a checkpoint and finish its response to preview context."
                )
            options = self._options(request)
            identifier = uuid.uuid4().hex
            self.state["context_preview"] = {
                "state": "loading",
                "request_id": identifier,
            }
            self._write(
                {
                    **request.model_dump(),
                    **{key: options[key] for key in ChatOptions.model_fields},
                    "history": context_messages(self.state["messages"]),
                    "operation": "preview",
                    "request_id": identifier,
                }
            )
            self._idle_timer()
            return self.status(run_id)

    def run_tests(self, run_id, request):
        with self.lock:
            if run_id != self.run_id or self.state["state"] != "ready":
                raise ValueError(
                    "Load a checkpoint and finish the response before running tests."
                )
            cases = []
            for case in request.cases:
                options = self._options(case)
                cases.append(
                    {
                        **case.model_dump(),
                        **{key: options[key] for key in ChatOptions.model_fields},
                        "history": [],
                    }
                )
            self.state.update(
                state="testing",
                error=None,
                test_suite={
                    "id": f"suite-{uuid.uuid4().hex}.json",
                    "state": "running",
                    "total": len(cases),
                    "results": [],
                },
            )
            (self.session / "cancel").unlink(missing_ok=True)
            self._write({"operation": "suite", "cases": cases})
            self._idle_timer()
            return self.status(run_id)

    def inject_result(self, run_id, request):
        with self.lock:
            if run_id != self.run_id or self.state["state"] != "ready":
                raise ValueError(
                    "Finish the response before adding a simulated tool result."
                )
            calls = {
                call["id"]: call
                for message in self.state["messages"]
                for call in message.get("tool_calls", [])
                if message.get("included_in_context")
            }
            if request.tool_call_id not in calls:
                raise ValueError("Choose a valid call included in this conversation.")
            if any(
                message.get("tool_call_id") == request.tool_call_id
                for message in self.state["messages"]
            ):
                raise ValueError("This call already has a result.")
            self.state["messages"].append(
                {
                    "role": "tool",
                    "content": request.content,
                    "tool_call_id": request.tool_call_id,
                    "name": calls[request.tool_call_id]["function"]["name"],
                    "simulated": request.simulated,
                    "included_in_context": True,
                }
            )
            atomic_json(self.session / "transcript.json", self.state)
            self._idle_timer()
            return self.status(run_id)

    def configure(self, run_id, request):
        with self.lock:
            if run_id != self.run_id or self.state["state"] != "ready":
                raise ValueError(
                    "Load a checkpoint and finish the response before applying tool settings."
                )
            options = self._options(request)
            self._check_pinned(options)
            self.state.update(options)
            self._save_preferences()
            atomic_json(self.session / "transcript.json", self.state)
            self._idle_timer()
            return self.status(run_id)

    def send(self, run_id, request):
        with self.lock:
            if (
                run_id != self.run_id
                or self.state["state"] != "ready"
                or not self.process
                or self.process.poll() is not None
            ):
                raise ValueError(
                    "Load a checkpoint and wait for the current response to finish."
                )
            options = self._options(request)
            if (
                options["mode"] == "chat"
                and not self.state["has_chat_template"]
                and not options["chat_template"]
            ):
                raise ValueError(
                    "The saved tokenizer has no chat template. Use text completion, or train with a tokenizer containing the matching chat template."
                )
            self._check_pinned(options)
            self.state["conversation_settings"] = {
                key: options[key] for key in ChatOptions.model_fields
            }
            self.state.update(
                state="generating", error=None, finish_reason=None, **options
            )
            payload = {
                **request.model_dump(),
                **{key: options[key] for key in ChatOptions.model_fields},
            }
            if options["decoder"] and "temperature" not in request.model_fields_set:
                payload["temperature"] = 0
            self.generation_options = {
                key: value for key, value in payload.items() if key != "text"
            }
            (self.session / "cancel").unlink(missing_ok=True)
            # A failed/empty generation must not leave an unmatched user turn in
            # the next template invocation. Preserve the displayed attempt, but
            # pass only completed user/assistant pairs back to the model.
            prior = context_messages(self.state["messages"])
            turn_id = uuid.uuid4().hex
            self.state["messages"].extend(
                [
                    {
                        "role": "user",
                        "content": request.text,
                        "turn_id": turn_id,
                        "included_in_context": False,
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "turn_id": turn_id,
                        "included_in_context": False,
                        "generation": self.generation_options,
                    },
                ]
            )
            try:
                self.process.stdin.write(
                    json.dumps({**payload, "history": prior}) + "\n"
                )
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                self.state.update(state="error", error=str(exc))
                raise ValueError(
                    "The chat worker disconnected. Reload the checkpoint."
                ) from exc
            self._idle_timer()
            self._save_preferences()
            return self.status(run_id)

    def stop(self, run_id):
        with self.lock:
            if run_id == self.run_id and self.state["state"] in (
                "generating",
                "testing",
            ):
                (self.session / "cancel").touch()
            return self.status(run_id)

    def clear(self, run_id):
        with self.lock:
            if run_id != self.run_id or self.state["state"] not in (
                "ready",
                "unloaded",
            ):
                raise ValueError(
                    "Wait for the response to finish before starting a new conversation."
                )
            self.state.update(
                messages=[],
                error=None,
                finish_reason=None,
                conversation_settings=None,
                context_preview=None,
            )
            return self.status(run_id)


class ChatSessions:
    """Independent run sessions. Viewing/loading another run never unloads this one."""

    def __init__(self):
        self.lock = threading.RLock()
        self.sessions = {}

    def for_run(self, run_id, preferences_path=None):
        with self.lock:
            if run_id not in self.sessions:
                self.sessions[run_id] = ChatService(preferences_path)
            return self.sessions[run_id]

    def close(self):
        with self.lock:
            for session in self.sessions.values():
                session.close()
