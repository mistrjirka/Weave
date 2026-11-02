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


class ChatOptions(BaseModel):
    tools: list[dict[str, JsonValue]] | None = Field(default=None, max_length=128)
    decoder: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")

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
                r"[A-Za-z0-9_-]{1,64}", name
            ):
                raise ValueError(
                    "Tool names need 1–64 letters, digits, underscores or hyphens."
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
        return tools


class DecoderRegistration(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    target: str = Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    python_path: str | None = None

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
        {"id": identifier, "label": info["label"], "available": True}
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


def support_directory(directory):
    return (
        directory / PACKAGE
        if (directory / PACKAGE / "model.py").is_file()
        else directory
    )


def checkpoint_catalog(directory):
    records = events(directory)
    losses = {}
    for record in records:
        value = (
            record["payload"].get("eval/loss") if record["kind"] == "metrics" else None
        )
        if type(value) in (int, float) and math.isfinite(value):
            losses[record["step"]] = float(value)
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
        "Lowest recorded validation loss among saved checkpoints."
        if scored
        else "Latest saved checkpoint; none has a matching validation-loss score."
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


class ChatService:
    """One loaded model per server, with explicit unload and a ten-minute idle timeout."""

    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.run_id = None
        self.state = self.empty()
        self.timer = None
        self.loading_timer = None

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
        }

    def status(self, run_id):
        with self.lock:
            return copy.deepcopy(self.state if run_id == self.run_id else self.empty())

    def _idle_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(600, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        with self.lock:
            if self.state["state"] in ("loading", "generating"):
                self._idle_timer()
            else:
                self.close()

    def close(self, run_id=None):
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
            self.run_id = None
            self.state = self.empty()

    def load(self, directory, request):
        decoders = configured_decoders()
        if request.decoder is not None and request.decoder not in decoders:
            raise ValueError(
                "Choose a decoder registered in the server's .weave-config.json."
            )
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
                "tools": request.tools,
                "decoder": request.decoder,
                "decoder_label": (
                    decoders[request.decoder]["label"] if request.decoder else None
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
                "decoder": request.decoder,
            }
            atomic_json(self.session / "worker.json", spec)
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
                    elif kind == "done":
                        self.state.update(
                            state="ready", finish_reason=event["finish_reason"]
                        )
                    elif kind == "error":
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
            for name in ("tools", "decoder")
        }
        decoder = options["decoder"]
        if decoder is not None and decoder not in self.decoders:
            raise ValueError(
                "Decoder is not registered for this session. Configure it on the server and reload the checkpoint."
            )
        return {
            **options,
            "decoder_label": self.decoders[decoder]["label"] if decoder else None,
        }

    def configure(self, run_id, request):
        with self.lock:
            if run_id != self.run_id or self.state["state"] != "ready":
                raise ValueError(
                    "Load a checkpoint and finish the response before applying tool settings."
                )
            self.state.update(self._options(request))
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
            if (
                request.mode == "chat"
                and not self.state["has_chat_template"]
                and not request.chat_template
            ):
                raise ValueError(
                    "The saved tokenizer has no chat template. Use text completion, or train with a tokenizer containing the matching chat template."
                )
            options = self._options(request)
            self.state.update(
                state="generating", error=None, finish_reason=None, **options
            )
            payload = {
                **request.model_dump(),
                **{key: options[key] for key in ("tools", "decoder")},
            }
            self.generation_options = {
                key: value for key, value in payload.items() if key != "text"
            }
            (self.session / "cancel").unlink(missing_ok=True)
            # A failed/empty generation must not leave an unmatched user turn in
            # the next template invocation. Preserve the displayed attempt, but
            # pass only completed user/assistant pairs back to the model.
            prior = []
            messages = self.state["messages"]
            for index in range(0, len(messages) - 1, 2):
                if messages[index + 1]["content"]:
                    prior.extend(messages[index : index + 2])
            self.state["messages"].extend(
                [
                    {"role": "user", "content": request.text},
                    {"role": "assistant", "content": ""},
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
            return self.status(run_id)

    def stop(self, run_id):
        with self.lock:
            if run_id == self.run_id and self.state["state"] == "generating":
                (self.session / "cancel").touch()
            return self.status(run_id)

    def clear(self, run_id):
        with self.lock:
            if run_id != self.run_id or self.state["state"] != "ready":
                raise ValueError(
                    "Wait for the response to finish before starting a new conversation."
                )
            self.state.update(messages=[], error=None, finish_reason=None)
            return self.status(run_id)
