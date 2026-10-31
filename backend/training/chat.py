"""Checkpoint selection and a bounded, isolated inference session. Never imports train.py."""

from __future__ import annotations

from contextlib import suppress
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Literal
import uuid

from pydantic import BaseModel, Field, FiniteFloat

from .layout import PACKAGE
from .service import events, inside
from .weave_training import atomic_json
from ..core import validate_project


class ChatLoad(BaseModel):
    checkpoint: str | None = None
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda:[0-9]+)$")


class ChatMessage(BaseModel):
    text: str = Field(min_length=1, max_length=32000)
    mode: Literal["chat", "completion"] = "chat"
    system: str = Field(default="", max_length=16000)
    chat_template: str = Field(default="", max_length=32000)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    context_length: int = Field(default=2048, ge=2, le=32768)
    temperature: FiniteFloat = Field(default=0.7, ge=0, le=2)
    top_p: FiniteFloat = Field(default=0.9, gt=0, le=1)
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
            self.state = {
                **self.empty(),
                "state": "loading",
                "checkpoint": path,
                "device": request.device,
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
            threading.Thread(
                target=self._read,
                args=(self.process, self.session),
                daemon=True,
                name="weave-chat",
            ).start()
            return self.status(directory.name)

    def _read(self, process, session):
        try:
            for line in process.stdout:
                event = json.loads(line)
                with self.lock:
                    if self.process is not process:
                        return
                    kind = event["kind"]
                    if kind == "ready":
                        self.state.update(
                            state="ready", has_chat_template=event["has_chat_template"]
                        )
                        self._idle_timer()
                    elif kind == "text":
                        self.state["messages"][-1]["content"] = event["text"]
                    elif kind == "done":
                        self.state.update(
                            state="ready", finish_reason=event["finish_reason"]
                        )
                    elif kind == "error":
                        self.state.update(
                            state=(
                                "error" if self.state["state"] == "loading" else "ready"
                            ),
                            error=event["message"],
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
                    self.state.update(
                        state="error",
                        error=f"Chat worker exited ({code}); see {session / 'worker.log'}",
                    )
        except (OSError, ValueError, KeyError) as exc:
            with self.lock:
                if self.process is process:
                    self.state.update(state="error", error=str(exc))
        finally:
            process.stdout.close()

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
            self.state.update(state="generating", error=None, finish_reason=None)
            self.generation_options = request.model_dump(exclude={"text"})
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
                    json.dumps({**request.model_dump(), "history": prior}) + "\n"
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
