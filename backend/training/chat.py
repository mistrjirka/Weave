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


from .chat_contract import (
    ChatOptions,
    DecoderRegistration,
    configured_decoders,
    decoder_catalog,
    ChatLoad,
    ChatMessage,
    ChatPreview,
    ExpectedCall,
    ChatTestCase,
    ChatTestSuite,
    ChatToolResult,
    option_diagnostics,
    support_directory,
    checkpoint_catalog,
)
from .chat_turns import ChatTurns


class ChatService(ChatTurns):
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
