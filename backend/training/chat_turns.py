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


class ChatTurns:
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
