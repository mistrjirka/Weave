"""Chat protocol, pinned context, previews and regression scoring."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.training.chat import (
    ChatService,
    ChatOptions,
    ChatMessage,
    ChatPreview,
    ChatTestSuite,
    ChatToolResult,
    ChatSessions,
)
from backend.training.chat_protocol import (
    clean_assistant,
    inspect_calls,
    catalog_issues,
    presets,
    context_messages,
)
from backend.training import chat_worker
from test_chat_decoders import seven_tools


def test_xml_markers_and_structured_arguments_are_normalized():
    tools = seven_tools()
    output = '<call name="weather"><arg name="value">A &amp; B</arg></call>\n<|end|>\n\n<|end|>\n'
    assert clean_assistant(output).endswith("</call>")
    parsed = inspect_calls(output, tools)
    assert parsed["kind"] == "calls" and not parsed["errors"]
    assert parsed["calls"][0]["arguments"] == {"value": "A & B"}
    assert inspect_calls("<no_call/>\n<|end|>", tools)["kind"] == "no_call"
    assert inspect_calls('<call name="unknown"/>', tools)["errors"]
    assert inspect_calls(
        '<call name="weather"><arg name="wrong">x</arg></call>', tools
    )["errors"]
    assert inspect_calls(
        '<call name="weather"><arg name="value">raw & bad</arg></call>', tools
    )["errors"]
    assert inspect_calls('<call name="weather">', tools)["errors"]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["string", "null"]},
        {"type": "string", "pattern": "x+"},
        {"anyOf": [{"type": "string"}]},
        {"type": "object", "properties": {"nested": {"type": "string"}}},
        {"$ref": "#/defs/value"},
    ],
)
def test_xml_profile_reports_unsupported_features_instead_of_dropping(schema):
    tools = seven_tools()
    tools[0]["function"]["parameters"]["properties"]["value"] = schema
    assert catalog_issues(tools, "xml-flat-v1")["errors"]


def test_catalog_profile_is_explicit_and_generic_decoders_can_support_unicode():
    tools = seven_tools()
    tools[0]["function"]["name"] = "dotted.tool"
    tools[0]["function"]["parameters"]["properties"] = {"město": {"type": "string"}}
    tools[0]["function"]["parameters"]["required"] = ["město"]
    assert not catalog_issues(tools, "generic")["errors"]
    assert "ASCII argument" in " ".join(catalog_issues(tools, "xml-flat-v1")["errors"])
    assert not ChatOptions(tools=tools).model_dump()["decoder"]
    assert catalog_issues([*seven_tools(), *seven_tools()])["warnings"]
    assert len(presets()[0]["tools"]) == 7


def fake_service(tmp_path):
    chat = ChatService()
    chat.run_id = "run-test"
    chat.session = tmp_path
    chat.state.update(state="ready", has_chat_template=True, tools=seven_tools())
    chat.process = SimpleNamespace(poll=lambda: None, stdin=io.StringIO())
    return chat


def finish(chat):
    chat.process = None
    chat.close()


def test_completed_calls_pin_context_and_tool_results_are_structured(tmp_path):
    chat = fake_service(tmp_path)
    try:
        chat.send("run-test", ChatMessage(text="weather", temperature=0))
        chat.state["messages"][-1][
            "content"
        ] = '<call name="weather"><arg name="value">Prague</arg></call>\n<|end|>\n'
        chat._finish_turn("turn_end")
        chat.state["state"] = "ready"
        turn = chat.state["messages"][-1]
        assert turn["included_in_context"] and turn["tool_calls"][0]["function"][
            "arguments"
        ] == {"value": "Prague"}
        assert "<|end|>" not in turn["content"]
        with pytest.raises(ValueError, match="pinned"):
            chat.configure("run-test", ChatOptions(system="different"))
        chat.inject_result(
            "run-test",
            ChatToolResult(
                tool_call_id=turn["tool_calls"][0]["id"], content='{"temperature":12}'
            ),
        )
        assert context_messages(chat.state["messages"])[-1]["role"] == "tool"
        with pytest.raises(ValueError, match="already has a result"):
            chat.inject_result(
                "run-test",
                ChatToolResult(
                    tool_call_id=turn["tool_calls"][0]["id"], content="duplicate"
                ),
            )
        chat.preview("run-test", ChatPreview(text="and tomorrow?", max_new_tokens=30))
        payload = json.loads(chat.process.stdin.getvalue().splitlines()[-1])
        assert payload["operation"] == "preview" and payload["history"][1]["tool_calls"]
        chat.clear("run-test")
        chat.configure("run-test", ChatOptions(system="different"))
        assert not chat.state["conversation_settings"]
    finally:
        finish(chat)


def test_failed_empty_and_invalid_turns_are_visible_but_excluded(tmp_path):
    chat = fake_service(tmp_path)
    try:
        for output, reason in [
            ("", "eos"),
            ("partial", "error"),
            ('<call name="unknown"/>', "eos"),
        ]:
            chat.send("run-test", ChatMessage(text="hello"))
            chat.state["messages"][-1]["content"] = output
            chat._finish_turn(reason)
            chat.state["state"] = "ready"
            assert not chat.state["messages"][-1]["included_in_context"]
            assert not chat.state["messages"][-2]["included_in_context"]
        assert not context_messages(chat.state["messages"])
    finally:
        finish(chat)


def test_template_gets_structured_calls_once_and_preview_matches_tokens(monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        Path(__file__).parent / "fixtures/xml_chat_template.jinja"
    ).read_text()
    rendered = []
    actual = tokenizer.apply_chat_template

    def capture(messages, **kwargs):
        rendered.append(actual(messages, **{**kwargs, "tokenize": False}))
        return actual(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", capture)
    request = ChatPreview(
        text="hello", tools=seven_tools(), max_new_tokens=16
    ).model_dump()
    request["history"] = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": '<call name="weather"><arg name="value">Prague</arg></call>\n<|end|>\n<|end|>',
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "weather", "arguments": {"value": "Prague"}},
                }
            ],
        },
        {"role": "tool", "content": '{"temperature":12}'},
    ]
    budget = chat_worker.preview(tokenizer, request)
    assert rendered[0].count('<call name="weather">') == 1
    assert "<|end|>\n<|end|>" not in rendered[0]
    assert '<result>\n{"temperature":12}\n</result>' in rendered[0]
    assert budget["history_tokens"] > 0
    assert budget["remaining_tokens"] == 2048 - budget["prompt_tokens"] - 16


def test_suite_scores_independent_cases_without_running_tools(tmp_path, monkeypatch):
    import torch

    tokenizer = SimpleNamespace(
        eos_token_id=1,
        all_special_tokens=[],
        encode=lambda *args, **kwargs: [0],
        decode=lambda ids, **kwargs: "<no_call/>" if ids and ids[0] == 0 else "",
    )

    def model(ids):
        return torch.tensor([2.0, 0.0]).reshape(1, 1, 2).expand(1, ids.shape[1], 2)

    events = []
    monkeypatch.setattr(
        chat_worker, "emit", lambda kind, **payload: events.append((kind, payload))
    )
    case = ChatMessage(
        text="hello", mode="completion", max_new_tokens=1, temperature=0
    ).model_dump()
    case.update(history=[], expected=[])
    chat_worker.run_suite(
        model, tokenizer, {"cases": [case, case]}, "cpu", tmp_path / "cancel", {}, {}
    )
    assert len([event for event in events if event[0] == "suite_case"]) == 2
    assert all(
        event[1]["result"]["exact_match"]
        for event in events
        if event[0] == "suite_case"
    )
    assert events[-1][0] == "suite_done"


def test_per_run_sessions_and_idle_unload_preserve_settings(tmp_path):
    sessions = ChatSessions()
    a = sessions.for_run("a")
    b = sessions.for_run("b")
    assert a is not b and sessions.for_run("a") is a
    a.run_id = "a"
    a.state.update(
        state="ready",
        tools=seven_tools(),
        messages=[{"role": "user", "content": "kept"}],
    )
    a._expire()
    assert (
        a.status("a")["state"] == "unloaded" and a.status("a")["tools"] == seven_tools()
    )
    assert a.status("a")["messages"][0]["content"] == "kept"
    sessions.close()


def test_checkpoint_task_metric_wins_over_lower_validation_loss(tmp_path):
    from backend.training.chat import checkpoint_catalog
    from backend.training.weave_training import Run, atomic_json
    from test_training import tiny_project

    atomic_json(tmp_path / "project.json", tiny_project())
    atomic_json(tmp_path / "tokenizers.json", {"ids": {"id": "saved"}})
    for name in ("first.pt", "second.pt"):
        (tmp_path / name).write_bytes(b"weights")
    with Run(tmp_path) as run:
        run.step = 1
        run.checkpoint(tmp_path / "first.pt")
        run.log({"eval/loss": 0.1, "eval/tool_exact_match": 0.4}, step=1)
        run.step = 2
        run.checkpoint(tmp_path / "second.pt")
        run.log({"eval/loss": 0.5, "eval/tool_exact_match": 0.8}, step=2)
    result = checkpoint_catalog(tmp_path)
    assert result["recommended_checkpoint"] == "second.pt"
    assert "Highest eval/tool_exact_match" in result["recommendation"]


def test_project_preferences_are_reusable_without_resetting_tools(tmp_path):
    path = tmp_path / "project-chat-settings.json"
    first = ChatService(path)
    first.run_id = "run"
    first.session = tmp_path
    first.state.update(state="ready")
    try:
        first.configure("run", ChatOptions(tools=seven_tools(), system="Pinned date"))
        second = ChatService(path)
        assert second.preferences()["tools"] == seven_tools()
        assert second.preferences()["system"] == "Pinned date"
        second.close()
    finally:
        first.close()


def test_unconstrained_tools_preserve_xml_special_tokens(tmp_path, monkeypatch):
    import torch

    tokenizer = SimpleNamespace(
        eos_token_id=1,
        all_special_tokens=[],
        encode=lambda *args, **kwargs: [0],
        decode=lambda ids, skip_special_tokens=True: (
            "" if skip_special_tokens else "<no_call/>" if ids else ""
        ),
    )

    def model(ids):
        return torch.tensor([5.0, 0.0]).reshape(1, 1, 2).expand(1, ids.shape[1], 2)

    events = []
    monkeypatch.setattr(
        chat_worker, "emit", lambda kind, **payload: events.append((kind, payload))
    )
    request = ChatMessage(
        text="hello", mode="completion", tools=[], temperature=0, max_new_tokens=1
    ).model_dump()
    chat_worker.generate(model, tokenizer, request, "cpu", tmp_path / "cancel")
    assert events[-2][1]["text"] == "<no_call/>"
