"""Optional chat processors are isolated, explicit and cannot bypass sampling masks."""

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.training import chat_worker
from backend.training.api import router
from backend.training.chat import (
    ChatLoad,
    ChatMessage,
    ChatOptions,
    ChatService,
)
from test_checkpoint_chat import make_chat_run, wait_ready


def seven_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Test tool {name}",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        for name in (
            "weather",
            "search",
            "clock",
            "read",
            "write",
            "calculate",
            "translate",
        )
    ]


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_decoder_masks_before_greedy_or_top_p_and_receives_only_new_tokens(
    tmp_path, monkeypatch, temperature
):
    import torch

    class Tokenizer:
        eos_token_id = 3

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"] == seven_tools()
            assert messages[-1] == {"role": "user", "content": "hello"}
            assert kwargs["add_generation_prompt"] is True
            return [0, 1]

        def decode(self, tokens, skip_special_tokens=True):
            # XML markers often are added special tokens: they must remain visible.
            return "" if skip_special_tokens else "".join("X" for _ in tokens)

    class Model:
        def __call__(self, ids):
            return (
                torch.tensor([100.0, 50.0, -100.0, 0.0])
                .reshape(1, 1, 4)
                .expand(1, ids.shape[1], 4)
            )

    prefixes = []
    builds = []

    def build(tools, tokenizer, request):
        builds.append(tools)

        def mask(ids, scores, options):
            assert ids.ndim == 1
            prefixes.append(ids.tolist())
            assert options is request
            result = torch.full_like(scores, -torch.inf)
            result[2] = scores[2]
            return result

        return mask

    emitted = []
    monkeypatch.setattr(
        chat_worker, "emit", lambda kind, **payload: emitted.append((kind, payload))
    )
    request = ChatMessage(
        text="hello",
        tools=seven_tools(),
        decoder="test",
        max_new_tokens=3,
        temperature=temperature,
        top_p=0.1,
        repetition_penalty=2,
    ).model_dump()
    request["history"] = []
    for _ in range(2):
        chat_worker.generate(
            Model(), Tokenizer(), request, "cpu", tmp_path / "cancel", build
        )
        assert emitted[-2] == ("text", {"text": "XXX"})
    assert prefixes == [[], [2], [2, 2], [], [2], [2, 2]]
    assert builds == [seven_tools(), seven_tools()]


def test_no_decoder_keeps_prompt_kwargs_sampling_and_special_token_behavior(
    tmp_path, monkeypatch
):
    import torch

    class Tokenizer:
        eos_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            assert "tools" not in kwargs
            return [0]

        def decode(self, tokens, skip_special_tokens=True):
            assert skip_special_tokens is True
            return ",".join(map(str, tokens))

    class Model:
        def __call__(self, ids):
            return torch.tensor([0.0, 2.0]).reshape(1, 1, 2).expand(1, ids.shape[1], 2)

    emitted = []
    monkeypatch.setattr(
        chat_worker, "emit", lambda kind, **payload: emitted.append((kind, payload))
    )
    request = ChatMessage(text="hello", temperature=0, max_new_tokens=2).model_dump()
    request["history"] = []

    def unused(*args):
        pytest.fail("A factory must not be called when decoder=None")

    chat_worker.generate(
        Model(), Tokenizer(), request, "cpu", tmp_path / "cancel", unused
    )
    assert emitted[-2] == ("text", {"text": "1,1"})


def test_real_template_contains_catalog_and_xml_special_token_survives_eos(
    tmp_path, monkeypatch
):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    names = [tool["function"]["name"] for tool in seven_tools()]
    vocabulary = {
        name: index
        for index, name in enumerate(["[UNK]", "hello", *names, "<no_call/>", "</s>"])
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="</s>",
        additional_special_tokens=["<no_call/>"],
    )
    tokenizer.chat_template = "{% for tool in tools %}{{ tool.function.name + ' ' }}{% endfor %}{{ messages[-1].content }}"
    seen = []

    def model(ids):
        seen.append(ids[0].tolist())
        return torch.zeros(1, ids.shape[1], len(tokenizer))

    sequence = [vocabulary["<no_call/>"], tokenizer.eos_token_id]

    def build(tools, tokenizer, request):
        def mask(ids, scores, request):
            output = torch.full_like(scores, -torch.inf)
            output[sequence[len(ids)]] = 0
            return output

        return mask

    emitted = []
    monkeypatch.setattr(
        chat_worker, "emit", lambda kind, **payload: emitted.append((kind, payload))
    )
    request = ChatMessage(
        text="hello",
        tools=seven_tools(),
        decoder="test",
        temperature=0,
        max_new_tokens=4,
    ).model_dump()
    request["history"] = []
    chat_worker.generate(model, tokenizer, request, "cpu", tmp_path / "cancel", build)
    assert seen[0] == [vocabulary[name] for name in names] + [vocabulary["hello"]]
    assert emitted[-2] == ("text", {"text": "<no_call/>"})
    assert emitted[-1] == ("done", {"finish_reason": "eos"})


@pytest.mark.parametrize(
    "invalid", ["blocked", "nan", "positive_inf", "shape", "integer", "not_callable"]
)
def test_bad_decoder_stops_instead_of_sampling_an_arbitrary_token(
    tmp_path, monkeypatch, invalid
):
    import torch

    tokenizer = SimpleNamespace(eos_token_id=None, encode=lambda *args, **kwargs: [0])

    def model(ids):
        return torch.zeros(1, ids.shape[1], 3)

    def build(*args):
        if invalid == "not_callable":
            return None

        def mask(ids, scores, request):
            return {
                "blocked": torch.full_like(scores, -torch.inf),
                "nan": torch.full_like(scores, torch.nan),
                "positive_inf": torch.full_like(scores, torch.inf),
                "shape": scores.unsqueeze(0),
                "integer": scores.long(),
            }[invalid]

        return mask

    request = ChatMessage(
        text="hello", mode="completion", decoder="test", temperature=0
    ).model_dump()
    with pytest.raises(ValueError, match="Decoder"):
        chat_worker.generate(
            model, tokenizer, request, "cpu", tmp_path / "cancel", build
        )


def test_module_loading_uses_only_registered_targets_and_valid_local_paths(tmp_path):
    marker = tmp_path / "imported"
    (tmp_path / "local_decoder.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef build(*args): return lambda ids, scores, request: scores\n"
    )
    registrations = {
        "test": {"target": "local_decoder:build", "python_path": str(tmp_path)}
    }
    with pytest.raises(ValueError, match="not registered"):
        chat_worker.decoder_builder("local_decoder:build", registrations, {})
    assert not marker.exists()
    builders = {}
    first = chat_worker.decoder_builder("test", registrations, builders)
    assert marker.exists() and callable(first)
    assert chat_worker.decoder_builder("test", registrations, builders) is first
    with pytest.raises(ValueError, match="local module"):
        chat_worker.decoder_builder(
            "bad", {"bad": {"target": "https://remote/code.py:build"}}, {}
        )


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "function"}],
        [{"type": "function", "function": {"name": "bad\nname"}}],
        [{"type": "function", "function": {"name": "x", "parameters": []}}],
        [seven_tools()[0], seven_tools()[0]],
        [{"type": "function", "function": {"name": "x", "description": "x" * 262144}}],
    ],
)
def test_reject_malformed_catalogs(tools):
    with pytest.raises(ValidationError):
        ChatLoad(tools=tools)


def test_session_defaults_overrides_and_configure_are_recorded_without_reloading(
    tmp_path,
):
    # An in-memory pipe captures the exact request sent to the isolated worker.
    chat = ChatService()
    chat.run_id = "run-test"
    chat.session = tmp_path
    chat.decoders = {"test": {"label": "XML tool grammar"}}
    chat.process = SimpleNamespace(poll=lambda: None, stdin=io.StringIO())
    chat.state.update(
        state="ready", has_chat_template=True, tools=seven_tools(), decoder="test"
    )
    try:
        state = chat.send("run-test", ChatMessage(text="hello"))
        payload = json.loads(chat.process.stdin.getvalue())
        assert payload["tools"] == seven_tools() and payload["decoder"] == "test"
        assert chat.generation_options["decoder"] == "test"
        assert state["decoder_label"] == "XML tool grammar"
        chat.state["state"] = "ready"
        chat.configure("run-test", ChatOptions(tools=[], decoder=None))
        saved = json.loads((tmp_path / "transcript.json").read_text())
        assert saved["tools"] == [] and saved["decoder"] is None
        chat.send("run-test", ChatMessage(text="hi"))
        payload = json.loads(chat.process.stdin.getvalue().splitlines()[-1])
        assert payload["tools"] == [] and payload["decoder"] is None
        chat.state["state"] = "ready"
        chat.send(
            "run-test", ChatMessage(text="hello", tools=seven_tools(), decoder="test")
        )
        payload = json.loads(chat.process.stdin.getvalue().splitlines()[-1])
        assert payload["tools"] == seven_tools() and payload["decoder"] == "test"
    finally:
        chat.process = None
        chat.close()


def test_allowlisted_decoder_with_real_checkpoint_worker_and_api_validation(
    tmp_path, monkeypatch
):
    config = tmp_path / "server.json"
    decoder_root = tmp_path / "plugins"
    decoder_root.mkdir()
    (decoder_root / "test_chat_plugin.py").write_text("""import torch
def build(tools, tokenizer, request):
    assert len(tools) == 7
    def mask(ids, scores, request):
        assert ids.ndim == 1
        output = torch.full_like(scores, -torch.inf)
        output[3] = 0.0
        return output
    return mask
""")
    config.write_text(
        json.dumps(
            {
                "chat_decoders": {
                    "xml-tool-grammar": {
                        "label": "XML tool grammar",
                        "target": "test_chat_plugin:build",
                        "python_path": str(decoder_root),
                    }
                }
            }
        )
    )
    monkeypatch.setattr("backend.installation.SETTINGS", config)
    service, directory = make_chat_run(tmp_path)
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / "weave_support").rglob("*.py")
    }
    chat = ChatService()
    try:
        chat.load(directory, ChatLoad(tools=seven_tools(), decoder="xml-tool-grammar"))
        state = wait_ready(chat, directory.name)
        assert state["state"] == "ready", state
        pid = chat.process.pid
        chat.send(
            directory.name, ChatMessage(text="hello", temperature=0, max_new_tokens=3)
        )
        state = wait_ready(chat, directory.name)
        assert not state["error"], state
        assert state["messages"][-1]["content"] == "hello hello hello"
        assert state["decoder"] == "xml-tool-grammar" and len(state["tools"]) == 7
        chat.clear(directory.name)
        chat.configure(directory.name, ChatOptions(tools=None, decoder=None))
        chat.send(
            directory.name, ChatMessage(text="hello", temperature=0, max_new_tokens=2)
        )
        state = wait_ready(chat, directory.name)
        assert state["messages"][-1]["content"] == "world world"
        assert chat.process.pid == pid
    finally:
        chat.close()
    assert before == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / "weave_support").rglob("*.py")
    }
    assert "Chat must never execute train.py" in (directory / "train.py").read_text()
    monkeypatch.setenv("WEAVE_TRAINING_ENABLED", "1")
    app = FastAPI()
    app.include_router(router(tmp_path))
    endpoint = f"/api/training/runs/{directory.name}/chat"
    with TestClient(app) as client:
        catalog = client.get(endpoint).json()
        assert catalog["decoders"] == [
            {
                "id": "xml-tool-grammar",
                "label": "XML tool grammar",
                "available": True,
                "schema_profile": "generic",
            }
        ]
        assert (
            client.post(
                endpoint + "/load", json={"decoder": "not-registered"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                endpoint + "/load", json={"decoder": "https://example.com/code.py"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                endpoint + "/load", json={"tools": [{"bad": "tool"}]}
            ).status_code
            == 422
        )
        assert (
            client.put(
                endpoint + "/settings", json={"tools": seven_tools()}
            ).status_code
            == 409
        )
