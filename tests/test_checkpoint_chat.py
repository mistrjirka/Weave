"""Use genuine frozen exports/checkpoints without executing user training files."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.training.api import router
from backend.training.chat import ChatService, ChatLoad, ChatMessage, checkpoint_catalog
from backend.training.service import TrainingService, TrainingConfig, Stage, Launch
from backend.training.weave_training import Run
from test_training import tiny_project


def make_chat_run(tmp_path, chat_template=True):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from backend.tokenizers import TokenizerService

    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "user": 1,
                "assistant": 2,
                "hello": 3,
                "world": 4,
                "end": 5,
                "system": 6,
                "test": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="end"
    )
    if chat_template:
        fast.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    local = tmp_path / "tokenizer-source"
    fast.save_pretrained(local)
    tokenizers = TokenizerService(tmp_path / "tokenizers")
    info = tokenizers.load(local)
    project = tiny_project()
    project["id"] = "project-checkpoint-chat"
    project["graph"]["nodes"][0]["data"]["tokenizer"] = info
    project["graph"]["nodes"][3]["data"]["params"]["bias"] = True
    # Avoid creating a huge side effect: importing this recipe must fail the test.
    source = "raise RuntimeError('Chat must never execute train.py')\n"
    service = TrainingService(tmp_path / "training", tmp_path / "tokenizers")
    run = service.launch(
        Launch(
            project=project,
            stage=Stage(
                id="chat-test",
                name="Chat test",
                script=source,
                config=TrainingConfig(
                    recipe="custom", execution="monitor", python=sys.executable
                ),
            ),
        )
    )
    directory = service.run_dir(run["id"])
    script = """from pathlib import Path
import torch
from weave_support.model import Model
model = Model()
with torch.no_grad():
    for p in model.parameters(): p.zero_()
    model.vocabulary.bias[4] = 10
Path('checkpoints').mkdir(exist_ok=True)
torch.save({'model':model.state_dict()}, 'checkpoints/first.pt')
with torch.no_grad(): model.vocabulary.bias[5] = 20
torch.save({'model':model.state_dict()}, 'checkpoints/latest.pt')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=directory,
        env={**os.environ, "USE_TORCH": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    with Run(directory) as run:
        run.step = 1
        run.checkpoint(directory / "checkpoints/first.pt")
        run.log({"eval/loss": 0.3}, step=1)
        run.step = 2
        run.checkpoint(directory / "checkpoints/latest.pt")
        run.log({"eval/loss": 0.7}, step=2)
    return service, directory


def wait_ready(chat, run_id, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = chat.status(run_id)
        if status["state"] not in ("loading", "generating"):
            return status
        time.sleep(0.05)
    pytest.fail(f"Chat worker did not respond: {chat.status(run_id)}")


def test_chat_recommends_best_heldout_checkpoint_and_preserves_training_recipe(
    tmp_path,
):
    service, directory = make_chat_run(tmp_path)
    original = (directory / "train.py").read_bytes()
    catalog = checkpoint_catalog(directory)
    assert catalog["recommended_checkpoint"] == "checkpoints/first.pt"
    assert catalog["checkpoints"][0]["step"] == 2
    assert not catalog["problem"]
    chat = ChatService()
    try:
        state = chat.load(directory, ChatLoad())
        assert state["state"] == "loading"
        state = wait_ready(chat, directory.name)
        assert state["state"] == "ready", state
        assert state["has_chat_template"]
        chat.send(
            directory.name, ChatMessage(text="hello", max_new_tokens=3, temperature=0)
        )
        state = wait_ready(chat, directory.name)
        assert state["state"] == "ready" and not state["error"], state
        assert [m["role"] for m in state["messages"]] == ["user", "assistant"]
        assert state["finish_reason"] == "length"
        assert state["messages"][-1]["content"] == "world world world"
        # Reuse the loaded model and send the previous conversation as context.
        pid = chat.process.pid
        chat.send(
            directory.name, ChatMessage(text="world", max_new_tokens=2, temperature=0)
        )
        state = wait_ready(chat, directory.name)
        assert len(state["messages"]) == 4 and chat.process.pid == pid
        chat.send(
            directory.name,
            ChatMessage(text="hello world", context_length=2, max_new_tokens=2),
        )
        state = wait_ready(chat, directory.name)
        assert "token budget" in state["error"]
        assert state["state"] == "ready"
        chat.clear(directory.name)
        assert not chat.status(directory.name)["messages"]
        chat.send(
            directory.name,
            ChatMessage(
                text="hello",
                mode="completion",
                max_new_tokens=2048,
                context_length=32768,
                temperature=0,
            ),
        )
        chat.stop(directory.name)
        state = wait_ready(chat, directory.name)
        assert state["finish_reason"] == "stopped"
        assert (directory / "train.py").read_bytes() == original
        transcripts = list((directory / "chat").glob("*/transcript.json"))
        assert (
            transcripts
            and json.loads(transcripts[0].read_text())["checkpoint"]
            == "checkpoints/first.pt"
        )
        archive = [
            json.loads(line)
            for line in (transcripts[0].parent / "conversations.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(archive) >= 4
        assert archive[-1]["generation"]["mode"] == "completion"
        assert archive[0]["messages"][0]["content"] == "hello"
    finally:
        process = chat.process
        chat.close()
        assert process.poll() is not None
    assert chat.status(directory.name)["state"] == "unloaded"


def test_chat_requires_local_published_weights_and_verifies_checkpoint_hash(tmp_path):
    service, directory = make_chat_run(tmp_path, chat_template=False)
    chat = ChatService()
    try:
        with pytest.raises(ValueError, match="published checkpoint"):
            chat.load(directory, ChatLoad(checkpoint="../../train.py"))
        chat.load(directory, ChatLoad())
        state = wait_ready(chat, directory.name)
        assert state["state"] == "ready" and not state["has_chat_template"]
        with pytest.raises(ValueError, match="no chat template"):
            chat.send(directory.name, ChatMessage(text="hello"))
        chat.send(
            directory.name,
            ChatMessage(
                text="hello", mode="completion", temperature=0, max_new_tokens=1
            ),
        )
        assert not wait_ready(chat, directory.name)["error"]
        chat.send(
            directory.name,
            ChatMessage(
                text="hello",
                chat_template="{{ messages[-1]['content'] }}",
                temperature=0,
                max_new_tokens=1,
            ),
        )
        assert not wait_ready(chat, directory.name)["error"]
        chat.close()
        config = json.loads((directory / "config.json").read_text())
        config["chat_template"] = "{{ messages[-1]['content'] }}"
        (directory / "config.json").write_text(json.dumps(config))
        chat.load(directory, ChatLoad())
        assert wait_ready(chat, directory.name)["has_chat_template"]
        chat.send(
            directory.name, ChatMessage(text="hello", max_new_tokens=1, temperature=0)
        )
        assert not wait_ready(chat, directory.name)["error"]
        chat.close()
        (directory / "checkpoints/first.pt").write_bytes(b"changed")
        chat.load(directory, ChatLoad())
        state = wait_ready(chat, directory.name)
        assert state["state"] == "error" and "hash differs" in state["error"]
    finally:
        chat.close()


def test_chat_api_honors_execution_switch_authentication_and_origin(
    tmp_path, monkeypatch
):
    service, directory = make_chat_run(tmp_path)
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "test-secret")
    monkeypatch.delenv("WEAVE_TRAINING_ENABLED", raising=False)
    app = FastAPI()
    app.include_router(router(tmp_path))
    path = "/api/training/runs/" + directory.name + "/chat"
    with TestClient(app) as client:
        assert client.get(path).status_code == 401
        client.post("/api/training/session", json={"token": "test-secret"})
        result = client.get(path)
        assert (
            result.status_code == 200
            and result.json()["recommended_checkpoint"] == "checkpoints/first.pt"
        )
        assert client.post(path + "/load", json={}).status_code == 403
        monkeypatch.setenv("WEAVE_TRAINING_ENABLED", "1")
        assert (
            client.post(
                path + "/load", json={}, headers={"origin": "http://evil.example"}
            ).status_code
            == 403
        )
        assert (
            client.post(path + "/load", json={"device": "cuda:-1"}).status_code == 422
        )
        assert (
            client.post(
                path + "/load", json={"checkpoint": "../../train.py"}
            ).status_code
            == 422
        )
        assert client.post(path + "/messages", json={"text": "hi"}).status_code == 409
        assert client.delete(path).status_code == 200


def test_checkpoint_recommendation_fallback_and_unavailable_weights(tmp_path):
    from backend.training.weave_training import atomic_json

    project = tiny_project()
    atomic_json(tmp_path / "project.json", project)
    atomic_json(tmp_path / "tokenizers.json", {"ids": {"id": "saved"}})
    for name in ("first.pt", "second.pt"):
        (tmp_path / name).write_bytes(b"weights")
    with Run(tmp_path) as run:
        run.step = 1
        run.checkpoint(tmp_path / "first.pt")
        run.step = 2
        run.checkpoint(tmp_path / "second.pt")
        run.emit("checkpoint", {"path": "../outside.pt", "sha256": "unused"})
        run.emit(
            "checkpoint", {"path": "remote.pt", "sha256": "unused", "remote": True}
        )
    result = checkpoint_catalog(tmp_path)
    assert result["recommended_checkpoint"] == "second.pt"
    assert "none has a matching validation" in result["recommendation"]
    assert not next(
        item for item in result["checkpoints"] if item["path"] == "../outside.pt"
    )["available"]
    assert not next(
        item for item in result["checkpoints"] if item["path"] == "remote.pt"
    )["available"]
    (tmp_path / "second.pt").unlink()
    assert checkpoint_catalog(tmp_path)["recommended_checkpoint"] == "first.pt"


@pytest.mark.skipif(
    not os.environ.get("WEAVE_CUDA_PYTHON"), reason="Requires CUDA environment"
)
def test_checkpoint_chat_on_selected_gpu_and_seeded_completion(tmp_path):
    service, directory = make_chat_run(tmp_path)
    config = json.loads((directory / "config.json").read_text())
    config["python"] = os.environ["WEAVE_CUDA_PYTHON"]
    (directory / "config.json").write_text(json.dumps(config))
    chat = ChatService()
    try:
        chat.load(directory, ChatLoad(device="cuda:1"))
        assert wait_ready(chat, directory.name)["state"] == "ready"
        request = ChatMessage(
            text="hello",
            mode="completion",
            max_new_tokens=4,
            temperature=1,
            top_p=1,
            seed=123,
        )
        chat.send(directory.name, request)
        first = wait_ready(chat, directory.name)
        assert not first["error"], first
        assert first["device"] == "cuda:1"
        text = first["messages"][-1]["content"]
        chat.clear(directory.name)
        chat.send(directory.name, request)
        assert wait_ready(chat, directory.name)["messages"][-1]["content"] == text
    finally:
        chat.close()
