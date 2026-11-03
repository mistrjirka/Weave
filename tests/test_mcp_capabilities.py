"""MCP covers graph calculations, bounded transfers and cursor-based monitoring."""

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import backend.app as server
from backend.mcp_bridge import WeaveBridge, create_mcp_server
from backend.mcp_files import UploadFileContent, download_file, upload_files
from backend.tokenizers import TokenizerService
from backend.training.api import router
from backend.training.service import build_snapshot, events
from test_training import tiny_project
from test_training_reporting import prepare

ROOT = Path(__file__).resolve().parents[1]


def run(awaitable):
    return asyncio.run(awaitable)


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    monkeypatch.delenv("WEAVE_TRAINING_TOKEN", raising=False)
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setattr(server, "tokenizers", TokenizerService(tmp_path / "tokenizers"))
    app = FastAPI()
    app.add_api_route("/api/validate", server.validate, methods=["POST"])
    app.add_api_route("/api/export", server.export, methods=["POST"])
    app.add_api_route(
        "/api/tokenizers/upload", server.upload_tokenizer, methods=["POST"]
    )
    app.include_router(router(tmp_path))
    with TestClient(app) as client:
        yield client, WeaveBridge(app=app), tmp_path


def operation(bridge, path):
    return next(
        key for key, entry in run(bridge.operations()).items() if entry["path"] == path
    )


def test_read_only_classification_enforced_and_pure_posts_work(workbench, monkeypatch):
    _, bridge, _ = workbench
    mcp = create_mcp_server(bridge)
    env = operation(bridge, "/api/training/environment")
    assert run(bridge.operations())[env]["read_only"] is False
    # The forbidden read must never reach interpreter inspection.
    monkeypatch.setattr(
        "backend.training.api.inspect_python",
        lambda _: pytest.fail("Executed an interpreter"),
    )
    denied = run(
        mcp.call_tool(
            "read_weave_operation",
            {"operation_id": env, "query": {"python": "/bin/sh"}},
        )
    )
    assert denied.is_error
    assert "execute_weave_operation" in denied.structured_content["error"]
    validate = operation(bridge, "/api/validate")
    result = run(
        mcp.call_tool(
            "read_weave_operation",
            {"operation_id": validate, "body": tiny_project()},
            context=SimpleNamespace(headers={}),
        )
    )
    assert not result.is_error
    assert result.structured_content["data"]["valid"]


def test_api_failures_are_mcp_tool_errors(workbench, monkeypatch):
    _, bridge, _ = workbench
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "operator-secret")
    mcp = create_mcp_server(bridge)
    templates = operation(bridge, "/api/training/templates")
    result = run(
        mcp.call_tool(
            "read_weave_operation",
            {"operation_id": templates},
            context=SimpleNamespace(headers={}),
        )
    )
    assert result.is_error
    assert result.structured_content["status"] == 401
    assert "access code" in result.structured_content["data"]["detail"]
    ctx = SimpleNamespace(headers={"x-weave-training-token": "operator-secret"})
    success = run(bridge.execute(templates, None, None, None, ctx, read_only=True))
    assert success["status"] == 200


def test_event_pages_status_and_transcript_are_json_readable(workbench):
    client, bridge, root = workbench
    run_id = prepare(client)
    directory = root / "training/runs" / run_id
    records = [
        {
            "version": 1,
            "id": f"record-{i}",
            "kind": "metrics",
            "time": i + 1,
            "step": i,
            "tokens": i * 8,
            "payload": {"train/loss": 2.0},
            "channel": "training",
        }
        for i in range(5)
    ]
    (directory / "events.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    pages, after = [], ""
    for _ in range(4):
        response = client.get(
            f"/api/training/runs/{run_id}/event-records",
            params={"after": after, "limit": 2},
        )
        assert response.status_code == 200
        batch = response.json()
        pages.extend(batch["events"])
        after = batch["next_after"]
        if not batch["events"]:
            break
    assert [item["id"] for item in pages] == [item["id"] for item in records]
    status = client.get(f"/api/training/runs/{run_id}/status").json()
    assert status["latest_metrics"]["id"] == "record-4"
    assert "logs" not in status and "events" not in status
    assert (
        client.get(f"/api/training/runs/{run_id}/event-records?limit=501").status_code
        == 422
    )
    transcript = operation(bridge, "/api/training/runs/{run_id}/chat/export")
    result = run(
        bridge.execute(transcript, {"run_id": run_id}, None, None, None, read_only=True)
    )
    assert result["status"] == 200
    assert result["data"]["records"][0]["run_id"] == run_id
    (directory / "events.jsonl").write_text(
        "\n".join(
            json.dumps({**item, "payload": {"large": "x" * 1500}}) for item in records
        )
        + "\n"
    )
    # Use a fresh index after replacing the test stream.
    (directory / "event-index.sqlite").unlink()
    bounded = events(directory, after="", limit=100, max_bytes=4000)
    assert 0 < len(bounded) < len(records)
    with pytest.raises(ValueError, match="response budget"):
        events(directory, after="", limit=100, max_bytes=10)


def test_upload_bundle_preserves_companion_files_and_auth(workbench, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    _, bridge, root = workbench
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "</s>": 2}, unk_token="[UNK]")
    )
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "</s>",
        "unk_token": "[UNK]",
        "chat_template": "{{ messages[0]['content'] }}",
    }
    files = [
        UploadFileContent(filename=name, content_base64=base64.b64encode(data).decode())
        for name, data in [
            ("tokenizer.json", tokenizer.to_str().encode()),
            ("tokenizer_config.json", json.dumps(config).encode()),
        ]
    ]
    mcp = create_mcp_server(bridge)
    called = run(
        mcp.call_tool(
            "upload_weave_files",
            {"kind": "tokenizer", "files": [item.model_dump() for item in files]},
            context=SimpleNamespace(headers={}),
        )
    )
    assert not called.is_error
    result = called.structured_content
    assert result["status"] == 200
    saved = result["data"]
    assert saved["capacity"] == 3
    assert saved["special_tokens"]["eos_token"]["id"] == 2
    from transformers import AutoTokenizer

    stored = AutoTokenizer.from_pretrained(
        root / "tokenizers" / saved["id"],
        local_files_only=True,
        trust_remote_code=False,
    )
    assert stored.chat_template == config["chat_template"]
    invalid = UploadFileContent(filename="../data.txt", content_base64="eA==")
    assert "error" in run(upload_files(bridge, "dataset", [invalid], None))
    invalid = UploadFileContent(filename="data.txt", content_base64="not base64")
    assert "error" in run(upload_files(bridge, "dataset", [invalid], None))
    file = UploadFileContent(filename="data.txt", content_base64="eA==")
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "operator-secret")
    assert run(upload_files(bridge, "dataset", [file], None))["status"] == 401
    ctx = SimpleNamespace(headers={"x-weave-training-token": "operator-secret"})
    assert run(upload_files(bridge, "dataset", [file], ctx))["status"] == 200
    monkeypatch.setattr("backend.mcp_files.MAX_FILE_BYTES", 0)
    assert "4 MiB" in run(upload_files(bridge, "dataset", [file], ctx))["error"]


def test_exports_and_run_artifacts_remain_scoped(workbench):
    client, bridge, root = workbench
    called = run(
        create_mcp_server(bridge).call_tool(
            "download_weave_file",
            {"kind": "model", "project": tiny_project()},
            context=SimpleNamespace(headers={}),
        )
    )
    assert not called.is_error
    metadata, content = called.structured_content, called.content[1:]
    assert metadata["status"] == 200
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(content[0].resource.blob)))
    assert "model.py" in archive.namelist()
    run_id = prepare(client)
    text, content = run(
        download_file(bridge, "artifact", None, run_id, "train.py", None)
    )
    assert text["status"] == 200
    assert "print" in content[0].resource.text
    denied, _ = run(
        download_file(bridge, "artifact", None, run_id, "../../outside.txt", None)
    )
    assert denied["status"] == 422
    denied, _ = run(download_file(bridge, "artifact", None, run_id, "weights.pt", None))
    assert "error" in denied
    package, content = run(
        download_file(bridge, "training_package", None, run_id, None, None)
    )
    assert package["status"] == 200
    assert (
        "train.py"
        in zipfile.ZipFile(
            io.BytesIO(base64.b64decode(content[0].resource.blob))
        ).namelist()
    )
    snapshot = build_snapshot(root / "training/runs" / run_id)
    blob = (root / "training/runs" / run_id / snapshot).read_bytes()
    imported = run(
        upload_files(
            bridge,
            "snapshot",
            [
                UploadFileContent(
                    filename="snapshot.zip",
                    content_base64=base64.b64encode(blob).decode(),
                )
            ],
            None,
        )
    )
    assert imported["status"] == 200


def test_monitoring_endpoints_require_caller_auth(workbench, monkeypatch):
    client, _, _ = workbench
    run_id = prepare(client)
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "operator-secret")
    for route in ("status", "event-records"):
        url = f"/api/training/runs/{run_id}/{route}"
        assert client.get(url).status_code == 401
        assert (
            client.get(
                url, headers={"x-weave-training-token": "operator-secret"}
            ).status_code
            == 200
        )


def test_resources_prompts_and_copyable_configurations():
    mcp = create_mcp_server(WeaveBridge())
    assert "weave://docs/index" in [str(item.uri) for item in run(mcp.list_resources())]
    assert "design_weave_model" in [item.name for item in run(mcp.list_prompts())]
    assert (
        "Weave documentation" in run(mcp.read_resource("weave://docs/index"))[0].content
    )
    assert (
        "Writing train.py"
        in run(mcp.read_resource("weave://guides/TRAINING_GUIDE.md"))[0].content
    )
    for filename in ("codex-http.toml", "codex-stdio.toml"):
        value = tomllib.loads((ROOT / "docs/examples" / filename).read_text())[
            "mcp_servers"
        ]["weave"]
        assert value["tool_timeout_sec"] == 120
    value = json.loads((ROOT / "docs/examples/mcp-client.json").read_text())
    assert value["mcpServers"]["weave"]["args"] == ["-m", "backend.mcp_bridge"]
