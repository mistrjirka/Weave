"""MCP discovery, transport, and training credential boundaries."""

import asyncio
import json
import socket
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import app, mcp_server
from backend.mcp_bridge import WeaveBridge, mcp_transport_security


def run(coroutine):
    return asyncio.run(coroutine)


def result(tool_result):
    return json.loads(tool_result.content[0].text)


def rpc_result(response):
    message = json.loads(next(line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")))
    return json.loads(message["result"]["content"][0]["text"])


def test_mcp_operations_and_openapi_schemas():
    bridge = WeaveBridge(app=app)
    operations = run(bridge.operations())
    assert any(item["path"] == "/api/projects/{project_id}" for item in operations.values())
    assert any(item["path"] == "/api/training/runs/{run_id}/commands" for item in operations.values())
    assert not any(item["path"] == "/api/export" for item in operations.values())
    assert not any(item["path"] == "/api/training/session" for item in operations.values())
    operation_id = next(key for key, value in operations.items() if value["path"] == "/api/training/runs" and value["method"] == "POST")
    description = result(run(mcp_server.call_tool("describe_weave_operation", {"operation_id": operation_id})))
    assert description["schemas"]


def test_read_and_change_tools_are_separate():
    bridge = WeaveBridge(app=app)
    operation_id = next(key for key, value in run(bridge.operations()).items() if value["path"] == "/api/health")
    assert run(bridge.execute(operation_id, None, None, None, None, read_only=True)) == {"status": 200, "data": {"status": "ok", "version": 1}}
    assert "error" in run(bridge.execute(operation_id, None, None, None, None, read_only=False))
    assert result(run(mcp_server.call_tool("read_weave_doc", {"guide": "../pyproject.toml"})))["error"] == "Unknown guide"


def test_training_token_is_never_inherited_by_http_callers(monkeypatch):
    monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "operator-secret")
    assert WeaveBridge(app=app).credentials(None) == {}
    assert WeaveBridge().credentials(None) == {"x-weave-training-token": "operator-secret"}
    caller = SimpleNamespace(headers={"x-weave-training-token": "caller-token", "cookie": "other=private; weave_training=caller-cookie"})
    assert WeaveBridge(app=app).credentials(caller) == {"x-weave-training-token": "caller-token", "cookie": "weave_training=caller-cookie"}


def test_mounted_mcp_initializes_lists_tools_and_enforces_training_auth(monkeypatch):
    headers = {"host": "127.0.0.1:8173", "accept": "application/json, text/event-stream", "content-type": "application/json"}
    with TestClient(app) as client:
        initialized = client.post("/mcp/", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
        assert initialized.status_code == 200
        assert "Weave Neural Graph Studio" in initialized.text
        listed = client.post("/mcp/", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert listed.status_code == 200
        assert "execute_weave_operation" in listed.text
        called = client.post("/mcp/", headers=headers, json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_weave_operation", "arguments": {"operation_id": "health_api_health_get"}}})
        assert called.status_code == 200
        assert rpc_result(called)["data"] == {"status": "ok", "version": 1}

        monkeypatch.setenv("WEAVE_TRAINING_TOKEN", "operator-secret")
        bridge = WeaveBridge(app=app)
        operation_id = next(key for key, value in run(bridge.operations()).items()
                            if value["path"] == "/api/training/templates" and value["method"] == "GET")

        def call(**extra_headers):
            return client.post("/mcp/", headers={**headers, **extra_headers}, json={
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "read_weave_operation", "arguments": {"operation_id": operation_id}},
            })

        assert rpc_result(call())["status"] == 401
        assert rpc_result(call(**{"x-weave-training-token": "wrong"}))["status"] == 401
        assert rpc_result(call(**{"x-weave-training-token": "operator-secret"}))["status"] == 200
        assert rpc_result(call(cookie="weave_training=operator-secret"))["status"] == 200
        assert call(origin="http://attacker.example").status_code == 403


def test_mcp_host_policy_includes_lan_and_rejects_unrelated_hosts(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "weave-box")
    monkeypatch.setattr("backend.mcp_bridge.psutil.net_if_addrs", lambda: {
        "eth0": [SimpleNamespace(family=socket.AF_INET, address="10.8.0.8")],
    })
    security = mcp_transport_security()
    assert security.enable_dns_rebinding_protection
    assert "10.8.0.8:*" in security.allowed_hosts
    assert "http://10.8.0.8:*" in security.allowed_origins
    assert "weave-box:*" in security.allowed_hosts
    assert "attacker.example:*" not in security.allowed_hosts


def test_mcp_container_public_origin_is_explicit(monkeypatch):
    monkeypatch.setenv("WEAVE_MCP_PUBLIC_ORIGINS", "https://weave.example.com,http://192.168.1.10:8173")
    security = mcp_transport_security()
    assert "weave.example.com" in security.allowed_hosts
    assert "192.168.1.10:8173" in security.allowed_hosts
    assert "https://weave.example.com" in security.allowed_origins
    assert "http://weave.example.com" not in security.allowed_origins
    for invalid in ("https://*.example.com", "https://weave.example.com/path", "https://user:pass@example.com", "http://example.com:99999"):
        monkeypatch.setenv("WEAVE_MCP_PUBLIC_ORIGINS", invalid)
        with pytest.raises(ValueError):
            mcp_transport_security()


