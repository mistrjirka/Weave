"""Resource lifetime and event-loop regressions found in the implementation review."""

import asyncio
import sqlite3
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

import backend.app as server
from backend.training.reporting import ReportBatch


def test_mcp_transport_survives_sequential_app_lifespans():
    headers = {
        "host": "127.0.0.1:8173",
        "accept": "application/json, text/event-stream",
    }
    for _ in range(2):
        with TestClient(server.app) as client:
            response = client.post(
                "/mcp/",
                headers=headers,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
            assert response.status_code == 200
            assert "read_weave_doc" in response.text


def test_project_database_closes_and_rolls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path)
    with server.db() as committed:
        committed.execute(
            "INSERT INTO projects(id,name,document) VALUES('saved','Saved','{}')"
        )
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        committed.execute("SELECT 1")

    with pytest.raises(RuntimeError):
        with server.db() as failed:
            failed.execute(
                "INSERT INTO projects(id,name,document) VALUES('failed','Failed','{}')"
            )
            raise RuntimeError("rollback")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        failed.execute("SELECT 1")
    with server.db() as check:
        assert check.execute("SELECT id FROM projects").fetchall() == [("saved",)]


def test_slow_tokenizer_upload_does_not_block_health(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path)
    original_write = server.Path.write_bytes
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def write(path, content):
        if path.parent.parent.name == "uploads":
            assert threading.get_ident() != loop_thread
            started.set()
            assert release.wait(5), "Health check never released the upload"
        return original_write(path, content)

    class Tokenizers:
        def load(self, source, display_name):
            return {"display_name": display_name}

    monkeypatch.setattr(server.Path, "write_bytes", write)
    monkeypatch.setattr(server, "tokenizers", Tokenizers())

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            upload = asyncio.create_task(
                client.post(
                    "/api/tokenizers/upload",
                    files={"files": ("tokenizer.json", b"{}", "application/json")},
                )
            )
            try:
                assert await asyncio.to_thread(started.wait, 5)
                assert not upload.done()
                health = await asyncio.wait_for(client.get("/api/health"), timeout=2)
                assert health.status_code == 200
            finally:
                release.set()
                result = await upload
            assert result.status_code == 200
            assert result.json()["display_name"] == "tokenizer.json"
            assert not list((tmp_path / "uploads").iterdir())

    asyncio.run(check())


def test_report_validation_runs_outside_event_loop(monkeypatch):
    original = ReportBatch.model_validate_json
    loop_thread = threading.get_ident()
    calls = []

    def validate(cls, content, *args, **kwargs):
        calls.append(threading.get_ident())
        assert calls[-1] != loop_thread
        return original(content, *args, **kwargs)

    monkeypatch.setattr(ReportBatch, "model_validate_json", classmethod(validate))

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            # Validation precedes the missing-run response; no real run is changed.
            response = await client.post(
                "/api/training/reports/review-missing-run", json={"events": []}
            )
            assert response.status_code == 404

    asyncio.run(check())
    assert len(calls) == 1
