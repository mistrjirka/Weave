import json
import sqlite3
import threading

from backend.training import service


def event(identifier, kind, **payload):
    return {"version": 1, "id": identifier, "kind": kind, "time": 1, "step": 0, "tokens": 0, "payload": payload}


def test_decode_does_not_hold_writer_lock(tmp_path, monkeypatch):
    (tmp_path / "events.jsonl").write_text(json.dumps(event("a", "started")) + "\n")
    service.events(tmp_path)
    with (tmp_path / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event("b", "finished")) + "\n")
    entered, release = threading.Event(), threading.Event()
    original = service.json.loads

    def slow_loads(value, *args, **kwargs):
        if isinstance(value, bytes):
            entered.set()
            assert release.wait(5)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(service.json, "loads", slow_loads)
    errors = []
    def reader():
        try:
            service.events(tmp_path)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert entered.wait(5)
        with sqlite3.connect(tmp_path / "event-index.sqlite", timeout=0.1) as connection:
            connection.execute("BEGIN IMMEDIATE")
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert [e["id"] for e in service.events(tmp_path)] == ["a", "b"]


def test_external_status_latest_lifecycle_and_terminal_without_index(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"execution": "monitor"}))
    records = [event("old", "stopped"), event("start", "started"), event("end", "finished", success=True)]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in records))
    instance = object.__new__(service.TrainingService)
    assert instance.status(tmp_path)["state"] == "completed"
    (tmp_path / "status.json").write_text(json.dumps({"state": "killed"}))
    with sqlite3.connect(tmp_path / "event-index.sqlite") as connection:
        connection.execute("BEGIN EXCLUSIVE")
        assert instance.status(tmp_path)["state"] == "killed"
