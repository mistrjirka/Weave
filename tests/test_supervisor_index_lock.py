"""A busy event index must not stop a healthy training run.

API, SSE and MCP readers take the index's SQLite writer lock. When a reader holds
it past the timeout, the supervisor must skip that tick instead of failing the run
and terminating the training processes.
"""

import json
import sqlite3

import pytest

from backend.training import supervisor
from backend.training.service import events


def write_events(directory):
    records = [
        {
            "version": 1,
            "id": f"event-{index}",
            "kind": "metrics",
            "time": index + 1,
            "step": index,
            "tokens": index * 8,
            "payload": {"train/loss": 2.0},
            "channel": "training",
        }
        for index in range(3)
    ]
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records)
    )


def test_locked_index_returns_none_then_recovers(tmp_path, monkeypatch):
    write_events(tmp_path)
    events(tmp_path)  # create the index
    with (tmp_path / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"version": 1, "id": "event-3", "kind": "metrics", "time": 4, "step": 3, "tokens": 24, "payload": {}}) + "\n")

    connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda path, timeout=5.0, **kw: connect(path, timeout=0.1, **kw)
    )
    holder = connect(tmp_path / "event-index.sqlite")
    holder.execute("BEGIN IMMEDIATE")
    try:
        assert supervisor.index_events(tmp_path, kind="metrics") is None
    finally:
        holder.rollback()
        holder.close()

    found = supervisor.index_events(tmp_path, kind="metrics")
    assert [item["id"] for item in found] == ["event-0", "event-1", "event-2", "event-3"]


def test_current_index_reads_under_reserved_lock(tmp_path):
    write_events(tmp_path)
    events(tmp_path)
    with sqlite3.connect(tmp_path / "event-index.sqlite") as holder:
        holder.execute("BEGIN IMMEDIATE")
        assert len(events(tmp_path, timeout=0.01)) == 3
        with (tmp_path / "events.jsonl").open("a") as stream:
            stream.write('{"unfinished":')
        assert len(events(tmp_path, timeout=0.01)) == 3


def test_other_database_errors_still_fail(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(supervisor, "events", broken)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        supervisor.index_events(tmp_path, kind="metrics")
