"""Scoped, reporting-only ingestion for independently launched training scripts."""

from __future__ import annotations

import hmac
from contextlib import closing
import json
import os
import secrets
import time
from threading import RLock
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat

from .weave_training import atomic_json


class ReportEvent(BaseModel):
    version: int = Field(ge=1, le=1)
    id: str = Field(min_length=1, max_length=128)
    time: FiniteFloat
    kind: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0)
    tokens: int = Field(ge=0)
    payload: dict
    channel: Literal["training", "evaluation", "system"] = "training"


class ReportBatch(BaseModel):
    events: list[ReportEvent] = Field(max_length=100)


class ReportStore:
    def __init__(self):
        self.lock = RLock()

    def connection(self, directory, base_url):
        with self.lock:
            path = directory / ".report-token"
            if not path.exists():
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(descriptor, "w") as stream:
                    stream.write(secrets.token_urlsafe(32))
            config = {
                "url": base_url.rstrip("/") + "/api/training/reports/" + directory.name,
                "token": path.read_text(),
            }
            atomic_json(directory / "reporting.json", config)
            (directory / "reporting.json").chmod(0o600)
            return config

    def ingest(self, directory, token, batch):
        from .service import events
        import sqlite3

        path = directory / ".report-token"
        if not path.exists() or not hmac.compare_digest(token, path.read_text()):
            raise PermissionError("Invalid run reporting key")
        info = json.loads((directory / "run.json").read_text())
        if info["execution"] != "monitor":
            raise ValueError("Independent reports require a terminal / external run")
        records = []
        for item in batch.events:
            event = item.model_dump()
            payload = event["payload"]
            if event["kind"] == "checkpoint":
                payload["remote"] = True
            if event["kind"] == "capabilities":
                # This API does not transfer checkpoint bytes back to the host.
                payload["checkpoint"] = False
                payload["snapshot"] = False
            line = json.dumps(event, allow_nan=False)
            if len(line.encode()) > 1024 * 1024:
                raise ValueError("Reporting events must be smaller than 1 MB")
            records.append(line)
        with self.lock:
            # The durable event index deduplicates retries by event ID.
            events(directory, limit=0)
            with closing(
                sqlite3.connect(directory / "event-index.sqlite", timeout=30)
            ) as connection, connection:
                ids = [event.id for event in batch.events]
                existing = (
                    {
                        row[0]
                        for row in connection.execute(
                            "SELECT id FROM events WHERE id IN ("
                            + ",".join("?" for _ in ids)
                            + ")",
                            ids,
                        )
                    }
                    if ids
                    else set()
                )
            accepted = []
            for event, line in zip(batch.events, records):
                if event.id not in existing:
                    accepted.append(line)
                    existing.add(event.id)
            with (directory / "events.jsonl").open("a") as output:
                output.write("".join(line + "\n" for line in accepted))
                output.flush()
                os.fsync(output.fileno())
            atomic_json(
                directory / "reporting-status.json",
                {
                    "remote": True,
                    "last_received": time.time(),
                },
            )
        commands = [
            json.loads(path.read_text())
            for path in sorted((directory / "commands").glob("*.json"))
        ]
        return {"accepted": len(accepted), "commands": commands}
