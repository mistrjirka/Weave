"""A newly launched run is read concurrently by SSE, status and dashboard APIs."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

from backend.training.service import events


def test_simultaneous_index_creation_and_incremental_reads(tmp_path):
    for generation in range(8):
        directory = tmp_path / str(generation)
        directory.mkdir()
        path = directory / "events.jsonl"
        records = [
            {
                "version": 1,
                "id": f"event-{index}",
                "kind": "metrics",
                "time": index + 1,
                "step": index,
                "tokens": index * 8,
                "payload": {"train/loss": 2.0},
            }
            for index in range(30)
        ]
        path.write_text("".join(json.dumps(item) + "\n" for item in records))
        gate = Barrier(8)

        def read(_):
            gate.wait(timeout=10)
            return events(directory, after="")

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(8)))
        assert all(
            [item["id"] for item in batch] == [item["id"] for item in records]
            for batch in results
        )
        new = {**records[-1], "id": "next-event", "step": 30}
        with path.open("a") as output:
            output.write(json.dumps(new) + "\n")
        gate = Barrier(8)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(8)))
        assert all(
            len(batch) == 31 and batch[-1]["id"] == "next-event" for batch in results
        )
