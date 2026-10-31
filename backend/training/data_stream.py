"""Optional resumable packed stream. Supply your own row tokenizer or replace it.

Only num_workers=0 allows dataset cursor snapshots; prefetching workers require
an external stateful DataLoader. Batch metadata identifies source rows and token spans.
"""

import glob
import json
import os
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from text_training import load_rows


class TokenStream(IterableDataset):
    distributed_sharding = True  # The iterator shards rows across ranks/workers.

    def __init__(self, pattern, tokenize, seq_len):
        self.files = sorted(
            str(Path(p).expanduser().resolve())
            for p in glob.glob(os.path.expanduser(pattern))
        )
        if not self.files:
            raise ValueError(
                f"No data files match {pattern!r}; check the path or glob in your data loader"
            )
        self.encode, self.seq_len = tokenize, seq_len
        self.reset()

    def reset(self):
        self.file_index = self.row_index = 0
        self.ids, self.labels, self.origins = [], [], []

    # Dataset state is usable with num_workers=0. For worker prefetching,
    # supply a stateful loader with its own state_dict/load_state_dict.
    def state_dict(self):
        return {
            name: getattr(self, name)
            for name in ("file_index", "row_index", "ids", "labels", "origins")
        }

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def chunks(self):
        while len(self.ids) >= self.seq_len + 1:
            ids, labels = self.ids[: self.seq_len + 1], self.labels[: self.seq_len + 1]
            source, remaining = [], self.seq_len + 1
            for origin in self.origins:
                count = min(remaining, origin["tokens"])
                if count:
                    source.append({**origin, "tokens": count})
                    remaining -= count
            self.ids, self.labels = (
                self.ids[self.seq_len :],
                self.labels[self.seq_len :],
            )
            consumed = self.seq_len
            while self.origins and consumed >= self.origins[0]["tokens"]:
                consumed -= self.origins.pop(0)["tokens"]
            if consumed:
                self.origins[0] = {
                    **self.origins[0],
                    "tokens": self.origins[0]["tokens"] - consumed,
                    "offset": self.origins[0]["offset"] + consumed,
                }
            if any(label != -100 for label in labels[1:]):
                yield {
                    "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                    "labels": torch.tensor(labels[1:], dtype=torch.long),
                    "_weave_source": json.dumps(source),
                }

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(
            os.environ.get("RANK", "0")
        )
        shard, shards = rank * workers + worker_id, world * workers
        yield from self.chunks()
        for file_index in range(self.file_index, len(self.files)):
            path = self.files[file_index]
            skip = self.row_index if file_index == self.file_index else 0
            for row_index, row in enumerate(load_rows(path)):
                if row_index < skip or row_index % shards != shard:
                    continue
                ids, labels = self.encode(row)
                if len(ids) != len(labels):
                    raise ValueError(
                        f"Mismatched input_ids/labels at {path}:{row_index}"
                    )
                self.file_index, self.row_index = file_index, row_index + 1
                self.ids.extend(ids)
                self.labels.extend(labels)
                if ids:
                    self.origins.append(
                        {
                            "path": path,
                            "row": row_index,
                            "offset": 0,
                            "tokens": len(ids),
                        }
                    )
                yield from self.chunks()
            self.file_index, self.row_index = file_index + 1, 0
