"""Checkpoint publication and cooperative command handling."""

import json
from pathlib import Path

if __package__:
    from .weave_training_files import file_hash
else:
    from weave_training_files import file_hash


class _CommandCheckpointMixin:
    def checkpoint(self, path, *, metadata=None):
        path = Path(path).resolve()
        relative = path.relative_to(self.directory)
        if not path.is_file():
            raise ValueError(
                "Publish a completed checkpoint file inside the run directory."
            )
        self.emit(
            "checkpoint",
            {
                "path": str(relative),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "metadata": metadata or {},
            },
        )

    def poll_commands(self):
        """Poll only at safe optimizer-step boundaries. Never interrupt backward."""
        commands = []
        for path in sorted((self.directory / "commands").glob("*.json")):
            if path.name in self.commands_seen:
                continue
            command = json.loads(path.read_text())
            self.commands_seen.add(path.name)
            commands.append(command)
        return commands

    def handle_commands(self, checkpoint_fn, *, commands=None):
        """Save at a user-chosen safe boundary, acknowledge requests, return stop.

        checkpoint_fn(snapshot_requests=[...]) must publish its completed file.
        In DDP all ranks must call this with DistributedContext.poll_commands().
        """
        commands = self.poll_commands() if commands is None else commands
        pending = [item for item in commands if item["kind"] in ("stop", "snapshot")]
        if not pending:
            return False
        checkpoint_fn(snapshot_requests=[item["id"] for item in pending])
        for item in pending:
            self.emit("command_ack", {"id": item["id"], "kind": item["kind"]})
        stopped = any(item["kind"] == "stop" for item in pending)
        if stopped:
            self.emit("stopped", {"reason": "Cooperative stop after checkpoint"})
        return stopped
