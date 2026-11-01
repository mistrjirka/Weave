"""CPU synchronization for data hooks that may build or wait for large caches."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta

import torch.distributed as dist

_group = ContextVar("weave_preparation_group", default=None)


def preparation_barrier():
    """Synchronize data preparation without occupying an NCCL collective.

    Call on every rank, in the same order, from a Trainer data-loader hook.
    Single-process preparation does not need a barrier.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    group = _group.get()
    if group is None:
        raise RuntimeError(
            "preparation_barrier() must be called inside the Trainer data-loader hook"
        )
    dist.barrier(group=group)


@contextmanager
def distributed_preparation(timeout_seconds):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        yield
        return
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))
    token = _group.set(group)
    try:
        yield
        # A fast rank must not enter training collectives while another rank
        # is still creating its loader or waiting for an external cache builder.
        preparation_barrier()
    finally:
        _group.reset(token)
        dist.destroy_process_group(group)
