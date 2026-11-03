"""Apply a verified benchmark without changing user scripts or dataset semantics."""

import json
import math
from pathlib import Path


def hardware_identity(devices):
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this interpreter")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "devices": [
            {
                "index": index,
                "uuid": str(
                    getattr(torch.cuda.get_device_properties(index), "uuid", "")
                ),
                "name": torch.cuda.get_device_name(index),
                "memory": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in devices
        ],
    }


def load_tuning(config, launch, support):
    profile = launch.get("gpu_tuning")
    if not profile or not config.use_gpu_tuning:
        return None, ""
    from .tuning_math import graph_signature

    try:
        if profile["model_signature"] != graph_signature(
            json.loads((Path(support) / "project.json").read_text())
        ):
            return None, "Saved GPU tuning skipped: graph changed."
        if profile["hardware"] != hardware_identity(profile["devices"]):
            return (
                None,
                "Saved GPU tuning skipped: GPU hardware or PyTorch/CUDA version changed.",
            )
        if (
            sorted(config.devices) != profile["devices"]
            or config.precision != profile["precision"]
        ):
            return (
                None,
                "Saved GPU tuning skipped: device selection or precision differs.",
            )
        if config.resolved_strategy != profile["strategy"]:
            return None, "Saved GPU tuning skipped: strategy differs."
        if config.resolved_strategy == "ddp":
            if (
                config.micro_batch_size * len(config.devices)
                != profile["global_batch_size"]
            ):
                return None, "Saved GPU tuning skipped: global microbatch size changed."
            sizes = [profile["batch_sizes"][str(index)] for index in config.devices]
            if (
                any(type(size) is not int or size < 1 for size in sizes)
                or sum(sizes) != profile["global_batch_size"]
            ):
                raise ValueError("Invalid rank batch sizes")
            config.rank_micro_batch_sizes = sizes
        elif config.resolved_strategy == "split":
            if config.placement:
                return (
                    None,
                    "Saved GPU tuning skipped: explicit block placement takes precedence.",
                )
            config.placement = profile["placement"]
        return profile, ""
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        return None, f"Saved GPU tuning skipped: {exc}"


def tune_loader(loader, config, rank):
    """Only replace the standard DistributedSampler/DataLoader combination."""
    if not config.rank_micro_batch_sizes:
        return loader, False
    if getattr(loader, "weave_tuning_applied", False):
        return loader, True
    from torch.utils.data import DataLoader, BatchSampler, IterableDataset
    from torch.utils.data.distributed import DistributedSampler

    sampler = getattr(loader, "sampler", None)
    if (
        type(loader) is not DataLoader
        or isinstance(loader.dataset, IterableDataset)
        or not isinstance(sampler, DistributedSampler)
        or type(loader.batch_sampler) is not BatchSampler
        or loader.batch_size != config.micro_batch_size
    ):
        return loader, False
    batches = WeightedBatchSampler(
        len(loader.dataset),
        config.rank_micro_batch_sizes,
        rank,
        sampler.seed,
        sampler.shuffle,
        loader.drop_last,
        pad_to_world=not sampler.drop_last,
    )
    kwargs = dict(
        dataset=loader.dataset,
        batch_sampler=batches,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
        persistent_workers=loader.persistent_workers,
    )
    if loader.num_workers:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["multiprocessing_context"] = loader.multiprocessing_context
    rebuilt = DataLoader(**kwargs)
    rebuilt.weave_tuning_applied = True
    return rebuilt, True


class WeightedBatchSampler:
    """One shared permutation; disjoint unequal slices, equal steps on every rank.

    Match DistributedSampler's epoch padding/truncation (at most world-1
    records). A partial final update gets smaller positive rank allocations.
    DataLoader drop_last=True omits the incomplete global batch.
    """

    def __init__(
        self,
        length,
        sizes,
        rank,
        seed=0,
        shuffle=True,
        drop_last=False,
        pad_to_world=True,
    ):
        if (
            not sizes
            or any(type(size) is not int or size < 1 for size in sizes)
            or not 0 <= rank < len(sizes)
        ):
            raise ValueError("Invalid DDP allocation")
        if sum(sizes) % len(sizes):
            raise ValueError(
                "Global batch must be divisible by rank count for standard-loader tuning"
            )
        self.length, self.sizes, self.rank = length, list(sizes), rank
        self.available = (
            math.ceil(length / len(sizes)) if pad_to_world else length // len(sizes)
        ) * len(sizes)
        self.seed, self.shuffle, self.drop_last, self.epoch = (
            seed,
            shuffle,
            drop_last,
            0,
        )

    def __len__(self):
        total = sum(self.sizes)
        return (
            self.available // total
            if self.drop_last
            else math.ceil(self.available / total)
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import torch

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = (
            torch.randperm(self.length, generator=generator).tolist()
            if self.shuffle
            else list(range(self.length))
        )
        total = sum(self.sizes)
        count = len(self) * total if self.drop_last else self.available
        if count > self.length and indices:
            indices = (indices * math.ceil(count / self.length))[:count]
        from .tuning_math import allocate

        for offset in range(0, count, total):
            sizes = (
                self.sizes
                if count - offset >= total
                else allocate(count - offset, self.sizes)
            )
            start = sum(sizes[: self.rank])
            yield indices[offset + start : offset + start + sizes[self.rank]]
