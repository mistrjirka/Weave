"""Optional PyTorch orchestration for user-owned recipes. No web dependencies.

The script owns model/data/loss/optimizer/scheduler. This helper handles devices,
AMP, accumulation, durable reporting, cooperative checkpoints and provenance.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Callable, Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from text_training import (
    place_model,
    capture_rng,
    restore_rng,
    evaluate as text_evaluate,
)
from runtime import SparseMoE
from weave_training import (
    Run,
    atomic_json,
    atomic_checkpoint_path,
    file_hash,
    available_cpu_threads,
)
from token_cache import cache_progress_reporting
from tuning_runtime import load_tuning, tune_loader, WeightedBatchSampler
from preparation import distributed_preparation
from performance import configure_execution, training_profile
from evaluation_worker import (
    EvaluationSession,
    memory_snapshot_directory,
    next_request,
    read_request,
)


@dataclass
class TrainingConfig:
    devices: list[int] = field(default_factory=list)
    strategy: Literal["auto", "single", "ddp", "split", "train_eval"] = "auto"
    placement: dict[str, int] = field(default_factory=dict)
    use_gpu_tuning: bool = True
    rank_micro_batch_sizes: list[int] = field(default_factory=list, init=False)
    evaluation_device: int | Literal["cpu"] | None = None
    async_evaluation: bool = False
    reuse_evaluation_worker: bool = False
    evaluation_snapshot_storage: Literal["auto", "disk"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    compile_mode: Literal["off", "default", "reduce-overhead", "max-autotune"] = "off"
    fused_optimizer: bool = False
    profile_steps: int = 0
    seed: int = 42
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_epochs: int | None = 3
    max_optimizer_steps: int | None = None
    total_optimizer_steps: int | None = field(default=None, init=False)
    evaluation_only: bool = field(default=False, init=False)
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    log_every: int = 1
    checkpoint_every: int = 50
    eval_every: int = 50
    eval_batches: int = 5
    generation_tokens: int = 32
    generation_context_length: int | None = None
    sample_prompts: list[str | dict] = field(default_factory=list)
    cpu_threads: int = field(default_factory=available_cpu_threads)
    preparation_timeout_seconds: int = 86400
    find_unused_parameters: bool | None = None
    data_fingerprint: Literal["stat", "sha256"] = "stat"
    trace_batches: Literal["metadata", "sha256", "none"] = "metadata"

    def __post_init__(self):
        for name in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "log_every",
            "checkpoint_every",
            "eval_every",
            "eval_batches",
            "generation_tokens",
            "cpu_threads",
            "preparation_timeout_seconds",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_epochs", "max_optimizer_steps", "generation_context_length"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_epochs is None and self.max_optimizer_steps is None:
            raise ValueError("Set max_epochs or max_optimizer_steps to bound training")
        if self.strategy not in ("auto", "single", "ddp", "split", "train_eval"):
            raise ValueError("Unknown device strategy")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.compile_mode not in (
            "off",
            "default",
            "reduce-overhead",
            "max-autotune",
        ):
            raise ValueError("Unknown compile_mode")
        if type(self.fused_optimizer) is not bool:
            raise ValueError("fused_optimizer must be a boolean")
        if type(self.reuse_evaluation_worker) is not bool:
            raise ValueError("reuse_evaluation_worker must be a boolean")
        if self.evaluation_snapshot_storage not in ("auto", "disk"):
            raise ValueError("evaluation_snapshot_storage must be auto or disk")
        if type(self.profile_steps) is not int or not 0 <= self.profile_steps <= 100:
            raise ValueError("profile_steps must be an integer from 0 to 100")
        if self.evaluation_device not in (None, "cpu") and (
            type(self.evaluation_device) is not int or self.evaluation_device < 0
        ):
            raise ValueError("evaluation_device must be 'cpu', a GPU index or None")
        if len(set(self.devices)) != len(self.devices) or any(
            type(i) is not int or i < 0 for i in self.devices
        ):
            raise ValueError("devices must contain unique nonnegative GPU indices")
        if self.strategy == "single" and len(self.devices) > 1:
            raise ValueError("single strategy accepts at most one GPU")
        if self.strategy == "split" and (
            len(self.devices) < 2 or self.precision != "fp32"
        ):
            raise ValueError("Split placement requires at least two GPUs and FP32")
        if self.strategy == "train_eval" and (
            len(self.devices) != 1
            or self.evaluation_device is None
            or self.evaluation_device in self.devices
        ):
            raise ValueError(
                "train_eval requires one training GPU and a different evaluation_device"
            )
        if (
            self.learning_rate <= 0
            or self.gradient_clip <= 0
            or self.weight_decay < 0
            or not 0 <= self.min_learning_rate <= self.learning_rate
        ):
            raise ValueError("Invalid optimization settings")
        if self.data_fingerprint not in (
            "stat",
            "sha256",
        ) or self.trace_batches not in ("metadata", "sha256", "none"):
            raise ValueError("Invalid provenance mode")

    @property
    def local_micro_batch_size(self):
        rank = int(os.environ.get("RANK", "0"))
        return (
            self.rank_micro_batch_sizes[rank]
            if self.rank_micro_batch_sizes
            else self.micro_batch_size
        )

    @property
    def resolved_strategy(self):
        return (
            ("ddp" if len(self.devices) > 1 else "single")
            if self.strategy == "auto"
            else self.strategy
        )

    @property
    def background_evaluation(self):
        return self.async_evaluation or self.resolved_strategy == "train_eval"


@dataclass
class StepResult:
    """A mean loss and its averaging weight (e.g. number of supervised tokens).

    Returning (loss, metrics) is also supported and averages microbatches equally.
    Metrics use the same weight; tensors are detached only for reporting.
    """

    loss: Tensor
    metrics: Mapping[str, float | Tensor] = field(default_factory=dict)
    weight: float = 1.0


def language_model_step(model, batch, device, *, chunk_size=None):
    """Next-token objective; optional chunked projection for generated LM heads.

    The loader owns shifting, sequence lengths and masking. Count labels on the
    CPU before transfer, avoiding a CUDA scalar read in the usual loader path.
    """
    labels = batch["labels"]
    count = int((labels != -100).sum())
    if not count:
        raise ValueError("Batch has no supervised targets.")
    inputs = batch["input_ids"].to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if chunk_size is not None:
        raw = model.module if isinstance(model, DDP) else model
        if not getattr(raw, "supports_chunked_loss", False):
            raise ValueError(
                "Chunked loss requires a newly exported graph with one terminal Vocabulary head"
            )
        loss = model(inputs, loss_targets=labels, loss_chunk_size=chunk_size)
    else:
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1),
            labels.to(logits.device).flatten(),
            ignore_index=-100,
        )
    return StepResult(loss, weight=count)


# Bind extracted methods to this module's globals so existing patches of trainer
# dependencies continue to affect every lifecycle path.
from types import FunctionType
from trainer_lifecycle import TrainerLifecycle
from trainer_state import TrainerState
from trainer_loop import TrainerLoop
from trainer_evaluation_jobs import TrainerEvaluationJobs
from trainer_evaluation import TrainerEvaluation


def _bind_methods(mixin):
    for name, value in vars(mixin).copy().items():
        static = isinstance(value, staticmethod)
        fn = value.__func__ if static else value
        if isinstance(fn, FunctionType):
            bound = FunctionType(fn.__code__, globals(), fn.__name__, fn.__defaults__, fn.__closure__)
            bound.__kwdefaults__ = fn.__kwdefaults__
            bound.__annotations__ = fn.__annotations__
            bound.__doc__ = fn.__doc__
            bound.__qualname__ = fn.__qualname__
            setattr(mixin, name, staticmethod(bound) if static else bound)
    return mixin


class Trainer(TrainerLifecycle, TrainerState, TrainerLoop, TrainerEvaluationJobs, TrainerEvaluation):
    pass


for _mixin in Trainer.__bases__:
    _bind_methods(_mixin)
