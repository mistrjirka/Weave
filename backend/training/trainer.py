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
from preparation import distributed_preparation, preparation_barrier
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


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module | Callable[[], nn.Module],
        config: TrainingConfig,
        dataloaders_fn: Callable,
        training_step_fn: Callable,
        tokenizer=None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        evaluation_step_fn: Callable | None = None,
        samples_fn: Callable | None = None,
    ):
        self.model_factory, self.config = model, config
        self.tokenizer_factory = tokenizer
        self.dataloaders_fn, self.training_step_fn = dataloaders_fn, training_step_fn
        self.optimizer_fn, self.scheduler_fn = optimizer_fn, scheduler_fn
        self.evaluation_step_fn = evaluation_step_fn or training_step_fn
        self.samples_fn = samples_fn
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.source_directory = Path(sys.argv[0]).resolve().parent
        self.epoch = 0
        self.batches_in_epoch = 0
        self.batches_processed = 0
        self.batches_per_epoch = None
        self.resuming = False
        self.evaluation_job = None
        self.evaluation_pending = None
        self.evaluation_session = None
        self.evaluation_reuse_unavailable = False
        self.evaluation_memory = None
        self.evaluation_memory_disabled = False
        self.profiler = None

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--evaluate")
        parser.add_argument("--evaluation-worker")
        parser.add_argument("--checkpoint")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--output-dir")
        args = parser.parse_args()
        if args.evaluation_worker:
            request = read_request(args.evaluation_worker)
            if request is None:
                raise ValueError("Validation worker has no initial request")
            args.evaluate = request["checkpoint"]
            self.evaluation_request = request
        cfg = self.config
        context_path = Path(__file__).parent / "launch.json"
        launch = json.loads(context_path.read_text()) if context_path.is_file() else {}
        if "use_gpu_tuning" in launch:
            cfg.use_gpu_tuning = launch["use_gpu_tuning"]
        if launch.get("compute_mode") == "override":
            cfg.strategy = launch["strategy"]
            cfg.devices = list(launch["devices"])
            cfg.precision = launch["precision"]
            cfg.placement = dict(launch.get("placement", {}))
        mode = launch.get("evaluation_mode", "script")
        if launch.get("performance_mode") == "override":
            for key in ("compile_mode", "fused_optimizer", "profile_steps"):
                setattr(cfg, key, launch[key])
        if mode != "script":
            cfg.async_evaluation = mode == "async"
            cfg.evaluation_device = launch.get("evaluation_device")
            if mode == "inline" and cfg.strategy == "train_eval":
                cfg.strategy = "single"
        validation_initial = Path(__file__).parent / "validation-initial.json"
        if validation_initial.is_file():
            self._set_validation(json.loads(validation_initial.read_text()))
        # An evaluator receives a snapshot of the settings chosen by its parent,
        # rather than rereading the original launch overrides.
        if args.evaluate and os.environ.get("WEAVE_VALIDATION_SETTINGS"):
            self._set_validation(json.loads(os.environ["WEAVE_VALIDATION_SETTINGS"]))
        if args.evaluation_worker:
            self._set_validation(self.evaluation_request["settings"])
        # Validate the effective settings before spawning workers or calling user
        # data hooks. launch.json controls the frozen run, never the source script.
        cfg.__post_init__()
        self.tuning_profile, self.tuning_note = (None, "")
        if not args.evaluate:
            self.tuning_profile, self.tuning_note = load_tuning(
                cfg, launch, Path(__file__).parent
            )
        # Evaluation is a standalone process, even when launched from torchrun.
        if args.evaluate:
            cfg.evaluation_only = True
            if cfg.background_evaluation and cfg.evaluation_device is None:
                cfg.evaluation_device = "cpu"
            self.rank, self.world = 0, 1
        output = args.output_dir or os.environ.get("WEAVE_RUN_DIR")
        if not output:
            output = str(self.source_directory / "runs" / f"run-{time.time_ns()}")
        if self.world == 1 and cfg.resolved_strategy == "ddp" and not args.evaluate:
            count = len(cfg.devices) or 2
            environment = dict(
                os.environ,
                WEAVE_RUN_DIR=output,
                USE_TORCH="1",
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={count}",
                    *sys.argv,
                ],
                env=environment,
                check=True,
            )
            return
        initialized = False
        try:
            if self.world > 1:
                if cfg.resolved_strategy != "ddp":
                    raise ValueError(
                        "torchrun requires strategy='ddp' (or auto with multiple GPUs)"
                    )
                if cfg.devices:
                    if self.world != len(cfg.devices):
                        raise ValueError("torchrun process count must match devices")
                    torch.cuda.set_device(cfg.devices[self.rank])
                if not dist.is_initialized():
                    dist.init_process_group("nccl" if cfg.devices else "gloo")
                    initialized = True
                # Also synchronize output paths when users invoke torchrun directly.
                values = [output if self.rank == 0 else None]
                dist.broadcast_object_list(values, src=0)
                output = values[0]
            directory = Path(output).expanduser().resolve()
            self._run(args, directory)
        finally:
            self._close_evaluation_session()
            if self.evaluation_job and self.evaluation_job.poll() is None:
                self.evaluation_job.terminate()
                try:
                    self.evaluation_job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.evaluation_job.kill()
                    self.evaluation_job.wait()
            if self.evaluation_memory:
                self.evaluation_memory.cleanup()
            if initialized:
                dist.destroy_process_group()

    def _run(self, args, directory):
        cfg = self.config
        torch.set_num_threads(cfg.cpu_threads)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        placement = dict(
            strategy=cfg.resolved_strategy, devices=cfg.devices, placement=cfg.placement
        )
        if self.world > 1:
            placement.update(
                strategy="single",
                devices=[cfg.devices[self.rank]] if cfg.devices else [],
            )
        if args.evaluate:
            placement.update(
                strategy="single",
                devices=(
                    []
                    if cfg.evaluation_device == "cpu"
                    else (
                        [cfg.evaluation_device]
                        if cfg.evaluation_device is not None
                        else cfg.devices[:1]
                    )
                ),
            )
            # CPU evaluation uses full precision even for an FP16 training job.
            if cfg.evaluation_device == "cpu":
                cfg.precision = "fp32"
            # Data hooks must also see a standalone evaluator, rather than
            # constructing a DistributedSampler for the training process group.
            cfg.strategy = "single"
            cfg.devices = placement["devices"]
            cfg.placement = {}
        self.model = (
            self.model_factory()
            if not isinstance(self.model_factory, nn.Module)
            else self.model_factory
        )
        self.model, self.device = place_model(self.model, placement)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        if cfg.precision == "fp16" and self.device == "cpu":
            raise ValueError("FP16 requires CUDA; use fp32 or bf16 on CPU")
        if cfg.precision == "bf16" and self.device != "cpu":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError(
                        "Selected GPU does not support native BF16; choose fp16 or fp32"
                    )
        preparation_directory = (
            directory / "preparation"
            if not args.evaluate
            else directory / ("evaluation-" + Path(args.evaluate).stem) / "preparation"
        )
        with Run(
            preparation_directory,
            rank=self.rank,
            source_directory=self.source_directory,
            channel="evaluation" if args.evaluate else "system",
        ) as preparation:
            preparation.emit(
                "preparation", {"phase": "initializing", "documents": 0, "tokens": 0}
            )

            def report_cache(progress):
                preparation.emit("preparation", progress)

            try:
                with (
                    distributed_preparation(cfg.preparation_timeout_seconds),
                    cache_progress_reporting(report_cache),
                ):
                    self.tokenizer = (
                        self.tokenizer_factory()
                        if callable(self.tokenizer_factory)
                        else self.tokenizer_factory
                    )
                    self.train_loader, self.val_loader = self.dataloaders_fn(
                        self.tokenizer, cfg
                    )
            except Exception as exc:
                preparation.emit(
                    "preparation",
                    {
                        "phase": "failed",
                        "message": str(exc),
                        "documents": 0,
                        "tokens": 0,
                    },
                )
                raise
        self.tuning_applied = bool(
            self.tuning_profile and cfg.resolved_strategy == "split"
        )
        if self.tuning_profile and cfg.resolved_strategy == "ddp" and not args.evaluate:
            self.train_loader, self.tuning_applied = tune_loader(
                self.train_loader, cfg, self.rank
            )
            if not self.tuning_applied:
                self.tuning_note = "GPU allocation not applied: custom loader owns batching. Use config.rank_micro_batch_sizes with matched steps and declare loader.weave_tuning_applied=True."
        try:
            self.batches_per_epoch = len(self.train_loader)
        except (TypeError, NotImplementedError):
            pass
        if self.batches_per_epoch == 0 and not args.evaluate:
            raise ValueError("Training loader contains no batches")
        if self.world > 1:
            lengths = [None] * self.world
            dist.all_gather_object(lengths, self.batches_per_epoch)
            if len(set(lengths)) != 1:
                raise ValueError(
                    "DDP loaders must have equal batch counts on every rank"
                )
        planned = []
        if cfg.max_optimizer_steps is not None:
            planned.append(cfg.max_optimizer_steps)
        if cfg.max_epochs is not None and self.batches_per_epoch is not None:
            planned.append(
                math.ceil(self.batches_per_epoch / cfg.gradient_accumulation_steps)
                * cfg.max_epochs
            )
        cfg.total_optimizer_steps = min(planned) if planned else None
        self.optimizer = (
            None
            if args.evaluate
            else (
                self.optimizer_fn(self.model, cfg)
                if self.optimizer_fn
                else torch.optim.AdamW(
                    self.model.parameters(),
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
            )
        )
        self.scheduler = (
            self.scheduler_fn(self.optimizer, cfg)
            if self.scheduler_fn and not args.evaluate
            else None
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
        context_path = Path(__file__).parent / "launch.json"
        launch = json.loads(context_path.read_text()) if context_path.is_file() else {}
        if args.evaluate:
            self._evaluation_jobs(args, directory)
            return
        checkpoint = args.checkpoint or launch.get("checkpoint")
        resume = args.resume or launch.get("start_mode") == "resume"
        step, tokens = 0, 0
        directory.mkdir(parents=True, exist_ok=True)
        report_directory = directory
        with Run(
            report_directory,
            rank=self.rank,
            source_directory=self.source_directory,
            channel="training",
        ) as run:
            self.run = run
            if cfg.background_evaluation and cfg.evaluation_device in cfg.devices:
                run.warning(
                    "Background validation shares a training GPU; it needs another model copy and competes for memory and compute. Select CPU or a separate GPU to avoid GPU contention.",
                    code="evaluation-contention",
                )
            run.capabilities(
                metrics=True,
                samples=True,
                checkpoint=True,
                commands=True,
                live_validation=True,
            )
            if self.rank == 0:
                print(f"Run outputs: {directory}", flush=True)
                print(
                    f"Compute: {cfg.resolved_strategy} · devices={cfg.devices or 'CPU'} "
                    f"· workers={self.world} · precision={cfg.precision}",
                    flush=True,
                )
                atomic_json(report_directory / "resolved-config.json", asdict(cfg))
                run.emit("configuration", asdict(cfg))
            self.manifest = self._sources(report_directory, run)
            self.code_sha256 = self._code_hash()
            if checkpoint:
                path = Path(checkpoint).expanduser()
                if not path.is_absolute():
                    path = self.source_directory / path
                state = torch.load(path, map_location="cpu", weights_only=True)
                try:
                    self.model.load_state_dict(state["model"], strict=True)
                    if resume:
                        self._restore(state)
                        self.resuming = True
                        step, tokens = state["step"], state["tokens"]
                except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                    if resume and self.rank == 0:
                        problem = {
                            "message": str(exc),
                            "checkpoint": str(path),
                            "recovery": "Fix the stage script and retry, or resume with the previous script. The source checkpoint and scripts are unchanged.",
                        }
                        atomic_json(report_directory / "resume-problem.json", problem)
                        run.emit("resume_problem", problem)
                    raise
            run.step, run.tokens = step, tokens
            if self.tuning_profile or self.tuning_note:
                run.emit(
                    "gpu_tuning",
                    {
                        "applied": self.tuning_applied,
                        "profile": self.tuning_profile,
                        "message": self.tuning_note,
                    },
                )
                if self.tuning_note:
                    run.warning(self.tuning_note, code="gpu-tuning")
            configure_execution(self.model, self.optimizer, cfg)
            run.emit(
                "execution",
                {
                    "compile_mode": cfg.compile_mode,
                    "fused_optimizer": bool(
                        self.optimizer.param_groups[0].get("fused")
                    ),
                    "profile_steps": cfg.profile_steps,
                },
            )
            if self.world > 1:
                dataset = getattr(self.train_loader, "dataset", None)
                sampler = getattr(self.train_loader, "sampler", None)
                from torch.utils.data.distributed import DistributedSampler

                if (
                    not isinstance(sampler, DistributedSampler)
                    and not isinstance(
                        getattr(self.train_loader, "batch_sampler", None),
                        WeightedBatchSampler,
                    )
                    and not getattr(dataset, "distributed_sharding", False)
                    and not getattr(self.train_loader, "weave_tuning_applied", False)
                ):
                    raise ValueError(
                        "DDP loader must use DistributedSampler or a sharded IterableDataset with distributed_sharding=True"
                    )
                self.model = DDP(
                    self.model,
                    device_ids=[cfg.devices[self.rank]] if cfg.devices else None,
                    find_unused_parameters=(
                        cfg.find_unused_parameters
                        if cfg.find_unused_parameters is not None
                        else any(
                            isinstance(module, SparseMoE)
                            for module in self.model.modules()
                        )
                    ),
                )
            if not self.resuming:
                random.seed(cfg.seed + self.rank)
                torch.manual_seed(cfg.seed + self.rank)
            with training_profile(directory, cfg, self.rank, run) as self.profiler:
                stopped = self._loop(directory)
            if self.rank == 0 and (cfg.background_evaluation or self.evaluation_job):
                if stopped:
                    self._cancel_evaluation(directory)
                else:
                    while self.evaluation_job or self.evaluation_pending:
                        self._evaluation_worker(directory, wait=True)
            if self.rank == 0:
                for info in self.manifest.values():
                    path = Path(info["path"])
                    if (
                        not path.is_file()
                        or path.stat().st_size != info["bytes"]
                        or path.stat().st_mtime_ns != info["mtime_ns"]
                    ):
                        run.warning(
                            f"Dataset changed during training: {path}",
                            code="dataset-changed",
                        )

    def _evaluation_jobs(self, args, directory):
        """Initialize once, but load and report every immutable checkpoint afresh."""
        mailbox = args.evaluation_worker
        request = self.evaluation_request if mailbox else None
        reusable = not any(
            isinstance(value, Iterator)
            for value in (self.val_loader, getattr(self.val_loader, "dataset", None))
        ) and not getattr(self.val_loader, "persistent_workers", False)
        parent, index = os.getppid(), 0
        while True:
            checkpoint = request["checkpoint"] if request else args.evaluate
            if request:
                self._set_validation(request["settings"])
            report_directory = directory / ("evaluation-" + Path(checkpoint).stem)
            try:
                with Run(
                    report_directory,
                    rank=0,
                    source_directory=self.source_directory,
                    channel="evaluation",
                ) as run:
                    self.run = run
                    run.capabilities(
                        metrics=True, samples=True, checkpoint=False, commands=False
                    )
                    atomic_json(
                        report_directory / "resolved-config.json", asdict(self.config)
                    )
                    run.emit("configuration", asdict(self.config))
                    self.manifest = self._sources(report_directory, run)
                    path = Path(checkpoint).expanduser()
                    if not path.is_absolute():
                        path = self.source_directory / path
                    started = time.monotonic()
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    self.model.load_state_dict(state["model"], strict=True)
                    run.step, run.tokens = state["step"], state["tokens"]
                    del state
                    loaded = time.monotonic()
                    # _evaluate opens a fresh AMP context: casts of old weights
                    # must never be cached across checkpoint replacements.
                    self._evaluate(str(checkpoint))
                    timing = {
                        "pid": os.getpid(),
                        "reused": index > 0,
                        "checkpoint_load_seconds": loaded - started,
                        "evaluation_seconds": time.monotonic() - loaded,
                    }
                    run.emit("evaluation_runtime", timing)
                    print(
                        f"Validation update {run.step}: "
                        f"{'warm' if index else 'fresh'} worker {os.getpid()} · "
                        f"weights {timing['checkpoint_load_seconds']:.2f}s · "
                        f"evaluation {timing['evaluation_seconds']:.2f}s",
                        flush=True,
                    )
                if request:
                    atomic_json(
                        Path(mailbox) / "result.json",
                        {
                            "id": request["id"],
                            "success": True,
                            "reusable": reusable,
                            "step": run.step,
                            **timing,
                        },
                    )
            except Exception as exc:
                if request:
                    atomic_json(
                        Path(mailbox) / "result.json",
                        {
                            "id": request["id"],
                            "success": False,
                            "reusable": False,
                            "message": str(exc),
                        },
                    )
                raise
            if not mailbox or not reusable:
                return
            request = next_request(mailbox, request["id"], parent)
            if request is None:
                return
            index += 1

    def _close_evaluation_session(self):
        session = getattr(self, "evaluation_session", None)
        if session:
            session.close()
            if self.evaluation_job is session:
                self.evaluation_job = None
            self.evaluation_session = None

    def _code_hash(self):
        # Include user helper packages even when imported lazily inside a hook.
        paths = set()
        excluded = {
            "runs",
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            "checkpoints",
            "snapshots",
            "packages",
            "lineage",
            "source-data",
            "starting-weights",
            "tokenizers",
            "licenses",
            "migrations",
            "resume-attempts",
        }
        for folder, children, files in os.walk(self.source_directory):
            children[:] = [
                name
                for name in children
                if name not in excluded and not Path(folder, name).is_symlink()
            ]
            paths.update(
                Path(folder, name)
                for name in files
                if name.endswith(".py") and not Path(folder, name).is_symlink()
            )
        digest = hashlib.sha256()
        for path in sorted(paths):
            digest.update(str(path.relative_to(self.source_directory)).encode())
            digest.update(file_hash(path).encode())
        return digest.hexdigest()

    def _sources(self, directory, run):
        manifest = {}
        for split, loader in (
            ("train", self.train_loader),
            ("validation", self.val_loader),
        ):
            if loader is None:
                continue
            dataset = getattr(loader, "dataset", loader)
            paths = getattr(dataset, "source_files", getattr(dataset, "files", []))
            for name in paths:
                path = Path(name).expanduser().resolve()
                stat = path.stat()
                info = dict(
                    path=str(path),
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    split=split,
                    fingerprint=self.config.data_fingerprint,
                    declared=True,
                )
                manifest[(split, str(path))] = info
        if self.world > 1:
            all_sources = [None] * self.world
            dist.all_gather_object(all_sources, manifest)
            manifest = {
                key: value for sources in all_sources for key, value in sources.items()
            }
        counters = {}
        ordered = {}
        for (split, _), info in sorted(manifest.items()):
            index = counters.get(split, 0)
            ordered[f"{split}-{index}"] = info
            counters[split] = index + 1
        manifest = ordered
        if self.rank == 0:
            if self.config.data_fingerprint == "sha256":
                for info in manifest.values():
                    info["sha256"] = file_hash(info["path"])
            atomic_json(directory / "datasets.json", manifest)
            run.emit("provenance", {"files": {}, "datasets": manifest})
            if not manifest:
                run.warning(
                    "Custom loader declares no source files. Expose dataset.source_files (or files) to record file paths; batch _weave_source metadata records consumed records.",
                    code="dataset-provenance",
                )
        if self.world > 1:
            values = [manifest if self.rank == 0 else None]
            dist.broadcast_object_list(values, src=0)
            manifest = values[0]
        return manifest

    def _loader_state(self):
        loader = self.train_loader
        if callable(getattr(loader, "state_dict", None)) and callable(
            getattr(loader, "load_state_dict", None)
        ):
            return {"kind": "loader", "state": loader.state_dict(), "epoch": self.epoch}
        dataset = getattr(loader, "dataset", None)
        if (
            not getattr(loader, "num_workers", 0)
            and callable(getattr(dataset, "state_dict", None))
            and callable(getattr(dataset, "load_state_dict", None))
        ):
            return {
                "kind": "dataset",
                "state": dataset.state_dict(),
                "epoch": self.epoch,
            }
        return None

    def _optimizer_structure(self):
        """Optimizer IDs are positional; tensor shapes alone cannot identify them."""
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        names = {id(param): name for name, param in raw.named_parameters()}
        return {
            "type": type(self.optimizer).__module__
            + "."
            + type(self.optimizer).__qualname__,
            "scheduler": (
                type(self.scheduler).__module__
                + "."
                + type(self.scheduler).__qualname__
                if self.scheduler is not None
                else None
            ),
            "groups": [
                [
                    {"name": names.get(id(param)), "shape": list(param.shape)}
                    for param in group["params"]
                ]
                for group in self.optimizer.param_groups
            ],
        }

    def _restore(self, state):
        if state.get("trainer_version") != 2:
            raise ValueError(
                "This checkpoint uses a different trainer. Load weights into a new stage instead."
            )
        migration_path = Path(__file__).parent / "resume-compatibility.json"
        migration = (
            json.loads(migration_path.read_text()) if migration_path.is_file() else {}
        )
        compatible_upgrade = (
            migration.get("previous_code_sha256") == state.get("code_sha256")
            and migration.get("current_code_sha256") == self.code_sha256
        )
        if state.get("code_sha256") != self.code_sha256 and not compatible_upgrade:
            raise ValueError(
                "Training/model code changed since checkpoint; start a new stage from weights"
            )
        if (
            compatible_upgrade
            and migration.get("edited_script")
            and not state.get("optimizer_structure")
        ):
            raise ValueError(
                "This older checkpoint has no optimizer parameter-name mapping. Resume once with the previous script and save a new checkpoint before using an edited script."
            )
        if (
            state.get("optimizer_structure") is not None
            and state["optimizer_structure"] != self._optimizer_structure()
        ):
            raise ValueError(
                "Optimizer parameter names, shapes, group order, optimizer type or scheduler type differ from the checkpoint. Fix the script or retry with the previous script."
            )
        previous = dict(state["config"])
        current = asdict(self.config)
        if previous.get("strategy") == "train_eval" and current["strategy"] == "single":
            # Both place the training model on one device. Only the evaluation
            # policy moved from the legacy alias into explicit live settings.
            previous["strategy"] = "single"
        mutable = {
            "max_epochs",
            "max_optimizer_steps",
            "total_optimizer_steps",
            "log_every",
            "checkpoint_every",
            "eval_every",
            "eval_batches",
            "generation_tokens",
            "sample_prompts",
            "async_evaluation",
            "reuse_evaluation_worker",
            "evaluation_snapshot_storage",
            "evaluation_device",
            "evaluation_only",
            "gradient_clip",
            "cpu_threads",
            "generation_context_length",
            "trace_batches",
            "compile_mode",
            "fused_optimizer",
            "profile_steps",
        }
        changed = [
            key
            for key, value in current.items()
            if key not in mutable and previous.get(key) != value
        ]
        if changed:
            raise ValueError(
                "Training configuration differs from checkpoint: "
                + ", ".join(changed)
                + ". This resume preserves optimizer/scheduler state; restore these settings or start a new stage from weights."
            )
        if state["datasets"] != self.manifest:
            raise ValueError(
                "Dataset paths or fingerprints differ from checkpoint; start from weights instead"
            )
        ranks = state.get("rank_state", [])
        if len(ranks) != self.world or not ranks[self.rank].get("loader"):
            raise ValueError(
                "Exact resume requires a stateful loader/dataset and the same process count. Use a new stage from weights, or implement state_dict/load_state_dict (with num_workers=0 for dataset state)."
            )
        rank_state = ranks[self.rank]
        saved = rank_state["loader"]
        target = (
            self.train_loader
            if saved["kind"] == "loader"
            else self.train_loader.dataset
        )
        if not callable(getattr(target, "load_state_dict", None)):
            raise ValueError(
                "Loader cannot restore the saved data cursor; start from weights instead"
            )
        target.load_state_dict(saved["state"])
        self.epoch = saved["epoch"]
        self.batches_in_epoch = rank_state["batches_in_epoch"]
        self.batches_processed = rank_state["batches_processed"]
        self.optimizer.load_state_dict(state["optimizer"])
        if (self.scheduler is None) != (state["scheduler"] is None):
            raise ValueError(
                "Scheduler presence differs from checkpoint; fix the script or retry with the previous script"
            )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        restore_rng(rank_state["rng"])

    def _amp(self):
        return torch.autocast(
            "cuda" if self.device != "cpu" else "cpu",
            dtype=torch.float16 if self.config.precision == "fp16" else torch.bfloat16,
            enabled=self.config.precision != "fp32",
        )

    @staticmethod
    def _result(value, *, check_finite=True):
        result = (
            value
            if isinstance(value, StepResult)
            else StepResult(*value) if isinstance(value, tuple) else StepResult(value)
        )
        result.weight = float(result.weight)
        if (
            result.loss.ndim != 0
            or not math.isfinite(result.weight)
            or result.weight <= 0
        ):
            raise ValueError(
                "Step hook must return a scalar mean loss with a positive finite weight"
            )
        if check_finite and not torch.isfinite(result.loss):
            raise ValueError("Nonfinite loss; optimizer update was not applied")
        return result

    def _loop(self, directory):
        cfg, run = self.config, self.run
        self.model.train()
        rng = capture_rng()
        iterator = iter(self.train_loader)
        if self.resuming:
            restore_rng(
                rng
            )  # Recreating a DataLoader iterator must not consume the saved model RNG.
        started, initial_tokens = time.monotonic(), run.tokens
        if self.rank == 0 and self._loader_state() is None:
            run.warning(
                "This loader has no restorable cursor. Checkpoints support new stages from weights; exact resume requires loader/dataset state_dict/load_state_dict.",
                code="loader-resume",
            )
        overflow_count = 0
        stopped = False
        self._last_checkpoint_boundary = None
        self._last_checkpoint_path = None
        last_evaluated_step = None
        missing_trace_warned = False
        with (directory / f"consumed-rank-{self.rank}.jsonl").open(
            "a", buffering=1
        ) as trace:
            while not self._training_done():
                commands = [run.poll_commands() if self.rank == 0 else []]
                if self.world > 1:
                    dist.broadcast_object_list(commands, src=0)
                commands = commands[0]
                for command in commands:
                    if command["kind"] == "validation":
                        self._set_validation(command["settings"])
                        if self.rank == 0:
                            run.emit(
                                "validation_settings",
                                {
                                    **command["settings"],
                                    "command_id": command["id"],
                                    "state": "applied",
                                    "effective_after_step": run.step,
                                },
                            )
                if self.rank == 0 and self.evaluation_job:
                    self._evaluation_worker(directory)
                if any(c["kind"] == "stop" for c in commands):
                    self._checkpoint(directory, commands)
                    stopped = True
                    break
                batches, epoch_ended = [], False
                for _ in range(cfg.gradient_accumulation_steps):
                    try:
                        batches.append(next(iterator))
                    except StopIteration:
                        epoch_ended = True
                        break
                if self.world > 1:
                    boundaries = [None] * self.world
                    dist.all_gather_object(boundaries, (len(batches), epoch_ended))
                    if len(set(boundaries)) != 1:
                        raise ValueError(
                            "DDP loaders must yield equal batch counts per epoch on every rank. Shard and batch the dataset consistently."
                        )
                if not batches:
                    if self.batches_in_epoch == 0:
                        raise ValueError(
                            f"Training loader yielded no batches on rank {self.rank}. Supply a nonempty, repeatable loader for epoch training."
                        )
                    iterator = self._advance_epoch()
                    self._progress()
                    continue
                if cfg.compile_mode in ("reduce-overhead", "max-autotune"):
                    torch.compiler.cudagraph_mark_step_begin()
                self.batches_in_epoch += len(batches)
                self.batches_processed += len(batches)
                self.optimizer.zero_grad(set_to_none=True)
                weight, tokens = 0.0, 0
                loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
                metrics, records = {}, []
                for index, batch in enumerate(batches):
                    sync = (
                        self.model.no_sync()
                        if self.world > 1 and index < len(batches) - 1
                        else contextlib.nullcontext()
                    )
                    with sync, self._amp():
                        value = self.training_step_fn(self.model, batch, self.device)
                        if (
                            self.tuning_applied
                            and len(set(cfg.rank_micro_batch_sizes)) > 1
                            and not isinstance(value, StepResult)
                        ):
                            raise ValueError(
                                "Unequal tuned DDP batches require StepResult(loss, weight=supervised_token_count) or another correct objective weight. Return a weighted StepResult, or disable GPU tuning for this stage."
                            )
                        result = self._result(value, check_finite=False)
                        scaled_loss = result.loss * (
                            result.weight / cfg.gradient_accumulation_steps
                        )
                    self.scaler.scale(scaled_loss).backward()
                    weight += result.weight
                    loss_sum += result.loss.detach().to(torch.float64) * result.weight
                    for key, value in result.metrics.items():
                        metrics[key] = (
                            metrics.get(key, 0.0)
                            + (value.detach() if isinstance(value, Tensor) else value)
                            * result.weight
                        )
                    if isinstance(batch, dict):
                        inputs = batch.get("input_ids")
                        tokens += inputs.numel() if isinstance(inputs, Tensor) else 0
                    if cfg.trace_batches != "none":
                        records.append(self._batch_trace(batch))
                stats = torch.stack(
                    (loss_sum.new_tensor(weight), loss_sum, loss_sum.new_tensor(tokens))
                )
                if self.world > 1:
                    dist.all_reduce(stats)
                # One scalar transfer at the update boundary, after all ranks
                # agree on the objective. No per-microbatch loss synchronization;
                # a nonfinite loss on any rank stops every worker before stepping.
                host_stats = stats.tolist()
                if not math.isfinite(host_stats[1]):
                    raise ValueError("Nonfinite loss; optimizer update was not applied")
                self.scaler.unscale_(self.optimizer)
                factor = self.world * cfg.gradient_accumulation_steps / stats[0]
                gradients = [
                    p.grad for p in self.model.parameters() if p.grad is not None
                ]
                # One foreach launch per device/dtype rather than one per tensor.
                groups = {}
                for grad in gradients:
                    groups.setdefault((grad.device, grad.dtype), []).append(grad)
                for (device, _), group in groups.items():
                    torch._foreach_mul_(group, factor.to(device))
                norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.gradient_clip,
                    error_if_nonfinite=cfg.precision != "fp16",
                    foreach=None,
                )
                applied_lrs = [group["lr"] for group in self.optimizer.param_groups]
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skipped = self.scaler.get_scale() < old_scale
                run.tokens += int(host_stats[2])
                if skipped:
                    overflow_count += 1
                    run.warning(
                        "FP16 gradient overflow: optimizer and scheduler update skipped. "
                        f"Loss scale reduced from {old_scale:g} to {self.scaler.get_scale():g}; "
                        "training will retry on the next batch. This can happen while "
                        "automatic loss scaling settles at startup.",
                        code="amp-overflow",
                    )
                    if overflow_count > 20:
                        raise ValueError(
                            "Repeated FP16 overflow; lower the loss scale/learning rate or use fp32"
                        )
                else:
                    overflow_count = 0
                    run.step += 1
                    if self.scheduler is not None:
                        self.scheduler.step()
                if records:
                    entry = {
                        "step": run.step,
                        "skipped_update": skipped,
                        "epoch": self.epoch,
                        "batches": records,
                    }
                    trace.write(json.dumps(entry) + "\n")
                    run.emit("data_trace", {"rank": self.rank, **entry})
                    if not missing_trace_warned and any(
                        record.get("source") is None for record in records
                    ):
                        run.warning(
                            "Loader batches contain no _weave_source metadata. Batch counts are recorded; exact consumed file/row attribution is unavailable.",
                            code="batch-provenance",
                        )
                        missing_trace_warned = True
                if epoch_ended:
                    iterator = self._advance_epoch()
                if self.profiler is not None:
                    self.profiler.step()
                if skipped:
                    self._progress()
                    continue
                if run.step % cfg.log_every == 0 or run.step == 1:
                    metrics = {key: float(value) for key, value in metrics.items()}
                    profile = {
                        "rank": self.rank,
                        "device": self.device,
                        "microbatches": [
                            self._batch_shapes(batch) for batch in batches
                        ],
                    }
                    profiles = [profile]
                    values = {
                        "train/loss": host_stats[1] / host_stats[0],
                        "train/gradient_norm": float(norm),
                        "train/tokens_per_second": (run.tokens - initial_tokens)
                        / max(0.001, time.monotonic() - started),
                    }
                    if self.world > 1:
                        all_metrics = [None] * self.world
                        dist.all_gather_object(
                            all_metrics, {"metrics": metrics, "profile": profile}
                        )
                        profiles = [item["profile"] for item in all_metrics]
                        metrics = {
                            key: sum(
                                item["metrics"].get(key, 0.0) for item in all_metrics
                            )
                            for key in {
                                k for item in all_metrics for k in item["metrics"]
                            }
                        }
                    run.emit("batch_profile", {"ranks": profiles})
                    if self.rank == 0:
                        print(
                            f"Update {run.step} · input batches: {json.dumps(profiles)}",
                            flush=True,
                        )
                    values.update(
                        {
                            "train/" + key: value / host_stats[0]
                            for key, value in metrics.items()
                        }
                    )
                    values.update(
                        {f"lr/group_{i}": lr for i, lr in enumerate(applied_lrs)}
                    )
                    run.log(values, step=run.step, tokens=run.tokens)
                    self._progress()
                checkpoint = None
                if (
                    run.step % cfg.checkpoint_every == 0
                    or self._training_done()
                    or any(c["kind"] == "snapshot" for c in commands)
                ):
                    checkpoint = self._checkpoint(directory, commands)
                if run.step % cfg.eval_every == 0 or self._training_done():
                    if self.rank == 0:
                        if cfg.background_evaluation:
                            checkpoint = checkpoint or self._evaluation_snapshot(
                                directory
                            )
                            self._queue_evaluation(directory / checkpoint)
                        else:
                            self._evaluate(checkpoint)
                    last_evaluated_step = run.step
                    if self.world > 1 and not cfg.background_evaluation:
                        dist.barrier()
                if self.rank == 0 and cfg.background_evaluation:
                    self._evaluation_worker(directory)

        self._progress()
        checkpoint = self._last_checkpoint_path
        if self._last_checkpoint_boundary != self._boundary():
            checkpoint = self._checkpoint(directory, [])
        if not stopped and last_evaluated_step != run.step:
            if self.rank == 0:
                if cfg.background_evaluation:
                    self._queue_evaluation(directory / checkpoint)
                    self._evaluation_worker(directory)
                else:
                    self._evaluate(checkpoint)
            if self.world > 1:
                dist.barrier()
        if stopped and self.rank == 0:
            run.emit("stopped", {"checkpoint": checkpoint, "reason": "user"})
        return stopped

    def _training_done(self):
        cfg = self.config
        return (cfg.max_epochs is not None and self.epoch >= cfg.max_epochs) or (
            cfg.max_optimizer_steps is not None
            and self.run.step >= cfg.max_optimizer_steps
        )

    def _advance_epoch(self):
        self.epoch += 1
        self.batches_in_epoch = 0
        sampler = getattr(self.train_loader, "sampler", None)
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(self.epoch)
        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        if callable(getattr(batch_sampler, "set_epoch", None)):
            batch_sampler.set_epoch(self.epoch)
        dataset = getattr(self.train_loader, "dataset", None)
        if callable(getattr(dataset, "reset", None)):
            dataset.reset()
        return None if self._training_done() else iter(self.train_loader)

    def _boundary(self):
        return (
            self.run.step,
            self.epoch,
            self.batches_processed,
            self.batches_in_epoch,
        )

    def _progress(self):
        self.run.emit(
            "progress",
            {
                "completed": self.run.step,
                "total": self.config.total_optimizer_steps,
                "unit": "optimizer updates",
                "epochs_completed": self.epoch,
                "max_epochs": self.config.max_epochs,
                "batches_processed": self.batches_processed,
                "batches_in_epoch": self.batches_in_epoch,
                "batches_per_epoch": self.batches_per_epoch,
                "world_size": self.world,
                "optimizer_updates": self.run.step,
                "tokens_processed": self.run.tokens,
            },
        )

    @staticmethod
    def _batch_shapes(batch):
        shapes = {}

        def visit(value, path):
            if isinstance(value, Tensor):
                shapes[path or "input"] = list(value.shape)
            elif isinstance(value, dict):
                for name, item in value.items():
                    if not str(name).startswith("_weave"):
                        visit(item, f"{path}.{name}" if path else str(name))
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    visit(item, f"{path}.{index}" if path else str(index))

        visit(batch, "")
        return shapes

    def _batch_trace(self, batch):
        result = (
            {"source": batch.get("_weave_source")}
            if isinstance(batch, dict) and "_weave_source" in batch
            else {"source": None}
        )
        result["shapes"] = self._batch_shapes(batch)
        if self.config.trace_batches == "sha256":
            digest = hashlib.sha256()

            def visit(value):
                if isinstance(value, Tensor):
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                    digest.update(
                        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                    )
                elif isinstance(value, dict):
                    for key in sorted(value):
                        digest.update(str(key).encode())
                        visit(value[key])
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        visit(item)
                else:
                    digest.update(repr(value).encode())

            visit(batch)
            result["sha256"] = digest.hexdigest()

        # Batch metadata may have been collated into tensors by DataLoader.
        def plain(value):
            if isinstance(value, Tensor):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value

        return plain(result)

    def _checkpoint(self, directory, commands):
        cfg, run = self.config, self.run
        local = {
            "rng": capture_rng(),
            "loader": self._loader_state(),
            "batches_in_epoch": self.batches_in_epoch,
            "batches_processed": self.batches_processed,
        }
        states = [local]
        if self.world > 1:
            states = [None] * self.world
            dist.all_gather_object(states, local)
        path = directory / "checkpoints" / f"step-{run.step}-{time.time_ns()}.pt"
        if self.rank == 0:
            raw = self.model.module if isinstance(self.model, DDP) else self.model
            state = {
                "trainer_version": 2,
                "model": raw.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_structure": self._optimizer_structure(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "scaler": self.scaler.state_dict(),
                "step": run.step,
                "tokens": run.tokens,
                "config": asdict(cfg),
                "datasets": self.manifest,
                "rank_state": states,
                "code_sha256": self.code_sha256,
            }
            with atomic_checkpoint_path(path) as temporary:
                torch.save(state, temporary)
            resumable = all(item["loader"] is not None for item in states)
            run.checkpoint(
                path,
                metadata={
                    "resume": (
                        "full trainer and loader state"
                        if resumable
                        else "weights only; loader has no state protocol"
                    ),
                    "exact_resume": resumable,
                    "snapshot_requests": [
                        c["id"] for c in commands if c["kind"] == "snapshot"
                    ],
                },
            )
        if self.world > 1:
            dist.barrier()
        self._last_checkpoint_boundary = self._boundary()
        self._last_checkpoint_path = (
            str(path.relative_to(directory)) if self.rank == 0 else None
        )
        return self._last_checkpoint_path

    def _validation_settings(self):
        return {
            "mode": "async" if self.config.background_evaluation else "inline",
            "device": (
                self.config.evaluation_device
                if self.config.evaluation_device is not None
                else "cpu"
            ),
            "eval_every": self.config.eval_every,
            "eval_batches": self.config.eval_batches,
        }

    def _set_validation(self, settings):
        candidate = copy.deepcopy(self.config)
        if settings["mode"] not in ("async", "inline"):
            raise ValueError("Unknown validation mode")
        candidate.async_evaluation = settings["mode"] == "async"
        candidate.evaluation_device = (
            settings["device"] if candidate.async_evaluation else None
        )
        candidate.eval_every = settings["eval_every"]
        candidate.eval_batches = settings["eval_batches"]
        # train_eval is a legacy alias; switching validation must not change
        # the training model's placement or distributed process group.
        if candidate.strategy == "train_eval":
            candidate.strategy = "single"
        candidate.__post_init__()
        for key in (
            "async_evaluation",
            "evaluation_device",
            "eval_every",
            "eval_batches",
            "strategy",
        ):
            setattr(self.config, key, getattr(candidate, key))

    def _evaluation_worker(self, directory, wait=False):
        if self.evaluation_job:
            if wait:
                self.evaluation_job.wait()
            if self.evaluation_job.poll() is None:
                return
            failed = bool(self.evaluation_job.returncode)
            if failed:
                self.run.warning(
                    "Separate evaluation failed; see evaluation.log",
                    code="evaluation-failed",
                )
            self.run.emit(
                "evaluation",
                {
                    "state": ("failed" if failed else "completed"),
                    "checkpoint": self.evaluation_active,
                    "checkpoint_step": self.evaluation_active_step,
                    "device": self.evaluation_active_device,
                },
            )
            self._discard_evaluation_snapshot(self.evaluation_active)
            if isinstance(self.evaluation_job, EvaluationSession):
                result = self.evaluation_job.result or {}
                if not result.get("reusable", False):
                    self.evaluation_reuse_unavailable = not failed
                    if not failed:
                        self.run.warning(
                            "Validation loader cannot be safely reused; using a fresh process for each pass. Return a repeatable loader without persistent data workers to enable reuse.",
                            code="evaluation-reuse-unavailable",
                        )
                if failed or not result.get("reusable", False):
                    self._close_evaluation_session()
            self.evaluation_job = None
        if not self.config.background_evaluation:
            self._close_evaluation_session()
        if self.evaluation_pending:
            if not self.config.background_evaluation:
                self._discard_evaluation_snapshot(self.evaluation_pending)
                self.evaluation_pending = None
                return
            environment = dict(os.environ, WEAVE_RUN_DIR=str(directory), USE_TORCH="1")
            environment["WEAVE_VALIDATION_SETTINGS"] = json.dumps(
                self._validation_settings()
            )
            # Do not let an evaluator join the parent's distributed process group.
            for key in list(environment):
                if key in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "GROUP_RANK",
                    "ROLE_RANK",
                    "ROLE_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                ) or key.startswith("TORCHELASTIC_"):
                    environment.pop(key)
            reuse = (
                self.config.reuse_evaluation_worker
                and not self.evaluation_reuse_unavailable
            )
            device = self._validation_settings()["device"]
            session = self.evaluation_session
            if session and (
                session.device != device or session.process.poll() is not None
            ):
                self._close_evaluation_session()
                session = None
            if reuse:
                if session is None:
                    session = EvaluationSession(
                        directory / "evaluation-workers" / str(time.time_ns()), device
                    )
                session.submit(self.evaluation_pending, self._validation_settings())
                if session.process is None:
                    with (directory / "evaluation.log").open("a") as output:
                        session.process = subprocess.Popen(
                            [
                                sys.executable,
                                str(self.source_directory / "train.py"),
                                "--evaluation-worker",
                                str(session.directory),
                            ],
                            cwd=self.source_directory,
                            env=environment,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                        )
                self.evaluation_session = session
                self.evaluation_job = session
            else:
                self._close_evaluation_session()
                with (directory / "evaluation.log").open("a") as output:
                    self.evaluation_job = subprocess.Popen(
                        [
                            sys.executable,
                            str(self.source_directory / "train.py"),
                            "--evaluate",
                            self.evaluation_pending,
                        ],
                        cwd=self.source_directory,
                        env=environment,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                    )
            self.evaluation_active = self.evaluation_pending
            self.evaluation_active_step = self.evaluation_pending_step
            self.evaluation_active_device = (
                self.config.evaluation_device
                if self.config.evaluation_device is not None
                else "cpu"
            )
            self.run.emit(
                "evaluation",
                {
                    "state": "running",
                    "checkpoint": self.evaluation_active,
                    "checkpoint_step": self.evaluation_active_step,
                    "device": (
                        self.config.evaluation_device
                        if self.config.evaluation_device is not None
                        else "cpu"
                    ),
                },
            )
            self.evaluation_pending = None

    def _cancel_evaluation(self, directory):
        self._discard_evaluation_snapshot(self.evaluation_pending)
        self.evaluation_pending = None
        job = self.evaluation_job
        if job is None:
            self._close_evaluation_session()
            return
        if job.poll() is not None:
            self._evaluation_worker(directory)
            return
        job.terminate()
        try:
            job.wait(timeout=10)
        except subprocess.TimeoutExpired:
            job.kill()
            job.wait()
        self.run.emit(
            "evaluation",
            {
                "state": "cancelled",
                "checkpoint": self.evaluation_active,
                "checkpoint_step": self.evaluation_active_step,
                "device": self.evaluation_active_device,
                "reason": "training stopped by user",
            },
        )
        self._discard_evaluation_snapshot(self.evaluation_active)
        self.evaluation_job = None
        self._close_evaluation_session()

    def _evaluation_snapshot(self, directory):
        """Immutable weights only; no optimizer/loader serialization for validation."""
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        if (
            self.config.evaluation_snapshot_storage == "auto"
            and not self.evaluation_memory_disabled
        ):
            if self.evaluation_memory is None:
                size = sum(
                    value.numel() * value.element_size()
                    for value in itertools.chain(raw.parameters(), raw.buffers())
                )
                self.evaluation_memory = memory_snapshot_directory(size)
        root = (
            Path(self.evaluation_memory.name)
            if self.config.evaluation_snapshot_storage == "auto"
            and self.evaluation_memory
            and not self.evaluation_memory_disabled
            else directory
        )
        name = f"step-{self.run.step}-{time.time_ns()}.pt"
        state = {
            "model": raw.state_dict(),
            "step": self.run.step,
            "tokens": self.run.tokens,
        }
        path = root / "evaluation-weights" / name

        def save(target):
            with atomic_checkpoint_path(target) as temporary:
                torch.save(state, temporary)

        started = time.monotonic()
        try:
            save(path)
        except OSError as exc:
            if root == directory or exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            # Do not remove older RAM snapshots: an active worker may read one.
            self.evaluation_memory_disabled = True
            path = directory / "evaluation-weights" / name
            save(path)
        self.run.emit(
            "evaluation_snapshot",
            {
                "storage": "disk" if path.is_relative_to(directory) else "memory",
                "seconds": time.monotonic() - started,
            },
        )
        return (
            str(path.relative_to(directory))
            if path.is_relative_to(directory)
            else str(path)
        )

    def _discard_evaluation_snapshot(self, path):
        # Published resumable checkpoints are retained; only private validation
        # snapshots are disposable after a worker finishes or a newer job replaces it.
        if path and Path(path).parent.name == "evaluation-weights":
            Path(path).unlink(missing_ok=True)

    def _queue_evaluation(self, path):
        if self.evaluation_pending:
            self._discard_evaluation_snapshot(self.evaluation_pending)
        self.evaluation_pending = str(path)
        self.evaluation_pending_step = self.run.step
        self.run.emit(
            "evaluation",
            {
                "state": "queued",
                "checkpoint": str(path),
                "checkpoint_step": self.run.step,
                "device": (
                    self.config.evaluation_device
                    if self.config.evaluation_device is not None
                    else "cpu"
                ),
            },
        )

    def _evaluate(self, checkpoint):
        cfg, run = self.config, self.run
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        was_training = raw.training
        # Preserve only the devices this model uses. CPU validation must not
        # initialize CUDA contexts, and a V100 worker must not touch GPU 0.
        rng = {"python_rng": random.getstate(), "torch_rng": torch.get_rng_state()}
        devices = {
            value.device
            for value in itertools.chain(raw.parameters(), raw.buffers())
            if value.device.type == "cuda"
        }
        cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
        numpy = sys.modules.get("numpy")
        numpy_rng = numpy.random.get_state() if numpy is not None else None
        loader = self.val_loader
        target = (
            loader
            if callable(getattr(loader, "state_dict", None))
            else getattr(loader, "dataset", None)
        )
        data_state = (
            copy.deepcopy(target.state_dict())
            if callable(getattr(target, "state_dict", None))
            and callable(getattr(target, "load_state_dict", None))
            else None
        )
        generators = {}
        for owner in (
            loader,
            getattr(loader, "sampler", None),
            getattr(getattr(loader, "batch_sampler", None), "sampler", None),
        ):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator):
                generators[id(generator)] = (generator, generator.get_state())
        raw.eval()
        try:
            with torch.inference_mode(), self._amp():
                loss, weight, metrics = 0.0, 0.0, {}
                if self.val_loader is not None:
                    run.emit(
                        "evaluation_progress",
                        {"completed": 0, "total": cfg.eval_batches},
                    )
                    for batch_index, batch in enumerate(
                        itertools.islice(self.val_loader, cfg.eval_batches), 1
                    ):
                        value = self._result(
                            self.evaluation_step_fn(raw, batch, self.device)
                        )
                        loss += float(value.loss) * value.weight
                        weight += value.weight
                        for key, item in value.metrics.items():
                            metrics[key] = (
                                metrics.get(key, 0.0) + float(item) * value.weight
                            )
                        run.emit(
                            "evaluation_progress",
                            {
                                "completed": batch_index,
                                "total": cfg.eval_batches,
                                "loss_running": (
                                    loss / weight
                                    if weight and math.isfinite(loss / weight)
                                    else None
                                ),
                            },
                        )
                    if weight:
                        run.log(
                            {
                                "eval/loss": loss / weight,
                                **{
                                    "eval/" + key: value / weight
                                    for key, value in metrics.items()
                                },
                            },
                            step=run.step,
                            tokens=run.tokens,
                        )
                if self.samples_fn:
                    for sample in self.samples_fn(
                        raw, self.tokenizer, cfg, self.device
                    ):
                        run.sample(checkpoint=checkpoint, **sample)
                else:
                    self._samples(raw, checkpoint)
        finally:
            if data_state is not None:
                target.load_state_dict(data_state)
            for generator, state in generators.values():
                generator.set_state(state)
            raw.train(was_training)
            restore_rng(rng)
            for device, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            if numpy_rng is not None:
                numpy.random.set_state(numpy_rng)

    def _samples(self, model, checkpoint):
        cfg = self.config
        if not cfg.sample_prompts:
            return
        text_evaluate(
            model,
            [],
            self.tokenizer,
            {
                "prompts": cfg.sample_prompts,
                "eval_batches": cfg.eval_batches,
                "batch_size": cfg.micro_batch_size,
                "generation_tokens": cfg.generation_tokens,
                "generation_context_length": cfg.generation_context_length,
                "seed": cfg.seed,
            },
            self.run,
            checkpoint,
            self.device,
        )
