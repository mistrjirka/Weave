"""Optional PyTorch orchestration for user-owned recipes. No web dependencies.

The script owns model/data/loss/optimizer/scheduler. This helper handles devices,
AMP, accumulation, durable reporting, cooperative checkpoints and provenance.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
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
from weave_training import Run, atomic_json, atomic_checkpoint_path, file_hash


@dataclass
class TrainingConfig:
    devices: list[int] = field(default_factory=list)
    strategy: Literal["auto", "single", "ddp", "split", "train_eval"] = "auto"
    placement: dict[str, int] = field(default_factory=dict)
    evaluation_device: int | Literal["cpu"] | None = None
    async_evaluation: bool = False
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
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
    cpu_threads: int = 2
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

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--evaluate")
        parser.add_argument("--checkpoint")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--output-dir")
        args = parser.parse_args()
        cfg = self.config
        context_path = Path(__file__).parent / "launch.json"
        launch = json.loads(context_path.read_text()) if context_path.is_file() else {}
        mode = launch.get("evaluation_mode", "script")
        if mode != "script":
            cfg.async_evaluation = mode == "async"
            cfg.evaluation_device = launch.get("evaluation_device")
            if mode == "inline" and cfg.strategy == "train_eval":
                cfg.strategy = "single"
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
            if self.evaluation_job and self.evaluation_job.poll() is None:
                self.evaluation_job.terminate()
                self.evaluation_job.wait()
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
        if cfg.precision == "fp16" and self.device == "cpu":
            raise ValueError("FP16 requires CUDA; use fp32 or bf16 on CPU")
        if cfg.precision == "bf16" and self.device != "cpu":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError(
                        "Selected GPU does not support native BF16; choose fp16 or fp32"
                    )
        self.tokenizer = (
            self.tokenizer_factory()
            if callable(self.tokenizer_factory)
            else self.tokenizer_factory
        )
        self.train_loader, self.val_loader = self.dataloaders_fn(self.tokenizer, cfg)
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
        checkpoint = args.evaluate or args.checkpoint or launch.get("checkpoint")
        resume = args.resume or launch.get("start_mode") == "resume"
        step, tokens = 0, 0
        directory.mkdir(parents=True, exist_ok=True)
        report_directory = (
            directory / ("evaluation-" + Path(args.evaluate).stem)
            if args.evaluate
            else directory
        )
        with Run(
            report_directory,
            rank=self.rank,
            source_directory=self.source_directory,
            channel="evaluation" if args.evaluate else "training",
        ) as run:
            self.run = run
            if (
                not args.evaluate
                and cfg.background_evaluation
                and cfg.evaluation_device in cfg.devices
            ):
                run.warning(
                    "Background validation shares a training GPU; it needs another model copy and competes for memory and compute. Select CPU or a separate GPU to avoid GPU contention.",
                    code="evaluation-contention",
                )
            run.capabilities(
                metrics=True,
                samples=True,
                checkpoint=not bool(args.evaluate),
                commands=not bool(args.evaluate),
            )
            if self.rank == 0:
                print(f"Run outputs: {directory}", flush=True)
                atomic_json(report_directory / "resolved-config.json", asdict(cfg))
                run.emit("configuration", asdict(cfg))
            self.manifest = self._sources(report_directory, run)
            self.code_sha256 = self._code_hash()
            if checkpoint:
                path = Path(checkpoint).expanduser()
                if not path.is_absolute():
                    path = self.source_directory / path
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state["model"])
                if resume and not args.evaluate:
                    self._restore(state)
                    self.resuming = True
                    step, tokens = state["step"], state["tokens"]
                elif args.evaluate:
                    step, tokens = state["step"], state["tokens"]
            run.step, run.tokens = step, tokens
            if args.evaluate:
                self._evaluate(str(args.evaluate))
                return
            if self.world > 1:
                dataset = getattr(self.train_loader, "dataset", None)
                sampler = getattr(self.train_loader, "sampler", None)
                from torch.utils.data.distributed import DistributedSampler

                if not isinstance(sampler, DistributedSampler) and not getattr(
                    dataset, "distributed_sharding", False
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
            self._loop(directory)
            if self.rank == 0 and cfg.background_evaluation:
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

    def _restore(self, state):
        if state.get("trainer_version") != 2:
            raise ValueError(
                "This checkpoint uses a different trainer. Load weights into a new stage instead."
            )
        if state.get("code_sha256") != self.code_sha256:
            raise ValueError(
                "Training/model code changed since checkpoint; start a new stage from weights"
            )
        previous = state["config"]
        current = asdict(self.config)
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
            "evaluation_device",
            "evaluation_only",
        }
        if any(
            previous.get(key) != value
            for key, value in current.items()
            if key not in mutable
        ):
            raise ValueError(
                "Training configuration differs from checkpoint; start a new stage from weights"
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
    def _result(value):
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
        if not torch.isfinite(result.loss):
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
                self.batches_in_epoch += len(batches)
                self.batches_processed += len(batches)
                self.optimizer.zero_grad(set_to_none=True)
                weight, loss_sum, tokens = 0.0, 0.0, 0
                metrics, records = {}, []
                for index, batch in enumerate(batches):
                    sync = (
                        self.model.no_sync()
                        if self.world > 1 and index < len(batches) - 1
                        else contextlib.nullcontext()
                    )
                    with sync, self._amp():
                        result = self._result(
                            self.training_step_fn(self.model, batch, self.device)
                        )
                        scaled_loss = result.loss * (
                            result.weight / cfg.gradient_accumulation_steps
                        )
                    self.scaler.scale(scaled_loss).backward()
                    weight += result.weight
                    loss_sum += float(result.loss.detach()) * result.weight
                    for key, value in result.metrics.items():
                        metrics[key] = (
                            metrics.get(key, 0.0)
                            + float(
                                value.detach() if isinstance(value, Tensor) else value
                            )
                            * result.weight
                        )
                    if isinstance(batch, dict):
                        inputs = batch.get("input_ids")
                        tokens += inputs.numel() if isinstance(inputs, Tensor) else 0
                    if cfg.trace_batches != "none":
                        records.append(self._batch_trace(batch))
                stats = torch.tensor(
                    [weight, loss_sum, tokens], dtype=torch.float64, device=self.device
                )
                if self.world > 1:
                    dist.all_reduce(stats)
                self.scaler.unscale_(self.optimizer)
                factor = self.world * cfg.gradient_accumulation_steps / stats[0].item()
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(factor)
                norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.gradient_clip,
                    error_if_nonfinite=cfg.precision != "fp16",
                    foreach=False,
                )
                applied_lrs = [group["lr"] for group in self.optimizer.param_groups]
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skipped = self.scaler.get_scale() < old_scale
                run.tokens += int(stats[2].item())
                if skipped:
                    overflow_count += 1
                    run.warning(
                        "FP16 gradient overflow: optimizer and scheduler update skipped",
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
                if skipped:
                    self._progress()
                    continue
                if run.step % cfg.log_every == 0 or run.step == 1:
                    values = {
                        "train/loss": stats[1].item() / stats[0].item(),
                        "train/gradient_norm": float(norm),
                        "train/tokens_per_second": (run.tokens - initial_tokens)
                        / max(0.001, time.monotonic() - started),
                    }
                    if self.world > 1:
                        all_metrics = [None] * self.world
                        dist.all_gather_object(all_metrics, metrics)
                        metrics = {
                            key: sum(item.get(key, 0.0) for item in all_metrics)
                            for key in {k for item in all_metrics for k in item}
                        }
                    values.update(
                        {
                            "train/" + key: value / stats[0].item()
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

    def _batch_trace(self, batch):
        result = (
            {"source": batch.get("_weave_source")}
            if isinstance(batch, dict) and "_weave_source" in batch
            else {"source": None}
        )
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

    def _evaluation_worker(self, directory, wait=False):
        if self.evaluation_job:
            if wait:
                self.evaluation_job.wait()
            if self.evaluation_job.poll() is None:
                return
            if self.evaluation_job.returncode:
                self.run.warning(
                    "Separate evaluation failed; see evaluation.log",
                    code="evaluation-failed",
                )
            self.run.emit(
                "evaluation",
                {
                    "state": (
                        "failed" if self.evaluation_job.returncode else "completed"
                    ),
                    "checkpoint": self.evaluation_active,
                    "checkpoint_step": self.evaluation_active_step,
                    "device": (
                        self.config.evaluation_device
                        if self.config.evaluation_device is not None
                        else "cpu"
                    ),
                },
            )
            self._discard_evaluation_snapshot(self.evaluation_active)
            self.evaluation_job = None
        if self.evaluation_pending:
            environment = dict(os.environ, WEAVE_RUN_DIR=str(directory), USE_TORCH="1")
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

    def _evaluation_snapshot(self, directory):
        """Immutable weights only; no optimizer/loader serialization for validation."""
        path = (
            directory
            / "evaluation-weights"
            / f"step-{self.run.step}-{time.time_ns()}.pt"
        )
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        with atomic_checkpoint_path(path) as temporary:
            torch.save(
                {
                    "model": raw.state_dict(),
                    "step": self.run.step,
                    "tokens": self.run.tokens,
                },
                temporary,
            )
        return str(path.relative_to(directory))

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
        was_training, rng = raw.training, capture_rng()
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
        raw.eval()
        try:
            with torch.inference_mode(), self._amp():
                loss, weight, metrics = 0.0, 0.0, {}
                if self.val_loader is not None:
                    for batch in itertools.islice(self.val_loader, cfg.eval_batches):
                        value = self._result(
                            self.evaluation_step_fn(raw, batch, self.device)
                        )
                        loss += float(value.loss) * value.weight
                        weight += value.weight
                        for key, item in value.metrics.items():
                            metrics[key] = (
                                metrics.get(key, 0.0) + float(item) * value.weight
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
            raw.train(was_training)
            restore_rng(rng)

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
