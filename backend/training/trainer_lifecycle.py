"""Trainer lifecycle responsibilities."""

from __future__ import annotations


class TrainerLifecycle:
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
