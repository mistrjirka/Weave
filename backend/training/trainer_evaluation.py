"""Trainer evaluation responsibilities."""

from __future__ import annotations


class TrainerEvaluation:
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
