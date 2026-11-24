"""Trainer loop responsibilities."""

from __future__ import annotations


class TrainerLoop:
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
