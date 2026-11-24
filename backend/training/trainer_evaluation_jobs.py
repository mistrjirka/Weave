"""Trainer evaluation_jobs responsibilities."""

from __future__ import annotations


class TrainerEvaluationJobs:
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
