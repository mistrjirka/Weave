"""Trainer state responsibilities."""

from __future__ import annotations


class TrainerState:
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
