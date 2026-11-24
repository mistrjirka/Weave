"""Training documents, immutable run exports, and verified portable archives."""

from __future__ import annotations

import ast
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import sys
import time
import uuid
import zipfile
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict
from ..compiler import export_zip
from ..core import validate_project
from .weave_training import (
    atomic_json,
    file_hash,
    instrumentation_warnings,
    available_cpu_threads,
)
from .layout import PACKAGE, normalize_script, write_support
from .recipe import is_code_first, literal_settings

from . import service as service_module
from .service import TrainingConfig, inside, valid_id

HERE = Path(__file__).parent

class TrainingLaunch:
    def launch(
        self,
        request,
        *,
        frozen_source=None,
        validation_settings=None,
        restart_record=None,
        edited_files_source=None,
    ):
        project, stage = request.project, request.stage
        stage = stage.model_copy(
            update={"script": normalize_script(stage.script, stage.config.recipe)}
        )
        code_first = is_code_first(stage.script)
        if code_first:
            settings = literal_settings(stage.script)
            controls = stage.config.model_dump(
                include={
                    "recipe",
                    "execution",
                    "python",
                    "start_mode",
                    "parent_run",
                    "parent_checkpoint",
                    "evaluation_mode",
                    "compute_mode",
                    "use_gpu_tuning",
                    "performance_mode",
                }
            )
            if stage.config.evaluation_mode != "script":
                controls["evaluation_device"] = stage.config.evaluation_device
            if stage.config.performance_mode == "override":
                controls.update(
                    stage.config.model_dump(
                        include={"compile_mode", "fused_optimizer", "profile_steps"}
                    )
                )
            if stage.config.compute_mode == "override":
                controls.update(
                    stage.config.model_dump(
                        include={"strategy", "devices", "precision", "placement"}
                    )
                )
            stage = stage.model_copy(
                update={"config": TrainingConfig(**{**settings, **controls})}
            )
        if not code_first:
            stage.config.validate_execution()
        elif stage.config.compute_mode == "override":
            stage.config.validate_execution()
            if stage.config.strategy == "split" and stage.config.precision != "fp32":
                raise ValueError("Split training currently requires FP32 in Weave.")
        elif stage.config.start_mode != "random" and not (
            stage.config.parent_run and stage.config.parent_checkpoint
        ):
            raise ValueError("Choose a parent run and checkpoint.")
        report = validate_project(project)
        if not report["valid"]:
            raise ValueError(
                "Resolve graph errors before training: "
                + "; ".join(e["message"] for e in report["errors"][:3])
            )
        ast.parse(stage.script)
        run_id = "run-" + uuid.uuid4().hex
        directory = self.root / "runs" / run_id
        directory.mkdir()
        try:
            from ..tokenizer_bindings import input_tokenizers

            directories = {
                info["id"]: self.tokenizer_root / valid_id(info["id"])
                for info in input_tokenizers(project).values()
            }
            if frozen_source is None:
                bundle = export_zip(project, tokenizer_dirs=directories)
                write_support(directory, bundle)
                extras = self.workspaces.freeze_extras(
                    project["id"], stage.id, directory
                )
            else:
                from .layout import rewrite_imports, runtime_upgrade_helpers
                from .run_controls import VALIDATION_UPGRADE_FROM

                previous_trainer = frozen_source / PACKAGE / "trainer.py"
                current_trainer = rewrite_imports(
                    (HERE / "trainer.py").read_text(), "."
                )
                if (
                    code_first
                    and file_hash(previous_trainer) not in VALIDATION_UPGRADE_FROM
                    and previous_trainer.read_text() != current_trainer
                ):
                    raise ValueError(
                        "This trainer revision needs a new stage from weights; its exact-resume upgrade has not been verified."
                    )
                if restart_record and restart_record.get("refresh_support"):
                    # Re-export the *frozen graph*, never an independently edited
                    # project. Tensor and optimizer identity are checked on restore.
                    bundle = export_zip(project, tokenizer_dirs=directories)
                    write_support(directory, bundle)
                else:
                    shutil.copytree(frozen_source / PACKAGE, directory / PACKAGE)
                    # The upgraded trainer imports this new helper even when
                    # keeping the model/runtime from an older frozen run.
                    if code_first:
                        refreshed = ("performance.py", "evaluation_worker.py", "weave_training.py", "run_contract.py")
                        for helper in (*refreshed, *runtime_upgrade_helpers((*refreshed, "trainer.py"))):
                            (directory / PACKAGE / helper).write_text(
                                rewrite_imports((HERE / helper).read_text(), ".")
                            )
                if code_first:
                    (directory / PACKAGE / "trainer.py").write_text(current_trainer)
                if restart_record and restart_record["script_source"] == "edited":
                    extras = list(restart_record["user_files"])
                    for name in extras:
                        target = inside(directory, name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(inside(edited_files_source, name), target)
                else:
                    extras = json.loads(
                        (frozen_source / ".workspace-files.json").read_text()
                    )
                    for name in extras:
                        source, target = inside(frozen_source, name), inside(
                            directory, name
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
            atomic_json(directory / ".workspace-files.json", extras)
            files = {"model.py": (directory / PACKAGE / "model.py").read_text()}
            metadata = service_module.model_metadata(files, project)
            if stage.config.strategy == "split" and not metadata["split_supported"]:
                raise ValueError(" ".join(metadata["split_reasons"]))
            config = stage.config.model_dump()
            code_digest = hashlib.sha256()
            for artifact in sorted(directory.rglob("*")):
                if artifact.is_file() and (
                    artifact.suffix == ".py" or "tokenizers" in artifact.parts
                ):
                    code_digest.update(str(artifact.relative_to(directory)).encode())
                    code_digest.update(file_hash(artifact).encode())
            config["model_sha256"] = code_digest.hexdigest()
            config["script_sha256"] = hashlib.sha256(stage.script.encode()).hexdigest()
            from ..codegen import has_moe

            config["has_moe"] = has_moe(project["graph"])
            if config["recipe"] != "custom":
                if (
                    len(report["outputs"]) != 1
                    or len(report["outputs"][0]["shape"]) != 3
                    or report["outputs"][0]["dtype"] != "float32"
                ):
                    raise ValueError(
                        "The text recipes require one floating-point [batch, sequence, vocabulary] output. Use a custom script for this graph."
                    )
                inputs = [
                    n for n in project["graph"]["nodes"] if n["data"]["kind"] == "input"
                ]
                if len(inputs) != 1:
                    raise ValueError(
                        "The text recipes require one token-ID input. Use a custom script for encoder-decoder or multiple-input graphs."
                    )
                config["vocabulary_size"] = report["outputs"][0]["shape"][-1]
                input_contract = inputs[0]["data"].get("params", {})
                if (
                    input_contract.get("dtype", "int64") != "int64"
                    or len(input_contract.get("shape", ["B", "T"])) != 2
                ):
                    raise ValueError(
                        "The text recipes require an int64 [batch, sequence] input."
                    )
                from ..core import params

                def check_causal(graph, scope):
                    scope = {**scope, **graph.get("variables", {})}
                    for item in graph["nodes"]:
                        p = params(item, scope)
                        if item["data"]["kind"] in ("attention", "sdpa") and not p.get(
                            "causal", True
                        ):
                            raise ValueError(
                                "Autoregressive text training requires causal attention; bidirectional attention can leak future target tokens. Use a custom objective for diffusion or masked prediction."
                            )
                        if item["data"].get("body"):
                            check_causal(item["data"]["body"], {**scope, **p})

                check_causal(project["graph"], project["config"])
            if config["strategy"] == "split" and config["precision"] != "fp32":
                raise ValueError("The split-model template currently requires FP32.")
            config["prepared_sha256"] = None
            datasets = {}
            for key in (() if code_first else ("dataset", "eval_dataset")):
                value = config[key]
                if value:
                    path = Path(value).expanduser().resolve()
                    if not path.is_file():
                        raise ValueError(
                            f"{key}: file does not exist on the Weave host: {path}"
                        )
                    config[key] = str(path)
                    datasets[key] = {
                        "path": str(path),
                        "sha256": file_hash(path),
                        "bytes": path.stat().st_size,
                    }
            if (
                not code_first

                and stage.config.recipe != "custom"
                and not config["dataset"]
            ):
                raise ValueError("Select a training dataset file.")
            config["dataset_sha256"] = datasets.get("dataset", {}).get("sha256")
            config["eval_dataset_sha256"] = datasets.get("eval_dataset", {}).get(
                "sha256"
            )
            if (
                not code_first
                and config["strategy"] == "split"
                and set(config["placement"]) != set(metadata["modules"])
            ):
                raise ValueError(
                    "Assign every top-level module in split placement: "
                    + ", ".join(metadata["modules"])
                )
            parent = None
            if config["start_mode"] != "random":
                parent_directory = self.run_dir(config["parent_run"])
                checkpoint = inside(parent_directory, config["parent_checkpoint"])
                published = next(
                    (
                        e
                        for e in service_module.events(parent_directory)
                        if e["kind"] == "checkpoint"
                        and e["payload"]["path"] == config["parent_checkpoint"]
                    ),
                    None,
                )
                if (
                    not published
                    or published["payload"].get("remote")
                    or not checkpoint.is_file()
                    or file_hash(checkpoint) != published["payload"]["sha256"]
                ):
                    raise ValueError(
                        "Parent checkpoint is missing, unpublished, or has changed."
                    )
                (directory / "starting-weights").mkdir()
                target = directory / "starting-weights" / "parent.pt"
                shutil.copyfile(checkpoint, target)
                config["checkpoint"] = str(target)
                parent = {
                    "run_id": config["parent_run"],
                    "checkpoint": config["parent_checkpoint"],
                    "sha256": file_hash(target),
                }
                provenance = directory / "lineage" / config["parent_run"]
                provenance.mkdir(parents=True)
                for artifact in parent_directory.iterdir():
                    if (
                        artifact.is_file()
                        and artifact.name != "events.jsonl"
                        and artifact.suffix in (".py", ".json", ".md", ".txt", ".log")
                        and not artifact.name.startswith("prepared-")
                        and artifact.name != "reporting.json"
                    ):
                        shutil.copyfile(artifact, provenance / artifact.name)
                atomic_json(provenance / "selected-checkpoint.json", published)
                (provenance / "events.jsonl").write_text(
                    "\n".join(
                        json.dumps(event)
                        for event in service_module.events(parent_directory)
                        if event.get("step", 0) <= published["step"]
                    )
                    + "\n"
                )
                for trace in parent_directory.glob("consumed-rank-*.jsonl"):
                    with (
                        trace.open() as source_trace,
                        (provenance / trace.name).open("w") as target_trace,
                    ):
                        for line in source_trace:
                            try:
                                if json.loads(line).get("step", 0) <= published["step"]:
                                    target_trace.write(line)
                            except ValueError:
                                continue
                for folder in ("tokenizers", "licenses", PACKAGE):
                    if (parent_directory / folder).exists():
                        shutil.copytree(
                            parent_directory / folder,
                            provenance / folder,
                            dirs_exist_ok=True,
                        )
                if (parent_directory / "lineage").exists():
                    shutil.copytree(
                        parent_directory / "lineage",
                        directory / "lineage",
                        dirs_exist_ok=True,
                    )
                lineage = json.loads((parent_directory / "run.json").read_text())
                atomic_json(
                    directory / "ancestor.json",
                    {
                        "run": lineage,
                        "dataset_manifest": json.loads(
                            (parent_directory / "datasets.json").read_text()
                        ),
                        "ancestors": (
                            json.loads((parent_directory / "ancestor.json").read_text())
                            if (parent_directory / "ancestor.json").exists()
                            else None
                        ),
                    },
                )
            config.setdefault("checkpoint", None)
            from .gpu_tuning import matching_profile

            profile = matching_profile(self.root, project, stage.config)
            if profile:
                config["gpu_tuning"] = profile
            if (
                code_first
                and config["compute_mode"] == "override"
                and config["strategy"] == "split"
            ):
                if not config["placement"] and not profile:
                    raise ValueError(
                        "Split training needs a matching saved FP32 tuning profile "
                        "or a GPU assignment for every top-level block."
                    )
                if config["placement"] and (
                    set(config["placement"]) != set(metadata["modules"])
                    or any(
                        device not in config["devices"]
                        for device in config["placement"].values()
                    )
                ):
                    raise ValueError(
                        "Assign every top-level block to a selected training GPU, "
                        "or clear all assignments to use saved tuning."
                    )
            (directory / "train.py").write_text(stage.script)
            if frozen_source is not None:
                from .run_controls import code_hash

                atomic_json(
                    directory / PACKAGE / "resume-compatibility.json",
                    {
                        "previous_code_sha256": code_hash(frozen_source),
                        "current_code_sha256": code_hash(directory),
                        "edited_script": bool(
                            restart_record
                            and restart_record["script_source"] == "edited"
                        ),
                        "reason": (
                            "User-requested script/support refresh; restore only compatible checkpoint state."
                            if restart_record
                            and (
                                restart_record["script_source"] == "edited"
                                or restart_record.get("refresh_support")
                            )
                            else "Verified trainer upgrade; frozen model, recipe and user helpers preserved."
                        ),
                    },
                )
            if validation_settings is not None:
                atomic_json(
                    directory / PACKAGE / "validation-initial.json", validation_settings
                )

            atomic_json(directory / "config.json", config)
            if code_first:
                atomic_json(directory / PACKAGE / "launch.json", config)
            atomic_json(directory / "stage.json", stage.model_dump())
            atomic_json(directory / "datasets.json", datasets)
            if restart_record is not None:
                from .run_controls import code_hash

                artifacts = [
                    "train.py",
                    "stage.json",
                    "config.json",
                    ".workspace-files.json",
                ]
                artifacts += [
                    p.relative_to(directory).as_posix()
                    for p in (directory / PACKAGE).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                ]
                artifacts += extras
                revision = {
                    **restart_record,
                    "run_id": run_id,
                    "created": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": parent,
                    "code_sha256": code_hash(directory),
                    "validation": validation_settings,
                    "files": {
                        name: file_hash(inside(directory, name))
                        for name in sorted(set(artifacts))
                    },
                    "state_policy": "Restore optimizer, scheduler, scaler, RNG and data cursor; reject incompatible state.",
                }
                atomic_json(directory / "revision.json", revision)
            warnings = instrumentation_warnings(stage.script)
            info = {
                "id": run_id,
                "project_id": project["id"],
                "stage_id": stage.id,
                "name": stage.name,
                "created": datetime.now(timezone.utc).isoformat(),
                "parent": parent,
                "recipe": config["recipe"],
                "strategy": config["strategy"],
                "execution": config["execution"],
                "code_first": code_first,
                "warnings": warnings,
                "model": metadata,
                "revision": restart_record,
            }
            atomic_json(directory / "run.json", info)
            if config["execution"] == "local":
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "backend.training.supervisor",
                        str(directory),
                        config["python"],
                    ],
                    cwd=HERE.parent.parent,
                    stdout=(directory / "supervisor.log").open("a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                info["supervisor_pid"] = process.pid
                atomic_json(directory / "run.json", info)
            return info
        except BaseException:
            shutil.rmtree(directory)
            raise
