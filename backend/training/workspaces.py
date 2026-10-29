"""Editable stage files shared by the browser and SSH editors.

Hashes provide optimistic concurrency: an SSH save cannot be silently replaced
by a stale browser draft. Run directories remain separate immutable snapshots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import RLock

from .service import Stage, Plan, TrainingConfig, valid_id, inside
from .weave_training import atomic_json, file_hash


class WorkspaceConflict(ValueError):
    pass


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class WorkspaceStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.lock = RLock()

    def directory(self, project_id, stage_id):
        destination = self.project_directory(project_id) / "stages" / valid_id(stage_id)
        legacy = self.root / "workspaces" / valid_id(project_id) / valid_id(stage_id)
        if legacy.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy), str(destination))
        return destination

    def project_directory(self, project_id):
        return self.root.parent / "projects" / valid_id(project_id)

    def plan_path(self, project_id):
        return self.root / "plans" / (valid_id(project_id) + ".json")

    def metadata_plan(self, project_id):
        path = self.plan_path(project_id)
        return (
            Plan.model_validate_json(path.read_text())
            if path.exists()
            else Plan(project_id=project_id)
        )

    def read(self, project_id, stage):
        directory = self.directory(project_id, stage.id)
        script = directory / "train.py"
        config_path = directory / "config.json"
        result = stage.model_copy(deep=True)
        result.script_path = str(script)
        result.config_path = str(config_path)
        if script.exists():
            if script.stat().st_size > 1024 * 1024:
                raise ValueError("Training script exceeds the 1 MB editor limit")
            result.script = script.read_text()
            result.script_revision = file_hash(script)
        if config_path.exists():
            try:
                result.config = TrainingConfig.model_validate_json(
                    config_path.read_text()
                )
                result.workspace_warning = None
            except ValueError:
                result.workspace_warning = "config.json contains invalid settings. Fix it in your SSH editor before launching."
            result.config_revision = file_hash(config_path)
        return result

    def check(self, project_id, stage):
        directory = self.directory(project_id, stage.id)
        script, config = directory / "train.py", directory / "config.json"
        if (
            script.exists()
            and script.read_text() != stage.script
            and stage.script_revision != file_hash(script)
        ):
            raise WorkspaceConflict(
                "train.py changed on the server. Reload the server file, or save your browser draft as another stage."
            )
        if config.exists():
            try:
                same = (
                    TrainingConfig.model_validate_json(config.read_text())
                    == stage.config
                )
            except ValueError:
                same = False
            if not same and stage.config_revision != file_hash(config):
                raise WorkspaceConflict(
                    "config.json changed on the server. Reload the server files before saving these settings."
                )

    def write(self, project_id, stage):
        self.check(project_id, stage)
        directory = self.directory(project_id, stage.id)
        directory.mkdir(parents=True, exist_ok=True)
        script, config = directory / "train.py", directory / "config.json"
        if not script.exists() or script.read_text() != stage.script:
            atomic_text(script, stage.script)
        try:
            same_config = (
                config.exists()
                and TrainingConfig.model_validate_json(config.read_text())
                == stage.config
            )
        except ValueError:
            same_config = False
        if not same_config:
            atomic_json(config, stage.config.model_dump())
        return self.read(project_id, stage)

    def load_plan(self, project_id):
        with self.lock:
            plan = self.metadata_plan(project_id)
            # Migrate legacy inline scripts only when no editable server file exists.
            plan.stages = [
                (
                    self.read(project_id, stage)
                    if (self.directory(project_id, stage.id) / "train.py").exists()
                    else self.write(project_id, stage)
                )
                for stage in plan.stages
            ]
            return plan

    def save_plan(self, plan):
        with self.lock:
            for stage in plan.stages:
                self.check(plan.project_id, stage)
            result = plan.model_copy(deep=True)
            result.stages = [
                self.write(plan.project_id, stage) for stage in plan.stages
            ]
            atomic_json(self.plan_path(plan.project_id), result.model_dump())
            return result

    def save_stage(self, project_id, stage):
        with self.lock:
            saved = self.write(project_id, stage)
            plan = self.metadata_plan(project_id)
            index = next(
                (i for i, item in enumerate(plan.stages) if item.id == stage.id), None
            )
            if index is None:
                plan.stages.append(saved)
            else:
                plan.stages[index] = saved
            atomic_json(self.plan_path(project_id), plan.model_dump())
            return saved

    def get_stage(self, project_id, stage_id):
        plan = self.metadata_plan(project_id)
        stage = next((item for item in plan.stages if item.id == stage_id), None)
        if stage is None:
            raise FileNotFoundError(
                "Training stage not found. Save the stage to create its server files."
            )
        return self.read(project_id, stage)

    def prepare_model(self, project, stage, tokenizer_root):
        """Refresh graph-owned files, preserving the editable training script."""
        import io
        import zipfile
        from ..compiler import export_zip
        from ..tokenizer_bindings import input_tokenizers

        project_directory = self.project_directory(project["id"])
        atomic_json(project_directory / "project.json", project)
        directory = self.directory(project["id"], stage.id)
        directory.mkdir(parents=True, exist_ok=True)
        directories = {
            info["id"]: Path(tokenizer_root) / valid_id(info["id"])
            for info in input_tokenizers(project).values()
        }
        bundle = export_zip(project, tokenizer_dirs=directories)
        generated = []
        atomic_text(
            project_directory / "README.md",
            "# Weave project files\n\nproject.json records the saved graph. stages/<stage-id>/ contains model.py, train.py, config.json and the portable reporting helper. Edit train.py and helper files in VS Code; model.py is generated from the graph and refreshed when saving a stage. Training runs are frozen separately in training/runs/.\n",
        )
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            for name in archive.namelist():
                if name in ("train.py", "config.json"):
                    continue
                target = inside(directory, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
                generated.append(name)
        shutil.copyfile(
            Path(__file__).with_name("weave_training.py"),
            directory / "weave_training.py",
        )
        generated.append("weave_training.py")
        atomic_json(directory / ".weave-generated.json", {"files": generated})

    def freeze_extras(self, project_id, stage_id, destination):
        directory = self.directory(project_id, stage_id)
        manifest = directory / ".weave-generated.json"
        generated = (
            set(json.loads(manifest.read_text())["files"])
            if manifest.exists()
            else set()
        )
        for source in directory.rglob("*"):
            name = source.relative_to(directory).as_posix()
            if (
                not source.is_file()
                or source.is_symlink()
                or source.name.startswith(".")
                or "__pycache__" in source.parts
                or name in generated
                or name in ("train.py", "config.json")
            ):
                continue
            if source.suffix not in (".py", ".json", ".yaml", ".yml", ".toml", ".txt"):
                continue
            target = inside(destination, name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
