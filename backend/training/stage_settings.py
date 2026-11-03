"""Preview stage launch settings without importing or running user Python."""

import ast
from functools import lru_cache
from pathlib import Path

from .recipe import is_code_first

COMPUTE_FIELDS = ("devices", "strategy", "precision", "placement")
PERFORMANCE_FIELDS = ("compile_mode", "fused_optimizer", "profile_steps")
PREVIEW_FIELDS = (
    *COMPUTE_FIELDS,
    *PERFORMANCE_FIELDS,
    "async_evaluation",
    "evaluation_device",
)


@lru_cache(maxsize=1)
def trainer_defaults():
    """Read the shipped dataclass literals; importing it would load PyTorch."""
    tree = ast.parse(Path(__file__).with_name("trainer.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig"
    )
    result = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name, value = node.target.id, node.value
        if name not in PREVIEW_FIELDS:
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "field"
        ):
            factory = next(
                (
                    item.value
                    for item in value.keywords
                    if item.arg == "default_factory"
                ),
                None,
            )
            if isinstance(factory, ast.Name) and factory.id in ("list", "dict"):
                result[name] = [] if factory.id == "list" else {}
                continue
        result[name] = ast.literal_eval(value)
    return result


def preview_stage_settings(stage):
    config = stage.config.model_dump()
    try:
        tree = ast.parse(stage.script)
    except SyntaxError as exc:
        return {
            "config": config,
            "unresolved": list(PREVIEW_FIELDS),
            "error": f"Python line {exc.lineno}: {exc.msg}",
        }
    if not is_code_first(stage.script):
        config["evaluation_mode"] = (
            "async" if config["strategy"] == "train_eval" else "inline"
        )
        return {"config": config, "unresolved": [], "error": None}
    constructors = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "weave_support.trainer"
        for alias in node.names
        if alias.name == "TrainingConfig"
    }
    values = dict(trainer_defaults())
    unresolved, constants, configuration_names = set(), {}, set()
    constructors_found = 0
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = node.targets if isinstance(node, ast.Assign) else [node.target]
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                for name in names:
                    if isinstance(name, ast.Name):
                        constants.pop(name.id, None)
            else:
                for name in names:
                    if isinstance(name, ast.Name):
                        constants[name.id] = literal
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in constructors
        ):
            continue
        constructors_found += 1
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            configuration_names.update(
                name.id for name in names if isinstance(name, ast.Name)
            )
        for keyword in value.keywords:
            if keyword.arg is None:
                unresolved.update(PREVIEW_FIELDS)
            elif keyword.arg in PREVIEW_FIELDS:
                try:
                    resolved = (
                        constants[keyword.value.id]
                        if isinstance(keyword.value, ast.Name)
                        else ast.literal_eval(keyword.value)
                    )
                except (ValueError, TypeError, KeyError):
                    unresolved.add(keyword.arg)
                else:
                    values[keyword.arg] = resolved
                    unresolved.discard(keyword.arg)
    if constructors_found != 1:
        unresolved.update(PREVIEW_FIELDS)
    # A later mutation (including one inside a function/conditional) makes that
    # field runtime-dependent. Do not present the constructor's old value as fact.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in configuration_names
                    and target.attr in PREVIEW_FIELDS
                ):
                    unresolved.add(target.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id in configuration_names
                and owner.attr in PREVIEW_FIELDS
            ):
                unresolved.add(owner.attr)
    if values["strategy"] == "auto":
        if "devices" in unresolved:
            unresolved.add("strategy")
            values["strategy"] = "single"
        else:
            values["strategy"] = (
                "ddp"
                if isinstance(values["devices"], (list, tuple))
                and len(values["devices"]) > 1
                else "single"
            )
    if config["compute_mode"] != "override":
        config.update({key: values[key] for key in COMPUTE_FIELDS})
    else:
        unresolved.difference_update(COMPUTE_FIELDS)
    if config["performance_mode"] != "override":
        config.update({key: values[key] for key in PERFORMANCE_FIELDS})
    else:
        unresolved.difference_update(PERFORMANCE_FIELDS)
    if config["evaluation_mode"] == "script":
        config["evaluation_mode"] = (
            "async"
            if values["async_evaluation"] or config["strategy"] == "train_eval"
            else "inline"
        )
        config["evaluation_device"] = values["evaluation_device"]
    else:
        unresolved.difference_update(("async_evaluation", "evaluation_device"))
    if config["evaluation_mode"] == "inline" and config["strategy"] == "train_eval":
        config["strategy"] = "single"
    if config["evaluation_mode"] == "async" and config["evaluation_device"] is None:
        if "devices" in unresolved:
            unresolved.add("evaluation_device")
        else:
            devices = config["devices"]
            config["evaluation_device"] = (
                devices[0] if isinstance(devices, (list, tuple)) and devices else "cpu"
            )
    # Invalid literal values are reported as a preview error, not executed or silently coerced.
    try:
        type(stage.config).model_validate(config, strict=True)
        if any(index < 0 for index in config["devices"]) or len(
            set(config["devices"])
        ) != len(config["devices"]):
            raise ValueError("Invalid GPU indices")
    except ValueError:
        return {
            "config": stage.config.model_dump(),
            "unresolved": sorted(unresolved),
            "error": "The Python device settings contain invalid values. Check TrainingConfig in the script.",
        }
    return {"config": config, "unresolved": sorted(unresolved), "error": None}
