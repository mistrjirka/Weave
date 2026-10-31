"""Inspect code-owned recipes without importing or executing user Python."""

import ast
from pathlib import Path

HERE = Path(__file__).parent


def is_code_first(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.ImportFrom) and node.module == "weave_support.trainer"
        for node in ast.walk(tree)
    )


SCRIPT_FIELDS = {
    "devices": "devices",
    "strategy": "strategy",
    "precision": "precision",
    "micro_batch_size": "batch_size",
    "gradient_accumulation_steps": "accumulation",
    "max_optimizer_steps": "max_steps",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "gradient_clip": "gradient_clip",
    "log_every": "log_every",
    "checkpoint_every": "checkpoint_every",
    "eval_batches": "eval_batches",
    "generation_tokens": "generation_tokens",
    "sample_prompts": "prompts",
    "seed": "seed",
    "cpu_threads": "cpu_threads",
    "evaluation_device": "evaluation_device",
    "placement": "placement",
}


def literal_settings(source):
    """Best-effort UI/launcher metadata; runtime Python remains authoritative."""
    tree = ast.parse(source)
    constructors = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "weave_support.trainer"
        for alias in node.names
        if alias.name == "TrainingConfig"
    }
    settings = {}
    for node in tree.body:
        value = getattr(node, "value", None)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in constructors
        ):
            for keyword in value.keywords:
                if keyword.arg not in SCRIPT_FIELDS:
                    continue
                try:
                    value = ast.literal_eval(keyword.value)
                    if value is not None:
                        settings[SCRIPT_FIELDS[keyword.arg]] = value
                except (ValueError, TypeError):
                    pass  # Dynamic Python is resolved only inside the training process.
    if settings.get("strategy", "auto") == "auto":
        settings["strategy"] = (
            "ddp" if len(settings.get("devices", [])) > 1 else "single"
        )
    return settings


def starter_script(recipe="causal"):
    source = (HERE / "hook_template.py").read_text()
    if recipe == "sft":
        source = source.replace(
            '"""Return (train_loader, optional_validation_loader)."""',
            '"""Return train/validation loaders; mask non-assistant labels with -100."""',
        )
    return source
