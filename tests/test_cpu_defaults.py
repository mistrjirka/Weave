"""Machine-sized CPU defaults must survive export and preserve explicit limits."""

import json
import os
import subprocess
import sys

from backend.training.weave_training import available_cpu_threads
from backend.training.service import TrainingConfig
from test_training_hooks import make_hook_run


def test_cpu_default_respects_process_affinity_and_explicit_limit(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {1, 3, 5}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert available_cpu_threads() == 3
    assert TrainingConfig().cpu_threads == 3
    assert TrainingConfig(cpu_threads=1).cpu_threads == 1


def test_cpu_default_portable_fallbacks(monkeypatch):
    def unsupported(_pid):
        raise OSError("Affinity unavailable")

    monkeypatch.setattr(os, "sched_getaffinity", unsupported, raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)
    assert available_cpu_threads() == 6
    monkeypatch.delattr(os, "process_cpu_count")
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    assert available_cpu_threads() == 12
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert available_cpu_threads() == 1


def test_exported_trainer_resolves_defaults_on_execution_machine(tmp_path):
    _, directory, _ = make_hook_run(tmp_path)
    probe = """import json
from weave_support.weave_training import available_cpu_threads
from weave_support.trainer import TrainingConfig
print(json.dumps({"available": available_cpu_threads(), "default": TrainingConfig().cpu_threads, "explicit": TrainingConfig(cpu_threads=1).cpu_threads}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=directory,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    config = json.loads(result.stdout)
    assert config["default"] == config["available"] == available_cpu_threads()
    assert config["explicit"] == 1
