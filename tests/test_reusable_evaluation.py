"""Reusable evaluators must load fresh weights and preserve validation sampling."""

import json
import os
import subprocess
import sys
import textwrap

from backend.training.service import events
from test_training_hooks import filled_starter, make_hook_run


def validation_recipe(tmp_path, *, one_shot=False):
    source = filled_starter(tmp_path / "rows.jsonl")
    source = "import numpy as np\nnp.random.seed(83)\n" + source
    source = source.replace(
        "import torch",
        "import torch\ndef forbidden_cuda_rng(*args, **kwargs):\n    raise AssertionError('CPU evaluation touched CUDA RNG')\ntorch.cuda.get_rng_state_all = forbidden_cuda_rng\ntorch.cuda.get_rng_state = forbidden_cuda_rng",
        1,
    )
    source = source.replace(
        "return StepResult(loss, weight=count)",
        "return StepResult(loss + float(np.random.random()), weight=count)",
    )
    validation = """rows = [
        {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([2, 3, 4, 5])},
        {"input_ids": torch.tensor([5, 4, 3, 2]), "labels": torch.tensor([4, 3, 2, 1])},
    ]
    validation = DataLoader(rows, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(73))
    return DataLoader(dataset, batch_size=config.micro_batch_size, num_workers=0), validation"""
    if one_shot:
        validation = validation.replace(
            "validation = DataLoader(rows, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(73))",
            "validation = iter(DataLoader(rows, batch_size=1))",
        )
    return source.replace(
        "return DataLoader(dataset, batch_size=config.micro_batch_size, num_workers=0), None",
        validation,
    )


def run_probe(directory, script):
    path = directory / "probe.py"
    path.write_text(textwrap.dedent(script))
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=directory,
        env={**os.environ, "WEAVE_RUN_DIR": str(directory), "USE_TORCH": "1"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_warm_worker_uses_latest_weights_resets_sampler_and_reports_failures(tmp_path):
    _, directory, _ = make_hook_run(tmp_path, script=validation_recipe(tmp_path))
    output = run_probe(
        directory,
        """
        import json, os, subprocess, sys
        from pathlib import Path
        import torch
        from weave_support.model import Model
        from weave_support.evaluation_worker import EvaluationSession
        torch.set_num_threads(2)
        root = Path.cwd()
        checkpoints = []
        for seed, step in ((17, 10), (37, 20)):
            torch.manual_seed(seed)
            path = root / f"weights-{step}.pt"
            torch.save({"model": Model().state_dict(), "step": step, "tokens": step * 10}, path)
            checkpoints.append(path)
        bad = root / "weights-bad.pt"
        torch.save({"model": {}, "step": 30, "tokens": 300}, bad)
        session = EvaluationSession(root / "evaluation-workers" / "test", "cpu")
        settings = {"mode": "async", "device": "cpu", "eval_every": 10, "eval_batches": 1}
        results = []
        try:
            for path in (*checkpoints, checkpoints[0]):
                session.submit(path, settings)
                if session.process is None:
                    with (root / "worker.log").open("w") as log:
                        session.process = subprocess.Popen([sys.executable, "train.py", "--evaluation-worker", str(session.directory)], stdout=log, stderr=log)
                assert session.wait(timeout=45) == 0, session.result
                results.append(session.result)
            # Settings are per job, not captured permanently by the first pass.
            session.submit(checkpoints[0], {**settings, "eval_batches": 2})
            assert session.wait(timeout=45) == 0
            results.append(session.result)
            session.submit(bad, settings)
            assert session.wait(timeout=45) == 1
            assert session.result["success"] is False
            assert "Missing key" in session.result["message"]
        finally:
            session.close()
        # Fresh process serves as an independent reference for the warm loader.
        env = dict(os.environ, WEAVE_VALIDATION_SETTINGS=json.dumps(settings))
        fresh = subprocess.run([sys.executable, "train.py", "--evaluate", str(checkpoints[0])], env=env, capture_output=True, text=True, timeout=45)
        assert fresh.returncode == 0, fresh.stdout + fresh.stderr
        print(json.dumps(results))
    """,
    )
    assert len({result["pid"] for result in output}) == 1
    assert [result["reused"] for result in output] == [False, True, True, True]
    assert [result["step"] for result in output] == [10, 20, 10, 10]
    metrics = [
        e for e in events(directory) if e["kind"] == "metrics" and e["step"] == 10
    ]
    losses = [e["payload"]["eval/loss"] for e in metrics]
    assert len(losses) == 4
    # Initial, repeated, and independent fresh-process evaluation use the same sample.
    assert losses[0] == losses[1] == losses[-1]
    errors = events(directory, kind="error")
    assert any("Missing key" in e["payload"]["message"] for e in errors)
    configs = [
        e["payload"]["eval_batches"] for e in events(directory, kind="configuration")
    ]
    assert 2 in configs


def test_nonrepeatable_loader_disables_reuse_and_worker_exits(tmp_path):
    _, directory, _ = make_hook_run(
        tmp_path, script=validation_recipe(tmp_path, one_shot=True)
    )
    result = run_probe(
        directory,
        """
        import json, subprocess, sys
        from pathlib import Path
        import torch
        from weave_support.model import Model
        from weave_support.evaluation_worker import EvaluationSession
        root = Path.cwd()
        path = root / "weights.pt"
        torch.save({"model": Model().state_dict(), "step": 10, "tokens": 100}, path)
        session = EvaluationSession(root / "evaluation-workers" / "test", "cpu")
        session.submit(path, {"mode": "async", "device": "cpu", "eval_every": 1, "eval_batches": 1})
        try:
            with (root / "worker.log").open("w") as log:
                session.process = subprocess.Popen([sys.executable, "train.py", "--evaluation-worker", str(session.directory)], stdout=log, stderr=log)
            assert session.wait(timeout=45) == 0
            assert session.process.wait(timeout=10) == 0
            print(json.dumps(session.result))
        finally:
            session.close()
    """,
    )
    assert result["reusable"] is False
    assert result["step"] == 10


def test_temporary_snapshots_use_ram_and_fall_back_without_removing_active_weights(
    tmp_path,
):
    _, directory, _ = make_hook_run(tmp_path)
    result = run_probe(
        directory,
        """
        import errno, json
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        import weave_support.evaluation_worker as ipc
        import weave_support.trainer as module
        from weave_support.model import Model
        root = Path.cwd()
        trainer = module.Trainer(model=Model, config=module.TrainingConfig(), dataloaders_fn=None, training_step_fn=None)
        trainer.model = Model()
        trainer.run = SimpleNamespace(step=10, tokens=100, emit=lambda *args: None)
        # Request actual tmpfs only where available; portable fallback is tested below.
        first = Path(trainer._evaluation_snapshot(root))
        first = first if first.is_absolute() else root / first
        state = torch.load(first, weights_only=True)
        assert state["step"] == 10
        assert state["model"].keys() == trainer.model.state_dict().keys()
        used_memory = trainer.evaluation_memory is not None
        if used_memory:
            trainer.config.evaluation_snapshot_storage = "disk"
            forced = trainer._evaluation_snapshot(root)
            assert not Path(forced).is_absolute(), "Explicit disk setting must override a warm RAM directory"
            trainer._discard_evaluation_snapshot(root / forced)
            trainer.config.evaluation_snapshot_storage = "auto"
            real_save = torch.save
            def full_memory(value, path):
                if Path(path).is_relative_to(trainer.evaluation_memory.name):
                    raise OSError(errno.ENOSPC, "tmpfs became full")
                return real_save(value, path)
            torch.save = full_memory
            trainer.run.step = 20
            fallback = trainer._evaluation_snapshot(root)
            assert not Path(fallback).is_absolute()
            assert (root / fallback).is_file()
            assert first.is_file(), "An active worker may still be reading the prior snapshot"
            torch.save = real_save
            trainer._discard_evaluation_snapshot(root / fallback)
        trainer._discard_evaluation_snapshot(first)
        assert not first.exists()
        if trainer.evaluation_memory:
            trainer.evaluation_memory.cleanup()
        # Low free space must select disk before attempting a RAM write.
        ipc.shutil.disk_usage = lambda path: SimpleNamespace(free=1)
        other = module.Trainer(model=Model, config=module.TrainingConfig(), dataloaders_fn=None, training_step_fn=None)
        other.model = Model(); other.run = trainer.run
        path = other._evaluation_snapshot(root)
        assert not Path(path).is_absolute()
        assert other.evaluation_memory is None
        other._discard_evaluation_snapshot(root / path)
        print(json.dumps({"memory_tested": used_memory, "fallback": True}))
    """,
    )
    assert result["fallback"] is True
