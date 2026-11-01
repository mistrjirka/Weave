"""Long data hooks must not leave a training collective pending."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_preparation_can_exceed_training_collective_timeout(tmp_path, backend):
    python = (
        os.environ.get("WEAVE_CUDA_PYTHON") if backend == "nccl" else sys.executable
    )
    if not python:
        pytest.skip("Set WEAVE_CUDA_PYTHON for real two-GPU acceptance")
    script = tmp_path / "prepare.py"
    script.write_text("""import os,time
from datetime import timedelta
import torch
import torch.distributed as dist
from backend.training.preparation import distributed_preparation, preparation_barrier
rank=int(os.environ['RANK'])
backend=os.environ['TEST_BACKEND']
if backend=='nccl': torch.cuda.set_device(rank)
dist.init_process_group(backend,timeout=timedelta(seconds=3))
value=torch.tensor([rank+1.0],device=f'cuda:{rank}' if backend=='nccl' else 'cpu')
dist.all_reduce(value)
with distributed_preparation(30):
    if rank==0: time.sleep(4)
    preparation_barrier()
    if rank==1: time.sleep(4)
# The context also waits for a slower loader after the user's barrier.
dist.all_reduce(value)
assert value.item()==6
print('PREPARATION_OK',rank,flush=True)
dist.destroy_process_group()
""")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(script),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "USE_TORCH": "1",
            "TEST_BACKEND": backend,
            "OMP_NUM_THREADS": "1",
        },
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PREPARATION_OK") == 2
