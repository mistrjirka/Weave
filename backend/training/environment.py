"""Inspect a selected Python executable with fixed code and no shell."""

import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

PROBE = """import sys,json,platform,importlib.util
result={"executable":sys.executable,"python_version":platform.python_version(),"torch_version":None,"cuda_version":None,"cuda_available":False,"device_count":0}
if importlib.util.find_spec("torch"):
    import torch
    result.update(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count())
print(json.dumps(result))
"""


def inspect_python(executable):
    # Interpreter/device checks import PyTorch in another process. Keep them off
    # the dashboard's critical path and reuse recent checks across browser tabs.
    try:
        info = Path(executable).expanduser().stat()
        signature = (info.st_mtime_ns, info.st_size)
    except OSError:
        signature = None
    return dict(_inspect_cached(executable, signature, int(time.monotonic() // 60)))


@lru_cache(maxsize=32)
def _inspect_cached(executable, signature, minute):
    try:
        result = subprocess.run(
            [executable, "-c", PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Could not inspect this Python executable: {exc}") from exc
