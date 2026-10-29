"""Inspect a selected Python executable with fixed code and no shell."""

import json
import subprocess

PROBE = """import sys,json,platform,importlib.util
result={"executable":sys.executable,"python_version":platform.python_version(),"torch_version":None,"cuda_version":None,"cuda_available":False,"device_count":0}
if importlib.util.find_spec("torch"):
    import torch
    result.update(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count())
print(json.dumps(result))
"""


def inspect_python(executable):
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
