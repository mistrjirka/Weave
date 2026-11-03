"""Optional PyTorch plumbing. The user owns the objective, data and training loop.

Nothing imports PyTorch, initializes DDP, seeds RNGs or moves a model until the
corresponding helper is explicitly called.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from datetime import timedelta
import uuid

from .weave_training import atomic_checkpoint_path, atomic_json

LAUNCH_FIELDS = {
    "devices",
    "strategy",
    "precision",
    "placement",
    "compile_mode",
    "fused_optimizer",
    "profile_steps",
    "evaluation_mode",
    "evaluation_device",
}


def launch_settings(defaults=None):
    """Read a frozen GUI launch, or use code defaults for an independent script.

    Reading settings does not apply them. Declare WEAVE_SUPPORTED_SETTINGS in
    train.py only for the fields your loop actually consumes.
    """
    values = dict(defaults or {})
    directory = os.environ.get("WEAVE_RUN_DIR")
    if directory:
        path = Path(directory) / "config.json"
        if path.is_file():
            values.update(json.loads(path.read_text()))
    return values


def seed_everything(seed, *, rank_offset=0):
    import torch

    seed = int(seed) + int(rank_offset)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2**32))


def _rng_state():
    import torch

    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        kind, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = [kind, keys.tolist(), pos, has_gauss, cached]
    return state


def _restore_rng(state):
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "numpy" in state:
        import numpy as np

        kind, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state(
            (kind, np.asarray(keys, dtype="uint32"), pos, has_gauss, cached)
        )


def _unwrapped(component):
    import torch

    if isinstance(component, torch.nn.parallel.DistributedDataParallel):
        component = component.module
    return getattr(component, "_orig_mod", component)


def _safe_state(value):
    """Keep the checkpoint readable with torch.load(weights_only=True)."""
    import torch

    if value is None or isinstance(value, (str, bool, int, float, torch.Tensor)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ValueError("Checkpoint dictionary keys must be primitive values")
            _safe_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _safe_state(item)
    else:
        raise ValueError(
            f"Unsupported checkpoint value: {type(value).__name__}. Save state_dicts, tensors and primitive containers."
        )


class Checkpointer:
    """Named state_dict components, atomic saves, and explicit resume claims.

    Include 'model' for checkpoint chat. Extra policy/value/optimizer/scheduler,
    scaler, sampler or environment components may have any other names.
    All DDP ranks must call save/load at the same safe boundary.
    """

    def __init__(self, run, *, components, rank_local=None, exact_resume=False):
        if not components or any(
            not isinstance(key, str) or not key for key in components
        ):
            raise ValueError("Name each checkpoint component")
        self.run, self.components = run, dict(components)
        self.rank_local = dict(rank_local or {})
        if any(not isinstance(key, str) or not key for key in self.rank_local):
            raise ValueError("Name each rank-local checkpoint component")
        if type(exact_resume) is not bool:
            raise ValueError("exact_resume must be an explicit boolean")
        self.exact_resume = exact_resume

    def save(self, path=None, *, extra=None, snapshot_requests=None):
        import torch
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        if not distributed and self.run.rank != 0:
            return None
        world = dist.get_world_size() if distributed else 1
        if distributed and dist.get_rank() != self.run.rank:
            raise ValueError("Use Run(rank=context.rank) with distributed_context")
        rng = [None] * world
        try:
            local = {
                "rng": _rng_state(),
                "components": {
                    key: value.state_dict() for key, value in self.rank_local.items()
                },
            }
            _safe_state(local)
        except Exception as exc:
            local = {"error": str(exc)}
        if distributed:
            dist.all_gather_object(rng, local)
        else:
            rng[0] = local
        if any("error" in item for item in rng):
            raise ValueError(
                "Cannot capture rank-local checkpoint state: "
                + str(next(item["error"] for item in rng if "error" in item))
            )
        path = (
            Path(path)
            if path is not None
            else self.run.directory
            / "checkpoints"
            / f"step-{self.run.step}-{uuid.uuid4().hex[:12]}.pt"
        )
        if not path.is_absolute():
            path = self.run.directory / path
        path = path.resolve()
        path.relative_to(self.run.directory)
        selected_path = [str(path) if self.run.rank == 0 else None]
        if distributed:
            dist.broadcast_object_list(selected_path, src=0)
        path = Path(selected_path[0])
        outcome = [None]
        if self.run.rank == 0:
            try:
                if path.exists():
                    raise ValueError(
                        "Checkpoint paths are immutable; choose a new filename"
                    )
                states = {
                    key: _unwrapped(value).state_dict()
                    for key, value in self.components.items()
                }
                payload = {
                    "weave_checkpoint_version": 1,
                    "components": states,
                    "rank_states": rng,
                    "world_size": world,
                    "step": self.run.step,
                    "tokens": self.run.tokens,
                    "counters": dict(self.run.counters),
                    "extra": extra or {},
                }
                if "model" in states:
                    payload["model"] = states["model"]
                _safe_state(payload)
                with atomic_checkpoint_path(path) as temporary:
                    torch.save(payload, temporary)
                self.run.checkpoint(
                    path,
                    metadata={
                        "exact_resume": self.exact_resume,
                        "components": list(states),
                        "snapshot_requests": snapshot_requests or [],
                    },
                )
            except Exception as exc:
                outcome[0] = str(exc)
        if distributed:
            dist.broadcast_object_list(outcome, src=0)
        if outcome[0] is not None:
            raise RuntimeError(f"Checkpoint save failed: {outcome[0]}")
        return path

    def load(self, path, *, weights_only=False, restore_rng=True):
        try:
            return self._load(path, weights_only=weights_only, restore_rng=restore_rng)
        except Exception as exc:
            problem = {
                "message": str(exc),
                "checkpoint": str(path),
                "recovery": "Fix the script and retry, or resume with the previous script.",
            }
            if self.run.rank == 0:
                atomic_json(self.run.directory / "resume-problem.json", problem)
            self.run.emit("resume_problem", problem)
            raise

    def _load(self, path, *, weights_only=False, restore_rng=True):
        import torch
        import torch.distributed as dist

        state = torch.load(path, map_location="cpu", weights_only=True)
        if weights_only:
            if "model" not in self.components:
                raise ValueError("Weights-only restore needs a model component")
            _unwrapped(self.components["model"]).load_state_dict(
                state.get("model", state), strict=True
            )
            return {}
        if state.get("weave_checkpoint_version") != 1:
            raise ValueError(
                "Use weights_only=True for a checkpoint from another trainer"
            )
        states = state["components"]
        if set(states) != set(self.components):
            raise ValueError(
                "Checkpoint components differ: restore the previous script or provide matching named components"
            )
        world = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if restore_rng and state["world_size"] != world:
            raise ValueError(
                "Restoring per-rank RNG state requires the original world size"
            )
        if state["world_size"] != world and self.rank_local:
            raise ValueError(
                "Restoring rank-local components requires the original world size"
            )
        local = state["rank_states"][rank if state["world_size"] == world else 0]
        if set(local["components"]) != set(self.rank_local):
            raise ValueError("Checkpoint rank-local components differ")
        # Validate module tensor shapes before mutating any component.
        for key, component in self.components.items():
            module = _unwrapped(component)
            if isinstance(module, torch.nn.Module):
                current = module.state_dict()
                if set(current) != set(states[key]) or any(
                    current[name].shape != states[key][name].shape for name in current
                ):
                    raise ValueError(
                        f"Incompatible tensor shapes in {key}; restore the previous script or matching architecture"
                    )
        for key, component in self.components.items():
            _unwrapped(component).load_state_dict(states[key])
        for key, component in self.rank_local.items():
            component.load_state_dict(local["components"][key])
        if restore_rng:
            _restore_rng(local["rng"])
        self.run.step, self.run.tokens = state["step"], state["tokens"]
        self.run.counters.clear()
        self.run.set_counters(**state.get("counters", {}))
        return state.get("extra", {})


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: object

    def wrap(self, model):
        from torch.nn.parallel import DistributedDataParallel

        model = model.to(self.device)
        if self.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
        )

    def poll_commands(self, run):
        import torch.distributed as dist

        commands = [run.poll_commands() if self.rank == 0 else None]
        if self.world_size > 1:
            dist.broadcast_object_list(commands, src=0)
        return commands[0]


@contextmanager
def distributed_context(devices=(), *, timeout_seconds=600):
    """Explicit torchrun lifecycle; never spawns workers or changes seeds.

    Standalone Python accepts one device. GUI DDP/torchrun supplies RANK,
    LOCAL_RANK and WORLD_SIZE. An existing process group is never destroyed.
    """
    import torch
    import torch.distributed as dist

    if len(set(devices)) != len(devices) or any(
        type(i) is not int or i < 0 for i in devices
    ):
        raise ValueError("Choose unique nonnegative GPU indices")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(
        os.environ.get("RANK", "0")
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if devices and (
        not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count()
    ):
        raise ValueError("A selected GPU is unavailable")
    if devices and local_rank >= len(devices):
        raise ValueError("Select one GPU per local torchrun rank")
    if world == 1 and len(devices) > 1:
        raise ValueError("Launch with torchrun to train on multiple GPUs")
    device = (
        torch.device("cuda", devices[local_rank]) if devices else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    owned = world > 1 and not dist.is_initialized()
    if owned:
        dist.init_process_group(
            backend="nccl" if devices else "gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )
    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
    try:
        yield DistributedContext(rank, world, device)
    finally:
        if owned:
            dist.destroy_process_group()
