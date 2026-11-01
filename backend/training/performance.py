"""Opt-in execution optimizations; model weights and checkpoint names stay intact."""

from contextlib import contextmanager
from pathlib import Path

import torch


def configure_execution(model, optimizer, config):
    """Apply after checkpoint restoration, which restores optimizer group flags."""
    if config.fused_optimizer:
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError(
                "Fused optimizer override requires torch.optim.AdamW; keep script settings for a custom optimizer"
            )
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if not parameters or any(p.device.type != "cuda" for p in parameters):
            raise ValueError("Fused AdamW override requires CUDA parameters")
        if optimizer.defaults.get("differentiable"):
            raise ValueError("Fused AdamW does not support differentiable optimizers")
        optimizer.defaults.update(fused=True, foreach=False)
        for group in optimizer.param_groups:
            group.update(fused=True, foreach=False)
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if "step" in state:
                    state["step"] = state["step"].to(parameter.device)
        optimizer._step_supports_amp_scaling = True
    if config.compile_mode != "off":
        if config.resolved_strategy == "split":
            raise ValueError(
                "Compilation is not supported with split placement; use eager execution"
            )
        if any(parameter.device.type == "cuda" for parameter in model.parameters()):
            try:
                import triton  # noqa: F401 — verify import, not just package presence.
            except ImportError as exc:
                raise ValueError(
                    "CUDA compilation needs a working Triton installation. Install weave_support/requirements.txt and the Triton version required by your PyTorch build (older Triton also needs setuptools)."
                ) from exc
        # Module.compile changes __call__, preserving parameter identities,
        # state_dict keys and user access to named model components.
        model.compile(mode=config.compile_mode, dynamic=False)


@contextmanager
def training_profile(directory, config, rank, run):
    """One bounded trace per worker; no profiler is active unless requested."""
    if not config.profile_steps:
        yield None
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if config.devices:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    target = Path(directory) / "profiles" / f"rank-{rank}.json"
    target.parent.mkdir(exist_ok=True)

    def ready(profiler):
        profiler.export_chrome_trace(str(target))
        if rank == 0:
            run.emit(
                "profile",
                {
                    "path": str(target.relative_to(directory)),
                    "steps": config.profile_steps,
                    "message": "Open this trace in Perfetto or chrome://tracing. Profiling adds overhead during capture.",
                },
            )

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=config.profile_steps, repeat=1
        ),
        on_trace_ready=ready,
        record_shapes=True,
    ) as profiler:
        yield profiler
