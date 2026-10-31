"""Isolated, bounded synthetic training benchmarks. Never imports user train.py."""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def measure(root, spec, session=False):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from weave_support.model import Model
    from weave_support.text_training import place_model
    from weave_support.runtime import moe_auxiliary_loss, SparseMoE

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = spec["strategy"] == "ddp"
    devices = spec["devices"]
    local = devices[rank] if ddp else devices[0]
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if ddp and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        model, device = place_model(
            Model(),
            {
                **spec,
                "strategy": "single" if ddp else spec["strategy"],
                "devices": [local] if ddp else devices,
            },
        )
        model.train()
        module_costs, handles, starts = {}, [], {}

        def before(name):
            def hook(module, args):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                starts[name] = event

            return hook

        def after(name):
            def hook(module, args, output):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                module_costs.setdefault(name, []).append((starts[name], event))

            return hook

        if spec["strategy"] == "single":
            for name, module in model._modules.items():
                for child in (
                    module if isinstance(module, torch.nn.ModuleList) else [module]
                ):
                    handles += [
                        child.register_forward_pre_hook(before(name)),
                        child.register_forward_hook(after(name)),
                    ]
        if ddp:
            model = DistributedDataParallel(
                model,
                device_ids=[local],
                find_unused_parameters=any(
                    isinstance(module, SparseMoE) for module in model.modules()
                ),
            )
        batch = spec["batch_sizes"][rank] if ddp else spec["global_batch_size"]
        inputs = []
        for item in spec["inputs"]:
            if item is None:
                inputs.append(None)
                continue
            shape = [
                (
                    batch
                    if dim == "B"
                    else spec["sequence_length"] if isinstance(dim, str) else dim
                )
                for dim in item["shape"]
            ]
            dtype = getattr(torch, item["dtype"])
            inputs.append(torch.zeros(shape, dtype=dtype, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        precision = spec["precision"]
        if precision == "bf16" and not all(
            torch.cuda.is_bf16_supported(including_emulation=False) for _ in [local]
        ):
            raise ValueError("Selected GPU does not support native BF16")
        scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "fp16", init_scale=1024
        )

        def synchronize():
            for index in ([local] if ddp else devices):
                torch.cuda.synchronize(index)
            if ddp:
                dist.barrier()

        for index in ([local] if ddp else devices):
            torch.cuda.reset_peak_memory_stats(index)
        timings = []
        for step in range(spec["warmup_steps"] + spec["measure_steps"]):
            synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.autocast(
                    "cuda",
                    dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                )
                if precision != "fp32"
                else nullcontext()
            ):
                output = model(*inputs)
                if (
                    not isinstance(output, torch.Tensor)
                    or not output.is_floating_point()
                    or not output.requires_grad
                ):
                    raise ValueError(
                        "Benchmark requires one differentiable floating-point output"
                    )
                if output.ndim == 3 and output.shape[-1] > 1:
                    labels = torch.zeros(
                        output.shape[:-1], dtype=torch.long, device=output.device
                    )
                    loss = torch.nn.functional.cross_entropy(
                        output.float().flatten(0, 1), labels.flatten()
                    )
                else:
                    loss = output.float().square().mean()
                raw = model.module if ddp else model
                loss = loss + moe_auxiliary_loss(raw).to(output.device)
                if ddp:
                    loss = loss * (len(devices) * batch / spec["global_batch_size"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite synthetic loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True, foreach=False
            )
            scaler.step(optimizer)
            scaler.update()
            synchronize()
            elapsed = time.perf_counter() - started
            if step == 0:
                for handle in handles:
                    handle.remove()
            if step >= spec["warmup_steps"]:
                timings.append(elapsed)
        memory = {
            str(index): torch.cuda.max_memory_allocated(index)
            for index in ([local] if ddp else devices)
        }
        if ddp:
            rows = [None] * len(devices)
            dist.all_gather_object(rows, {"timings": timings, "memory": memory})
            timings = [
                max(row["timings"][step] for row in rows)
                for step in range(len(timings))
            ]
            memory = {
                key: value for row in rows for key, value in row["memory"].items()
            }
        if rank == 0:
            median = statistics.median(timings)
            costs = {
                name: sum(a.elapsed_time(b) for a, b in events) / len(events)
                for name, events in module_costs.items()
            }
            write(
                root / spec["result_file"],
                {
                    **spec,
                    "status": "completed",
                    "step_seconds": median,
                    "min_seconds": min(timings),
                    "max_seconds": max(timings),
                    "sequences_per_second": spec["global_batch_size"] / median,
                    "tokens_per_second": spec["global_batch_size"]
                    * spec["sequence_length"]
                    / median,
                    "peak_memory_bytes": memory,
                    "module_cost_ms": costs,
                },
            )
    finally:
        if ddp and not session and dist.is_initialized():
            dist.destroy_process_group()


def orchestrate(root):
    from weave_support.tuning_math import (
        ddp_candidates,
        split_candidates,
        graph_signature,
    )
    from weave_support.tuning_runtime import hardware_identity

    request = json.loads((root / "request.json").read_text())
    devices = request["devices"]
    identity = hardware_identity(devices)
    rows = []
    active = []
    stop_requested = False

    def terminate(_signal, _frame):
        nonlocal stop_requested
        stop_requested = True
        if active and active[0].poll() is None:
            os.killpg(active[0].pid, signal.SIGKILL)
            raise SystemExit(0)
        # A signal during Popen must not leave an untracked child: the post-spawn
        # check below handles that narrow window before the active list is set.

    signal.signal(signal.SIGTERM, terminate)

    def status(state, message):
        write(
            root / "status.json",
            {
                "state": state,
                "message": message,
                "results": rows,
                "updated": time.time(),
            },
        )

    def candidates(strategy, choices):
        specs = []
        for index, options in enumerate(choices):
            specs.append(
                {
                    **{
                        key: value for key, value in request.items() if key != "project"
                    },
                    "strategy": strategy,
                    "batch_sizes": [],
                    "placement": {},
                    "result_file": f"result-{len(rows) + index}.json",
                    **options,
                }
            )
        completed = []
        pending = specs.copy()
        while pending:
            if stop_requested:
                raise SystemExit(0)
            token = f"session-{len(rows)}"
            session_path = root / f"{token}.json"
            write(session_path, pending)
            command = [sys.executable]
            if strategy == "ddp":
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={len(devices)}",
                ]
            command += [
                str(root / "tuning_worker.py"),
                "--root",
                str(root),
                "--session",
                str(session_path),
            ]
            status(
                "running",
                f"Testing {strategy}; compatible candidates reuse this worker",
            )
            log_path = root / f"{token}.log"
            with log_path.open("w") as logs:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    stdout=logs,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[:] = [process]
                if stop_requested:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise SystemExit(0)
                last_progress = time.monotonic()
                timed_out = False
                while True:
                    while pending and (root / pending[0]["result_file"]).exists():
                        result = json.loads(
                            (root / pending.pop(0)["result_file"]).read_text()
                        )
                        rows.append(result)
                        completed.append(result)
                        last_progress = time.monotonic()
                        status(
                            "running",
                            f"Finished {len(rows)} trials; testing {strategy}",
                        )
                    code = process.poll()
                    if code is not None:
                        break
                    if time.monotonic() - last_progress > request["candidate_timeout"]:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        break
                    time.sleep(0.1)
                # Collect the last atomic result if the worker exited between polls.
                while pending and (root / pending[0]["result_file"]).exists():
                    result = json.loads(
                        (root / pending.pop(0)["result_file"]).read_text()
                    )
                    rows.append(result)
                    completed.append(result)
                if pending:
                    failed = {
                        **pending.pop(0),
                        "status": "failed",
                        "error": (
                            "Benchmark timed out"
                            if timed_out
                            else log_path.read_text()[-2500:]
                            or f"Worker exited {process.returncode}"
                        ),
                    }
                    rows.append(failed)
                    completed.append(failed)
                    status(
                        "running",
                        f"Failed {strategy} trial; remaining candidates restart in a fresh worker",
                    )
        return completed

    try:
        standalone = candidates("single", [{"devices": [device]} for device in devices])
        speeds = (
            [row["sequences_per_second"] for row in standalone]
            if all(row["status"] == "completed" for row in standalone)
            else [1] * len(devices)
        )
        if len(devices) > 1:
            candidates(
                "ddp",
                [
                    {"batch_sizes": list(sizes)}
                    for sizes in ddp_candidates(
                        request["global_batch_size"], speeds, request["max_candidates"]
                    )
                ],
            )
            if request["split_supported"]:
                names = request["modules"]
                measured = next(
                    (
                        row["module_cost_ms"]
                        for row in standalone
                        if row["status"] == "completed"
                    ),
                    {},
                )
                costs = [max(0.001, measured.get(name, 1)) for name in names]
                # FP32 is the supported split implementation; compare like precision in UI.
                candidates(
                    "split",
                    [
                        {"placement": placement, "precision": "fp32"}
                        for placement in split_candidates(
                            names, costs, devices, speeds, request["max_candidates"]
                        )
                    ],
                )
        best = {}
        for strategy in ("single", "ddp", "split"):
            choices = [
                row
                for row in rows
                if row["strategy"] == strategy and row["status"] == "completed"
            ]
            if choices:
                best[strategy] = min(choices, key=lambda row: row["step_seconds"])
        if not best:
            raise ValueError(
                "No benchmark configuration completed; inspect candidate errors"
            )
        profile = {
            "id": root.name,
            "created": time.time(),
            "model_name": request["project"]["name"],
            "model_signature": graph_signature(request["project"]),
            "python": request["python"],
            "devices": devices,
            "hardware": identity,
            "sequence_length": request["sequence_length"],
            "global_batch_size": request["global_batch_size"],
            "initial_speed_ratios": [speed / sum(speeds) for speed in speeds],
            "initial_estimate": (
                "standalone"
                if all(row["status"] == "completed" for row in standalone)
                else "equal-fallback"
            ),
            "duration_seconds": time.time() - request.get("started_at", time.time()),
            "best": best,
            "results": rows,
        }
        write(root / "profile.json", profile)
        status(
            "completed",
            f"Measured {len(rows)} trials in {profile['duration_seconds']:.0f}s. Matching training launches can load this profile."
            + (
                " Split was not tested: " + " ".join(request.get("split_reasons", []))
                if not request["split_supported"]
                else ""
            ),
        )
    except Exception as exc:
        status("failed", str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--measure", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    if args.session:
        import torch
        import torch.distributed as dist

        try:
            for spec in json.loads(args.session.read_text()):
                measure(args.root, spec, session=True)
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    elif args.measure:
        measure(args.root, json.loads(args.measure.read_text()))
    else:
        try:
            orchestrate(args.root)
        except Exception as exc:
            path = args.root / "status.json"
            previous = json.loads(path.read_text()) if path.exists() else {}
            write(
                path,
                {
                    **previous,
                    "state": "failed",
                    "message": str(exc),
                    "updated": time.time(),
                },
            )
            raise
