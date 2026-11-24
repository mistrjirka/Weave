"""Managed text training lifecycle and CLI implementation."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from model import Model
from weave_training import Run, atomic_checkpoint_path, available_cpu_threads, file_hash
from text_training_data import TokenDataset, batch, language_loss, prepare_data
from text_training_support import checkpoint_state, evaluate, gather_rng, place_model, restore_rng


def main(loss_fn=None):
    loss_function = loss_fn or language_loss
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(sys.argv[0]).resolve().parent / "config.json")
    )
    parser.add_argument("--evaluate")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text())
    for key in ("dataset", "eval_dataset", "checkpoint"):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key] = str((config_path.parent / config[key]).resolve())
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl" if config["devices"] else "gloo")
        if config["devices"]:
            torch.cuda.set_device(config["devices"][rank])
    if args.evaluate and not Path(args.evaluate).is_absolute():
        args.evaluate = str((config_path.parent / args.evaluate).resolve())
    if os.environ.get("WEAVE_RUN_DIR"):
        directory = Path(os.environ["WEAVE_RUN_DIR"]).resolve()
    elif args.evaluate:
        directory = Path(args.evaluate).resolve().parent.parent
    else:
        identity = [str(time.time_ns()) if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(identity, src=0)
        directory = config_path.parent / "runs" / ("run-" + identity[0])
    if rank == 0:
        print(f"Run outputs: {directory}", flush=True)
    torch.set_num_threads(config.get("cpu_threads", available_cpu_threads()))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokenizer = None
    if (Path(__file__).parent / "tokenizers.json").is_file() and json.loads(
        (Path(__file__).parent / "tokenizers.json").read_text()
    ):
        from load_tokenizer import load_tokenizer

        tokenizer = load_tokenizer()
    if config.get("chat_template") and tokenizer is not None:
        tokenizer.chat_template = config["chat_template"]
    from types import SimpleNamespace
    from tuning_runtime import load_tuning

    tuned = SimpleNamespace(
        use_gpu_tuning=config.get("use_gpu_tuning", True),
        resolved_strategy=config["strategy"],
        devices=config["devices"],
        precision=config["precision"],
        micro_batch_size=config["batch_size"],
        placement=config.get("placement", {}),
        rank_micro_batch_sizes=[],
    )
    tuning_profile, tuning_note = (
        (None, "")
        if args.evaluate
        else load_tuning(tuned, config, Path(__file__).parent)
    )
    config["placement"] = tuned.placement
    if tuning_note:
        print(tuning_note, flush=True)
    model = Model()
    placement_config = dict(config)
    if world > 1:
        placement_config["devices"] = (
            [config["devices"][rank]] if config["devices"] else []
        )
    if args.evaluate:
        placement_config.update(
            strategy="single", devices=[config["evaluation_device"]]
        )
    model, device = place_model(model, placement_config)
    precision = config["precision"]
    if precision == "bf16" and config["devices"]:
        for index in config["devices"]:
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError(
                        f"GPU {index} does not support native BF16; choose FP32 or FP16."
                    )
    if precision == "fp16" and device == "cpu":
        raise ValueError("FP16 training requires CUDA. Choose FP32 on CPU.")
    if config["strategy"] == "split" and precision != "fp32":
        raise ValueError(
            "The initial split-model template uses FP32; custom scripts can add per-device autocast."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, config["warmup_steps"]))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    torch.manual_seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    start_step, tokens = 0, 0
    source = args.evaluate or config.get("checkpoint")
    if source:
        state = torch.load(source, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        if config["start_mode"] == "resume" and not args.evaluate:
            old = state["config"]
            required = (
                "dataset_sha256",
                "batch_size",
                "accumulation",
                "seed",
                "sequence_length",
                "strategy",
                "devices",
                "precision",
                "recipe",
                "model_sha256",
                "script_sha256",
                "chat_template",
                "learning_rate",
                "weight_decay",
                "warmup_steps",
            )
            if any(old.get(key) != config.get(key) for key in required):
                raise ValueError(
                    "Resume configuration differs from the checkpoint. Start a new stage from its weights instead."
                )
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_step, tokens = state["step"], state["tokens"]
            restore_rng(state.get("rank_rng", [state])[rank])
        elif args.evaluate:
            start_step, tokens = state["step"], state["tokens"]
    evaluation = args.evaluate is not None
    # Separate writers per distributed rank and evaluation checkpoint.
    report_directory = (
        directory / ("evaluation-" + Path(args.evaluate).stem)
        if evaluation
        else directory
    )
    if evaluation:
        rank = 0
    with Run(
        report_directory,
        rank=rank,
        source_directory=config_path.parent,
        channel="evaluation" if args.evaluate else "training",
    ) as run:
        run.step, run.tokens = start_step, tokens
        run.capabilities(
            metrics=True,
            samples=True,
            checkpoint=not evaluation,
            commands=not evaluation,
        )
        eval_rows = (
            prepare_data(config, report_directory, tokenizer, evaluation=True)
            if rank == 0
            else []
        )
        if evaluation:
            evaluate(
                model,
                eval_rows,
                tokenizer,
                config,
                run,
                str(Path(args.evaluate).relative_to(directory)),
                device,
                loss_fn=loss_function,
            )
            return
        rows = prepare_data(config, directory, tokenizer) if rank == 0 else None
        if world > 1:
            dist.barrier()
            if rank != 0:
                rows = TokenDataset(
                    directory / "prepared-train.jsonl",
                    json.loads((directory / "prepared-train-offsets.json").read_text()),
                )
        config["prepared_sha256"] = file_hash(directory / "prepared-train.jsonl")
        if (
            source
            and config["start_mode"] == "resume"
            and state["config"].get("prepared_sha256") != config["prepared_sha256"]
        ):
            raise ValueError(
                "Prepared training data changed; exact resume is unavailable."
            )
        if world > 1:
            if config["devices"]:
                model = DistributedDataParallel(
                    model,
                    device_ids=[config["devices"][rank]],
                    find_unused_parameters=config.get("has_moe", False),
                )
            else:
                model = DistributedDataParallel(
                    model, find_unused_parameters=config.get("has_moe", False)
                )
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        model.train()
        block_activations = {}
        if config.get("inspect_blocks"):
            for name, module in raw_model._modules.items():

                def summarize(_module, _inputs, output, label=name):
                    if (
                        _module.training
                        and run.step % config["summary_every"] == 0
                        and isinstance(output, torch.Tensor)
                    ):
                        block_activations[label] = (
                            output.detach().float().square().mean().sqrt().item()
                        )

                targets = module if isinstance(module, nn.ModuleList) else [module]
                for target in targets:
                    target.register_forward_hook(summarize)
        trace_path = directory / f"consumed-rank-{rank}.jsonl"
        started = time.monotonic()
        initial_tokens = tokens
        stop = False
        latest_checkpoint = None
        for step in range(start_step, config["max_steps"]):
            commands = run.poll_commands() if rank == 0 else []
            requests = [commands]
            if world > 1:
                dist.broadcast_object_list(requests, src=0)
                commands = requests[0]
            stop = any(command["kind"] == "stop" for command in commands)
            snapshot = any(command["kind"] == "snapshot" for command in commands)
            if stop:
                break
            # Indexed sampling: batch contents are reproducible from seed + step.
            generator = random.Random(config["seed"] + step)
            count = config["batch_size"] * config["accumulation"] * world
            indices = [generator.randrange(len(rows)) for _ in range(count)]
            if tuned.rank_micro_batch_sizes:
                sizes = tuned.rank_micro_batch_sizes
                global_microbatch = sum(sizes)
                start = sum(sizes[:rank])
                microbatches = [
                    indices[offset + start : offset + start + sizes[rank]]
                    for offset in range(0, count, global_microbatch)
                ]
                local_indices = [
                    index for batch_indices in microbatches for index in batch_indices
                ]
            else:
                local_indices = indices[rank::world]
                microbatches = [
                    local_indices[i : i + config["batch_size"]]
                    for i in range(0, len(local_indices), config["batch_size"])
                ]
            label_count = sum(
                sum(value != -100 for value in rows[i]["labels"][1:])
                for i in local_indices
            )
            statistics = torch.tensor(
                [
                    0.0,
                    float(label_count),
                    float(sum(len(rows[i]["input_ids"]) - 1 for i in local_indices)),
                ],
                device=device,
                dtype=torch.float64,
            )
            total_labels = statistics[1].clone()
            if world > 1:
                dist.all_reduce(total_labels)
            optimizer.zero_grad(set_to_none=True)
            for number, indices_batch in enumerate(microbatches):
                inputs, targets = batch([rows[i] for i in indices_batch], device)
                sync = (
                    model.no_sync()
                    if world > 1 and number < len(microbatches) - 1
                    else contextlib.nullcontext()
                )
                autocast = torch.autocast(
                    "cuda" if config["devices"] else "cpu",
                    dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                    enabled=precision != "fp32",
                )
                with sync, autocast:
                    loss_sum = loss_function(model(inputs), targets)
                    loss = loss_sum * (world / total_labels.item())
                    from runtime import moe_auxiliary_loss

                    auxiliary = moe_auxiliary_loss(raw_model)
                    if isinstance(auxiliary, torch.Tensor):
                        loss = loss + auxiliary.to(loss.device) / len(microbatches)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; optimizer step was not applied.")
                scaler.scale(loss).backward()
                statistics[0] += loss_sum.detach().to(device)
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config["gradient_clip"],
                error_if_nonfinite=True,
                foreach=False,
            )
            if config.get("inspect_blocks") and step % config["summary_every"] == 0:
                summaries = {}
                for name, module in raw_model._modules.items():
                    squares = [
                        parameter.grad.detach().float().square().sum().item()
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    ]
                    summaries[f"blocks/{name}/gradient_norm"] = math.sqrt(sum(squares))
                    if name in block_activations:
                        summaries[f"blocks/{name}/activation_rms"] = block_activations[
                            name
                        ]
                run.log(summaries, step=step + 1, tokens=tokens)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if world > 1:
                dist.all_reduce(statistics)
            tokens += int(statistics[2].item())
            run.step, run.tokens = step + 1, tokens
            with trace_path.open("a") as trace:
                trace.write(
                    json.dumps(
                        {
                            "step": step + 1,
                            "records": [
                                {
                                    "index": rows[i]["record"],
                                    "sha256": rows[i]["sha256"],
                                }
                                for i in local_indices
                            ],
                        }
                    )
                    + "\n"
                )
            if (step + 1) % config["log_every"] == 0 or step == start_step:
                run.log(
                    {
                        "train/loss": statistics[0].item() / statistics[1].item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/gradient_norm": float(norm),
                        "train/tokens_per_second": (tokens - initial_tokens)
                        / max(0.001, time.monotonic() - started),
                    },
                    step=step + 1,
                    tokens=tokens,
                )
                run.progress(completed=step + 1, total=config["max_steps"])
            checkpoint_due = (
                snapshot
                or (step + 1) % config["checkpoint_every"] == 0
                or step + 1 == config["max_steps"]
            )
            if checkpoint_due:
                rng_states = gather_rng(world)
                if rank == 0:
                    latest_checkpoint = (
                        directory
                        / "checkpoints"
                        / f"step-{step + 1}-{time.time_ns()}.pt"
                    )
                    state = checkpoint_state(
                        raw_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step=step + 1,
                        tokens=tokens,
                        config=config,
                    )
                    state["rank_rng"] = rng_states
                    with atomic_checkpoint_path(latest_checkpoint) as temporary:
                        torch.save(state, temporary)
                    run.checkpoint(
                        latest_checkpoint,
                        metadata={
                            "resume": "full template state including per-rank RNG",
                            "snapshot_requests": [
                                c["id"] for c in commands if c["kind"] == "snapshot"
                            ],
                        },
                    )
                    if config["strategy"] != "train_eval":
                        evaluate(
                            raw_model,
                            eval_rows,
                            tokenizer,
                            config,
                            run,
                            str(latest_checkpoint.relative_to(directory)),
                            device,
                            loss_fn=loss_function,
                        )
                if world > 1:
                    dist.barrier()
        if stop:
            rng_states = gather_rng(world)
            if rank == 0:
                destination = (
                    directory
                    / "checkpoints"
                    / f"stopped-{run.step}-{time.time_ns()}.pt"
                )
                with atomic_checkpoint_path(destination) as temporary:
                    state = checkpoint_state(
                        raw_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step=run.step,
                        tokens=tokens,
                        config=config,
                    )
                    state["rank_rng"] = rng_states
                    torch.save(state, temporary)
                run.checkpoint(destination)
                run.emit(
                    "stopped", {"message": "Stopped safely at an optimizer boundary."}
                )
        if world > 1:
            dist.destroy_process_group()
