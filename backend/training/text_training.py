"""Managed text trainer: causal text / assistant-only tool-call SFT.

Run: python train.py --config config.json
Multi-GPU: python -m torch.distributed.run --standalone --nproc_per_node=2 train.py --config config.json
The user-owned train.py supplies the loss function; custom loops can use Run directly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from model import Model
from weave_training import Run, atomic_checkpoint_path, atomic_json, file_hash


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def place_model(model, config):
    """Sequential model parallelism; shared tensors must have one owner device."""
    devices = config.get("devices", [])
    if not devices:
        return model.to("cpu"), "cpu"
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable in the selected Python environment. Choose CPU or a CUDA-enabled interpreter."
        )
    for index in devices:
        if index >= torch.cuda.device_count():
            raise ValueError(f"GPU {index} is not visible to this process.")
    device = f"cuda:{devices[0]}"
    if config["strategy"] != "split":
        return model.to(device), device
    placement = config.get("placement", {})
    names = set(model._modules)
    if set(placement) != names:
        raise ValueError(
            "Split placement must assign every exported top-level module: "
            + ", ".join(sorted(names))
        )
    owners = {}
    for name, module in model._modules.items():
        target = int(placement[name])
        if target not in devices:
            raise ValueError(f"{name} is assigned to an unselected GPU.")
        for tensor in [*module.parameters(), *module.buffers()]:
            previous = owners.setdefault(id(tensor), target)
            if previous != target:
                raise ValueError(
                    f"{name} shares weights or buffers with a module on another GPU. Keep shared/tied modules on one GPU."
                )
    for name, module in model._modules.items():
        target = f"cuda:{placement[name]}"
        module.to(target)
        # ModuleList itself is never called; each loop application is.
        targets = module if isinstance(module, nn.ModuleList) else [module]
        for child in targets:

            def transfer(_module, args, kwargs, destination=target):
                return move(args, destination), move(kwargs, destination)

            child.register_forward_pre_hook(transfer, with_kwargs=True)
    return model, device


class TokenDataset:
    """Disk-backed prepared records; only a small working set lives in RAM."""

    def __init__(self, path, offsets):
        from collections import OrderedDict

        self.path, self.offsets = Path(path), offsets
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index not in self.cache:
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[index])
                self.cache[index] = json.loads(stream.readline())
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
        return self.cache[index]


def load_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        for batch_rows in parquet.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch_rows.to_pylist()
    elif path.suffix == ".json":
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError(
                "Use JSONL or Parquet for JSON datasets over 64 MB to keep preparation memory bounded."
            )
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("JSON data must contain an array of records.")
        yield from rows
    else:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield (
                        {"text": line.rstrip()}
                        if path.suffix == ".txt"
                        else json.loads(line)
                    )


def prepare_data(config, directory, tokenizer, *, evaluation=False):
    source = config.get("eval_dataset" if evaluation else "dataset")
    if not source:
        return []
    expected = config.get("dataset_sha256" if not evaluation else "eval_dataset_sha256")
    before = file_hash(source)
    if expected and before != expected:
        raise ValueError("Dataset contents changed since this run was created.")
    rows = load_rows(source)
    suffix = "eval" if evaluation else "train"
    target = directory / f"prepared-{suffix}.jsonl"
    offsets = []
    output = target.with_suffix(".pending").open("wb")
    for index, row in enumerate(rows):
        if "input_ids" in row:
            ids = row["input_ids"]
            labels = row.get("labels", ids)
        elif config["recipe"] == "sft":
            if tokenizer is None:
                raise ValueError(
                    "SFT requires a tokenizer on the token input, or pretokenized input_ids and labels."
                )
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tools=row.get("tools"),
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
            )
            ids = encoded["input_ids"]
            mask = encoded.get("assistant_masks")
            if not mask or not any(mask):
                raise ValueError(
                    "Assistant-only SFT needs a chat template with generation/assistant masks. Supply a compatible template or explicit input_ids and labels (-100 for ignored tokens)."
                )
            labels = [token if active else -100 for token, active in zip(ids, mask)]
        else:
            if tokenizer is None:
                raise ValueError(
                    "Text records require a tokenizer selected on the model input."
                )
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            labels = list(ids)
        if len(ids) != len(labels):
            raise ValueError(
                f"Record {index}: input_ids and labels have different lengths."
            )
        if any(not isinstance(v, int) or v < 0 for v in ids):
            raise ValueError(f"Record {index}: token IDs must be nonnegative integers.")
        if any(
            type(value) is not int or (value < 0 and value != -100) for value in labels
        ):
            raise ValueError(
                f"Record {index}: labels must be token IDs or -100 for ignored targets."
            )
        capacity = config.get("vocabulary_size")
        if capacity and (
            max(ids, default=0) >= capacity or max(labels, default=0) >= capacity
        ):
            raise ValueError(
                f"Record {index}: token IDs exceed the vocabulary head size ({capacity}). Check the tokenizer and head configuration."
            )
        chunks = (
            range(0, max(1, len(ids) - 1), config["sequence_length"])
            if config["recipe"] == "causal"
            else [0]
        )
        for start in chunks:
            chunk_ids = list(ids[start : start + config["sequence_length"] + 1])
            chunk_labels = list(labels[start : start + config["sequence_length"] + 1])
            if len(chunk_ids) < 2:
                continue
            if not any(v != -100 for v in chunk_labels[1:]):
                raise ValueError(
                    f"Record {index}: no supervised next-token targets after truncation."
                )
            identity = hashlib.sha256(
                json.dumps(
                    {"input_ids": chunk_ids, "labels": chunk_labels}, sort_keys=True
                ).encode()
            ).hexdigest()
            record = {
                "input_ids": chunk_ids,
                "labels": chunk_labels,
                "record": len(offsets),
                "source_record": index,
                "chunk_start": start,
                "sha256": identity,
            }
            offsets.append(output.tell())
            output.write((json.dumps(record) + "\n").encode())
    output.flush()
    os.fsync(output.fileno())
    output.close()
    if file_hash(source) != before:
        raise ValueError("Dataset changed while it was being read.")
    if not offsets:
        raise ValueError(
            "Dataset is empty or contains no sequences with next-token targets."
        )
    os.replace(target.with_suffix(".pending"), target)
    atomic_json(directory / f"prepared-{suffix}-offsets.json", offsets)
    atomic_json(
        directory / f"data-{suffix}.json",
        {
            "path": str(Path(source).resolve()),
            "sha256": file_hash(source),
            "prepared_sha256": file_hash(target),
            "rows": len(offsets),
            "coverage": "source and prepared records verified; consumed record IDs recorded per optimizer step; deterministic sampling with replacement",
            "sequence_length": config["sequence_length"],
            "recipe": config["recipe"],
        },
    )
    return TokenDataset(target, offsets)


def batch(rows, device):
    width = max(len(row["input_ids"]) for row in rows) - 1
    inputs = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    targets = torch.full_like(inputs, -100)
    for i, row in enumerate(rows):
        length = len(row["input_ids"]) - 1
        inputs[i, :length] = torch.tensor(row["input_ids"][:-1], device=device)
        targets[i, :length] = torch.tensor(row["labels"][1:], device=device)
    return inputs, targets


def language_loss(logits, targets):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != targets.shape
    ):
        raise ValueError(
            "This recipe expects one floating-point [batch, sequence, vocabulary] output. Use a custom script for other contracts."
        )
    targets = targets.to(logits.device)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def checkpoint_state(model, optimizer, scheduler, scaler, *, step, tokens, config):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens": tokens,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": config,
        "data_cursor": {
            "next_step": step,
            "scheme": "seed + step, deterministic indexed batches",
        },
    }


def capture_rng():
    return {
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def gather_rng(world):
    local = capture_rng()
    if world == 1:
        return [local]
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    return gathered


def restore_rng(state):
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def evaluate(
    model, rows, tokenizer, config, run, checkpoint, device, loss_fn=language_loss
):
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            total_loss, count = 0.0, 0
            for offset in range(
                0,
                min(len(rows), config["eval_batches"] * config["batch_size"]),
                config["batch_size"],
            ):
                inputs, targets = batch(
                    rows[offset : offset + config["batch_size"]], device
                )
                total_loss += loss_fn(model(inputs), targets).item()
                count += (targets != -100).sum().item()
            if count:
                value = total_loss / count
                run.log(
                    {"eval/loss": value, "eval/perplexity": math.exp(min(value, 50))},
                    step=run.step,
                    tokens=run.tokens,
                )
            for prompt in config.get("prompts", []):
                if tokenizer is None:
                    run.warning(
                        "Generation prompts need a tokenizer; evaluation loss is still available.",
                        code="generation-tokenizer",
                    )
                    break
                prompt_text = (
                    prompt
                    if isinstance(prompt, str)
                    else json.dumps(prompt["messages"], ensure_ascii=False)
                )
                ids = (
                    tokenizer.encode(prompt, add_special_tokens=False)
                    if isinstance(prompt, str)
                    else tokenizer.apply_chat_template(
                        prompt["messages"],
                        tools=prompt.get("tools"),
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
                if not ids:
                    continue
                generated = torch.tensor([ids], device=device)
                for _ in range(config["generation_tokens"]):
                    logits = model(generated[:, -config["sequence_length"] :])[
                        :, -1
                    ].float()
                    token = logits.argmax(-1).to(device)
                    generated = torch.cat([generated, token[:, None]], dim=1)
                    if token.item() == tokenizer.eos_token_id:
                        break
                output = tokenizer.decode(
                    generated[0, len(ids) :].tolist(), skip_special_tokens=False
                )
                metadata = {"decoding": "greedy", "seed": config["seed"]}
                if not isinstance(prompt, str):
                    try:
                        parsed = json.loads(output)
                        call = (
                            parsed[0] if isinstance(parsed, list) and parsed else parsed
                        )
                        function = call.get("function", call)
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        metadata["parse_success"] = True
                        schemas = {
                            tool.get("function", tool)["name"]: tool.get(
                                "function", tool
                            ).get("parameters", {})
                            for tool in prompt.get("tools", [])
                        }
                        name = function.get("name")
                        metadata["tool_name_valid"] = name in schemas
                        if name in schemas:
                            import jsonschema

                            try:
                                jsonschema.validate(arguments, schemas[name])
                                metadata["schema_valid"] = True
                            except jsonschema.ValidationError:
                                metadata["schema_valid"] = False
                        if "expected_tool" in prompt:
                            metadata["tool_name_correct"] = (
                                name == prompt["expected_tool"]
                            )
                        if "expected_arguments" in prompt:
                            metadata["arguments_correct"] = (
                                arguments == prompt["expected_arguments"]
                            )
                    except (ValueError, TypeError, AttributeError, IndexError):
                        metadata["parse_success"] = False
                    run.log(
                        {
                            "eval/" + key: float(value)
                            for key, value in metadata.items()
                            if isinstance(value, bool)
                        },
                        step=run.step,
                    )
                run.sample(
                    prompt=prompt_text,
                    output=output,
                    checkpoint=checkpoint,
                    metadata=metadata,
                )
    finally:
        model.train(was_training)


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
    torch.set_num_threads(config.get("cpu_threads", 2))
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
    with Run(report_directory, rank=rank, source_directory=config_path.parent) as run:
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


if __name__ == "__main__":
    main()
