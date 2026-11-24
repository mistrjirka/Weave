"""Device placement, checkpoint state and evaluation for text training."""

from __future__ import annotations

import json
import math
import random

import torch
from torch import nn
import torch.distributed as dist
from text_training_data import batch, language_loss


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
                    context = config.get(
                        "generation_context_length", config.get("sequence_length")
                    )
                    logits = model(generated[:, -context:] if context else generated)[
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
