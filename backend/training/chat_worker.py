"""Isolated inference over a frozen export. Does not import the user's training recipe."""

from contextlib import redirect_stdout
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).parent))
from chat_protocol import clean_assistant, inspect_calls


def emit(kind, **payload):
    sys.__stdout__.write(json.dumps({"kind": kind, **payload}, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def decoder_builder(identifier, registrations, builders):
    """Import only a target registered by the server owner, inside this worker."""
    if identifier not in registrations:
        raise ValueError("The selected decoder is not registered for this session.")
    if identifier not in builders:
        registration = registrations[identifier]
        module, separator, attribute = registration["target"].partition(":")
        if (
            not separator
            or not all(part.isidentifier() for part in module.split("."))
            or not attribute.isidentifier()
        ):
            raise ValueError("Decoder target must be a local module:factory.")
        if registration.get("python_path"):
            directory = Path(registration["python_path"])
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError(
                    "Decoder python_path must be an existing absolute local directory."
                )
            sys.path.insert(0, str(directory))
        factory = getattr(importlib.import_module(module), attribute)
        if not callable(factory):
            raise ValueError("The registered decoder factory must be callable.")
        builders[identifier] = factory
    return builders[identifier]


def prompt_ids(tokenizer, request, history=None):
    if request["mode"] == "chat":
        template = (
            request.get("chat_template")
            or getattr(tokenizer, "chat_template", "")
            or ""
        )
        prior = []
        for original in (request.get("history", []) if history is None else history):
            message = dict(original)
            if message["role"] == "assistant":
                message["content"] = clean_assistant(message.get("content", ""))
                if message.get("tool_calls") and "tool_calls" in template:
                    message["content"] = inspect_calls(
                        message["content"], request.get("tools")
                    )["text"]
                elif "tool_calls" not in template:
                    message.pop("tool_calls", None)
            prior.append(message)
        messages = (
            (
                [{"role": "system", "content": request["system"]}]
                if request["system"]
                else []
            )
            + prior
            + [{"role": "user", "content": request["text"]}]
        )
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            chat_template=request.get("chat_template") or None,
            **({"tools": request["tools"]} if request.get("tools") is not None else {}),
        )
    else:
        ids = tokenizer.encode(request["text"], add_special_tokens=False)
    if not ids:
        if tokenizer.bos_token_id is None:
            raise ValueError(
                "The prompt produced no token IDs. Enter a nonempty prompt."
            )
        ids = [tokenizer.bos_token_id]
    return ids


def preview(tokenizer, request):
    ids = prompt_ids(tokenizer, request)
    without_history = prompt_ids(tokenizer, request, [])
    return {
        "prompt_tokens": len(ids),
        "history_tokens": max(0, len(ids) - len(without_history)),
        "reserved_reply_tokens": request["max_new_tokens"],
        "remaining_tokens": request["context_length"]
        - len(ids)
        - request["max_new_tokens"],
        "context_length": request["context_length"],
    }


def generate(model, tokenizer, request, device, cancel, build=None, publish=None):
    import torch

    publish = publish or emit
    ids = prompt_ids(tokenizer, request)
    if len(ids) + request["max_new_tokens"] > request["context_length"]:
        raise ValueError(
            f"Prompt uses {len(ids)} tokens; with {request['max_new_tokens']} new tokens it exceeds the {request['context_length']}-token budget. Increase the context budget within the model's limits, shorten the prompt or start a new conversation."
        )
    generated = torch.tensor([ids], device=device, dtype=torch.long)
    processor = None
    if request.get("decoder") is not None:
        if build is None:
            raise ValueError("The selected decoder is unavailable.")
        processor = build(request.get("tools"), tokenizer, request)
        if not callable(processor):
            raise ValueError("Decoder build() must return a callable logits mask.")
    result = []
    preserve_xml = processor is not None or request.get("tools") is not None
    generator = torch.Generator(device=device).manual_seed(request["seed"])
    last_emitted = 0
    reason = "length"
    endings = (
        {
            tokenizer.convert_tokens_to_ids(marker)
            for marker in ("<|end|>", "<|im_end|>", "<|eot_id|>")
            if marker in getattr(tokenizer, "all_special_tokens", [])
        }
        if request["mode"] == "chat"
        else set()
    )
    with torch.inference_mode():
        for _ in range(request["max_new_tokens"]):
            if cancel.is_file():
                reason = "stopped"
                break
            logits = model(generated)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[:2] != generated.shape
            ):
                raise ValueError(
                    "The frozen model must return vocabulary logits shaped [batch, sequence, vocabulary]."
                )
            scores = logits[0, -1].float().to(device)
            if not torch.isfinite(scores).all():
                raise ValueError("Model logits contain NaN or infinity.")
            penalty = request.get("repetition_penalty", 1.0)
            if penalty != 1.0:
                seen = generated[0].unique()
                values = scores[seen]
                scores[seen] = torch.where(
                    values < 0, values * penalty, values / penalty
                )
            if processor is not None:
                masked = processor(generated[0, len(ids) :], scores, request)
                if (
                    not isinstance(masked, torch.Tensor)
                    or masked.shape != scores.shape
                    or masked.device != scores.device
                    or not masked.is_floating_point()
                ):
                    raise ValueError(
                        "Decoder must return float logits with the same shape and device."
                    )
                if (
                    not (torch.isfinite(masked) | torch.isneginf(masked)).all()
                    or not torch.isfinite(masked).any()
                ):
                    raise ValueError(
                        "Decoder returned invalid scores or blocked every token; generation stopped."
                    )
                scores = masked
            if request["temperature"] == 0:
                token = scores.argmax().reshape(1)
            else:
                sorted_scores, indices = torch.sort(
                    scores / request["temperature"], descending=True
                )
                probabilities = torch.softmax(sorted_scores, dim=-1)
                remove = probabilities.cumsum(-1) - probabilities >= request["top_p"]
                probabilities = probabilities.masked_fill(remove, 0)
                token = indices[
                    torch.multinomial(probabilities, 1, generator=generator)
                ]
            identifier = int(token.item())
            result.append(identifier)
            generated = torch.cat((generated, token.reshape(1, 1)), dim=1)
            if identifier == tokenizer.eos_token_id:
                reason = "eos"
                break
            if identifier in endings:
                reason = "turn_end"
                break
            if time.monotonic() - last_emitted >= 0.1:
                publish(
                    "text",
                    text=tokenizer.decode(result, skip_special_tokens=not preserve_xml),
                )
                last_emitted = time.monotonic()
    displayed = (
        result[:-1] if preserve_xml and reason in ("eos", "turn_end") else result
    )
    publish(
        "text", text=tokenizer.decode(displayed, skip_special_tokens=not preserve_xml)
    )
    publish("done", finish_reason=reason)


def run_suite(model, tokenizer, request, device, cancel, registrations, builders):
    for index, case in enumerate(request["cases"]):
        if cancel.is_file():
            break
        output, reason = "", None

        def capture(kind, **payload):
            nonlocal output, reason
            if kind == "text":
                output = payload["text"]
            if kind == "done":
                reason = payload["finish_reason"]

        try:
            factory = (
                decoder_builder(case["decoder"], registrations, builders)
                if case.get("decoder")
                else None
            )
            generate(model, tokenizer, case, device, cancel, factory, capture)
            parsed = inspect_calls(output, case.get("tools"))
            actual = [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in parsed["calls"]
            ]
            expected = case.get("expected")
            if isinstance(expected, str):
                expected_parsed = inspect_calls(expected, case.get("tools"))
                if expected_parsed["errors"]:
                    raise ValueError(
                        "Invalid expected XML: " + "; ".join(expected_parsed["errors"])
                    )
                expected = [
                    {"name": call["name"], "arguments": call["arguments"]}
                    for call in expected_parsed["calls"]
                ]
            valid = parsed["kind"] in ("calls", "no_call") and not parsed["errors"]
            tool_match = (
                [call["name"] for call in actual] == [call["name"] for call in expected]
                if expected is not None
                else None
            )
            exact = valid and actual == expected if expected is not None else None
            emit(
                "suite_case",
                result={
                    "index": index,
                    "name": case.get("name") or f"Case {index + 1}",
                    "output": output,
                    "parsed": parsed,
                    "finish_reason": reason,
                    "parse_valid": valid,
                    "tool_match": tool_match,
                    "arguments_match": (
                        exact if tool_match else False if expected is not None else None
                    ),
                    "exact_match": exact,
                    "settings": case,
                },
            )
        except Exception as exc:
            emit(
                "suite_case",
                result={
                    "index": index,
                    "name": case.get("name") or f"Case {index + 1}",
                    "output": output,
                    "error": str(exc),
                    "parse_valid": False,
                    "exact_match": False,
                    "settings": case,
                },
            )
    emit("suite_done", stopped=cancel.is_file())


def main():
    emit("loading", stage="Importing PyTorch")
    import torch

    torch.set_num_threads(2)
    spec = json.loads(Path(sys.argv[-1]).read_text())
    checkpoint = Path(spec["checkpoint"])
    emit("loading", stage="Verifying checkpoint integrity")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(
            "Checkpoint hash differs from its published record. The file may have changed."
        )
    prefix = (
        "weave_support." if (Path.cwd() / "weave_support/model.py").is_file() else ""
    )
    emit("loading", stage="Building the frozen model")
    model = importlib.import_module(prefix + "model").Model()
    emit("loading", stage="Loading checkpoint weights")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state.get("model", state))
    del state
    device = spec["device"]
    if device.startswith("cuda:"):
        index = int(device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(
                f"{device} is unavailable in the run's Python environment."
            )
        torch.cuda.set_device(index)
    emit("loading", stage=f"Moving model to {device}")
    model = model.to(device).eval()
    emit("loading", stage="Loading the saved tokenizer")
    tokenizer = importlib.import_module(prefix + "load_tokenizer").load_tokenizer()
    if spec.get("chat_template"):
        tokenizer.chat_template = spec["chat_template"]
    builders = {}
    registrations = spec.get("decoders", {})
    if spec.get("decoder"):
        emit("loading", stage="Loading the selected decoder")
        decoder_builder(spec["decoder"], registrations, builders)
    emit("ready", has_chat_template=bool(tokenizer.chat_template))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "preview":
                emit(
                    "preview",
                    request_id=request["request_id"],
                    **preview(tokenizer, request),
                )
                continue
            if request.get("operation") == "suite":
                run_suite(
                    model,
                    tokenizer,
                    request,
                    device,
                    Path(spec["cancel_path"]),
                    registrations,
                    builders,
                )
                continue
            build = (
                decoder_builder(request["decoder"], registrations, builders)
                if request.get("decoder")
                else None
            )
            generate(
                model, tokenizer, request, device, Path(spec["cancel_path"]), build
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit(
                "error",
                message=str(exc),
                operation=request.get("operation"),
                request_id=request.get("request_id"),
            )


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        try:
            main()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc))
            raise SystemExit(1)
