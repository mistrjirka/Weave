"""Isolated inference over a frozen export. Does not import the user's training recipe."""

from contextlib import redirect_stdout
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback


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


def generate(model, tokenizer, request, device, cancel, build=None):
    import torch

    if request["mode"] == "chat":
        messages = (
            (
                [{"role": "system", "content": request["system"]}]
                if request["system"]
                else []
            )
            + request["history"]
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
    generator = torch.Generator(device=device).manual_seed(request["seed"])
    last_emitted = 0
    reason = "length"
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
            if time.monotonic() - last_emitted >= 0.1:
                emit(
                    "text",
                    text=tokenizer.decode(
                        result, skip_special_tokens=processor is None
                    ),
                )
                last_emitted = time.monotonic()
    displayed = result[:-1] if processor is not None and reason == "eos" else result
    emit(
        "text", text=tokenizer.decode(displayed, skip_special_tokens=processor is None)
    )
    emit("done", finish_reason=reason)


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
            emit("error", message=str(exc))


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        try:
            main()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc))
            raise SystemExit(1)
