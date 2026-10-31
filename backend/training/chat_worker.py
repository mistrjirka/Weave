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


def generate(model, tokenizer, request, device, cancel):
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
                emit("text", text=tokenizer.decode(result, skip_special_tokens=True))
                last_emitted = time.monotonic()
    emit("text", text=tokenizer.decode(result, skip_special_tokens=True))
    emit("done", finish_reason=reason)


def main():
    import torch

    torch.set_num_threads(2)
    spec = json.loads(Path(sys.argv[-1]).read_text())
    checkpoint = Path(spec["checkpoint"])
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
    model = importlib.import_module(prefix + "model").Model()
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
    model = model.to(device).eval()
    tokenizer = importlib.import_module(prefix + "load_tokenizer").load_tokenizer()
    if spec.get("chat_template"):
        tokenizer.chat_template = spec["chat_template"]
    emit("ready", has_chat_template=bool(tokenizer.chat_template))
    for line in sys.stdin:
        try:
            generate(
                model, tokenizer, json.loads(line), device, Path(spec["cancel_path"])
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
