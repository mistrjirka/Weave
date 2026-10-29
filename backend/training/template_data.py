"""Bounded data preview; no training script or graph custom code is executed."""

from pathlib import Path
import json


def preview_records(config, project, tokenizer_root):
    path = Path(config.dataset).expanduser()
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        rows = next(parquet.ParquetFile(path).iter_batches(batch_size=3)).to_pylist()
    elif path.suffix == ".json":
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError(
                "Use JSONL or Parquet for previews of datasets larger than 16 MB."
            )
        rows = json.loads(path.read_text())[:3]
    else:
        rows = []
        with path.open() as stream:
            for _ in range(3):
                line = stream.readline(1024 * 1024)
                if not line:
                    break
                if len(line) >= 1024 * 1024:
                    raise ValueError("Preview records must be smaller than 1 MB.")
                rows.append(
                    {"text": line.rstrip()}
                    if path.suffix == ".txt"
                    else json.loads(line)
                )
    from ..tokenizer_bindings import input_tokenizers

    configured = input_tokenizers(project)
    tokenizer = None
    if configured:
        from transformers import AutoTokenizer

        selected = next(iter(configured.values()))
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root / selected["id"],
            trust_remote_code=False,
            local_files_only=True,
        )
        if config.chat_template:
            tokenizer.chat_template = config.chat_template
    previews = []
    for row in rows:
        if "input_ids" in row:
            ids = row["input_ids"]
            labels = row.get("labels", ids)
        elif tokenizer is None:
            raise ValueError(
                "Select a tokenizer on the graph input to preview text, or supply input_ids and labels."
            )
        elif config.recipe == "sft":
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tools=row.get("tools"),
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            ids = encoded["input_ids"]
            mask = encoded.get("assistant_masks")
            if not mask or not any(mask):
                raise ValueError(
                    "This chat template does not supply assistant masks. Provide a generation-aware template or explicit labels."
                )
            labels = [token if active else -100 for token, active in zip(ids, mask)]
        else:
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            labels = ids
        ids = ids[: config.sequence_length + 1]
        labels = labels[: config.sequence_length + 1]
        previews.append(
            {
                "tokens": [
                    {
                        "id": token,
                        "text": (
                            tokenizer.convert_ids_to_tokens(token)
                            if tokenizer
                            else str(token)
                        ),
                        "supervised": index > 0 and labels[index] != -100,
                    }
                    for index, token in enumerate(ids)
                ],
                "supervised_count": sum(label != -100 for label in labels[1:]),
            }
        )
    return {
        "records": previews,
        "note": "Highlights mark next-token targets contributing to loss; the first token has no preceding prediction.",
    }
