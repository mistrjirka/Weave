"""Small, JSON-only reporting contracts shared by custom scripts and the server."""

import hashlib
import json
import math
from pathlib import Path
import uuid


def reporting_contract(metrics, primary_metric, step_unit, headline_metrics):
    if not isinstance(step_unit, str) or not 1 <= len(step_unit) <= 80:
        raise ValueError("step_unit needs 1–80 characters")
    if not isinstance(metrics, dict) or len(metrics) > 128:
        raise ValueError("Define at most 128 metrics")
    result = {}
    for key, spec in metrics.items():
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 120
            or not isinstance(spec, dict)
        ):
            raise ValueError("Metric definitions need a name and a dictionary")
        if set(spec) - {"label", "direction", "min", "max"}:
            raise ValueError("Unknown metric presentation field")
        if "label" in spec and (
            not isinstance(spec["label"], str) or not 1 <= len(spec["label"]) <= 120
        ):
            raise ValueError("Metric labels need 1–120 characters")
        if "direction" in spec and spec["direction"] not in ("min", "max"):
            raise ValueError("Metric direction must be min or max")
        for bound in ("min", "max"):
            if bound in spec and (
                type(spec[bound]) not in (int, float) or not math.isfinite(spec[bound])
            ):
                raise ValueError("Metric bounds must be finite numbers")
        if "min" in spec and "max" in spec and spec["min"] >= spec["max"]:
            raise ValueError("Metric minimum must be less than maximum")
        result[key] = dict(spec)
    if primary_metric is not None and (
        primary_metric not in result or "direction" not in result[primary_metric]
    ):
        raise ValueError("Define a direction for the primary metric")
    if headline_metrics is not None and (
        not isinstance(headline_metrics, (list, tuple))
        or len(headline_metrics) > 8
        or any(key not in result for key in headline_metrics)
    ):
        raise ValueError("Choose at most eight defined headline metrics")
    return {
        "metrics": result,
        "primary_metric": primary_metric,
        "step_unit": step_unit,
        "headline_metrics": (
            list(headline_metrics) if headline_metrics is not None else None
        ),
    }


def latest_contract(records):
    configuration = next(
        (
            item["payload"]
            for item in reversed(records)
            if item["kind"] == "configuration"
            and item.get("channel", "training") == "training"
        ),
        {},
    )
    value = configuration.get("reporting", {})
    try:
        return reporting_contract(
            value.get("metrics", {}),
            value.get("primary_metric"),
            value.get("step_unit", "optimizer updates"),
            value.get("headline_metrics"),
        )
    except (ValueError, TypeError, AttributeError):
        return reporting_contract({}, None, "optimizer updates", None)


def write_table(directory, name, rows, columns, metadata):
    if not isinstance(name, str) or not 1 <= len(name) <= 120:
        raise ValueError("Table names need 1–120 characters")
    if columns is not None and (
        not isinstance(columns, (list, tuple))
        or len(columns) > 64
        or len(set(columns)) != len(columns)
        or any(not isinstance(key, str) or not 1 <= len(key) <= 120 for key in columns)
    ):
        raise ValueError("Choose at most 64 unique column names")
    path = Path(directory) / "artifacts" / "tables" / (uuid.uuid4().hex + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, preview, count, preview_bytes = hashlib.sha256(), [], 0, 0
    selected = list(columns) if columns is not None else None
    try:
        with path.open("xb") as stream:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Table rows must be dictionaries")
                if selected is None:
                    selected = list(row)
                if len(row) > 64 or any(
                    not isinstance(key, str) or len(key) > 120 for key in row
                ):
                    raise ValueError("Table rows need at most 64 named columns")
                data = (
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                ).encode()
                if len(data) > 1024 * 1024:
                    raise ValueError("A table row must be smaller than 1 MB")
                stream.write(data)
                digest.update(data)
                count += 1
                if len(preview) < 20:
                    candidate = {
                        key: (
                            row.get(key)
                            if len(json.dumps(row.get(key))) <= 512
                            else json.dumps(row.get(key))[:512] + "…"
                        )
                        for key in selected
                    }
                    size = len(json.dumps(candidate).encode())
                    if preview_bytes + size <= 64_000:
                        preview.append(candidate)
                        preview_bytes += size
        payload = {
            "path": str(path.relative_to(directory)),
            "name": name,
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
            "rows": count,
            "columns": selected or [],
            "preview": preview,
            "metadata": metadata or {},
        }
        if len(json.dumps(payload, allow_nan=False).encode()) > 128_000:
            raise ValueError("Table preview or metadata is too large")
        return payload
    except BaseException:
        path.unlink(missing_ok=True)
        raise
