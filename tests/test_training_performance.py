"""Numerical and checkpoint contracts for optional execution optimizations."""

import copy
import json
import sys

import pytest
import torch
from torch.nn import functional as F

from backend import runtime, export_helpers, sparse_runtime
from backend.compiler import compile_project
from backend.training.performance import configure_execution
from test_training import tiny_project
from test_training_hooks import make_hook_run, execute, filled_starter
from backend.training.service import events
from backend.training.weave_training import atomic_json, file_hash


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scaling", ["none", "yarn"])
def test_long_rope_uses_float32_angles_and_reuses_transient_cache(dtype, scaling):
    attention = runtime.Attention(
        16, 2, 2, theta=1e6, rope_scaling=scaling, rope_factor=2
    )
    x = torch.randn(1, 2, 8192, 8).to(dtype).requires_grad_()
    actual = attention.rotate(x)
    expected = export_helpers.rotary_positions(x, 1e6, scaling, 2)
    torch.testing.assert_close(actual, expected)
    cache = attention._rope_cache
    assert attention.rotate(x).shape == x.shape
    assert attention._rope_cache is cache
    assert not any("cache" in key for key in attention.state_dict())
    actual.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    attention.float()
    assert attention._rope_cache is None
    # Catch the previous bug independently of the other RoPE implementation.
    one = torch.zeros(1, 1, 8192, 8, dtype=dtype)
    one[..., 0] = 1
    if scaling == "none":
        torch.testing.assert_close(
            attention.rotate(one)[0, 0, 4095, 0], torch.tensor(4095.0).cos().to(dtype)
        )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("chunk", [1, 4, 32])
def test_chunked_projection_matches_loss_and_all_gradients_with_masking(bias, chunk):
    torch.manual_seed(19)
    x = torch.randn(2, 7, 8, requires_grad=True)
    weight = torch.randn(13, 8, requires_grad=True)
    offset = torch.randn(13, requires_grad=True) if bias else None
    labels = torch.randint(13, (2, 7))
    labels[0, :5] = -100  # Includes wholly masked chunks.
    parameters = [x, weight] + ([offset] if bias else [])
    reference = F.cross_entropy(
        F.linear(x, weight, offset).float().flatten(0, 1), labels.flatten()
    )
    expected = torch.autograd.grad(reference, parameters)
    actual = export_helpers.linear_cross_entropy(x, weight, labels, offset, chunk)
    gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, reference)
    for got, wanted in zip(gradients, expected):
        torch.testing.assert_close(got, wanted, atol=1e-6, rtol=1e-5)


def test_generated_loss_keeps_weight_tying_and_checkpoint_keys(monkeypatch):
    project = tiny_project()
    project["graph"]["nodes"][1]["data"]["share"] = "tokens"
    project["graph"]["nodes"][3]["data"]["share"] = "tokens"
    files = compile_project(project)
    for name, module in [
        ("runtime", runtime),
        ("export_helpers", export_helpers),
        ("sparse_runtime", sparse_runtime),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    namespace = {}
    exec(files["model.py"], namespace)
    original = namespace["Model"]()
    chunked = copy.deepcopy(original)
    inputs = torch.randint(8, (2, 7))
    labels = torch.randint(8, (2, 7))
    labels[0, 0] = -100
    logits = original(inputs)
    expected = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
    loss = chunked(inputs, loss_targets=labels, loss_chunk_size=3)
    expected.backward()
    loss.backward()
    torch.testing.assert_close(loss, expected)
    assert original.state_dict().keys() == chunked.state_dict().keys()
    for left, right in zip(original.parameters(), chunked.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)


def test_profiler_exports_bounded_trace_and_preserves_training(tmp_path):
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps({"input_ids": [1, 2, 3, 4] * 40}) + "\n")
    script = filled_starter(data).replace("profile_steps=0", "profile_steps=1")
    _, directory, _ = make_hook_run(tmp_path, script=script)
    execute(directory)
    records = events(directory)
    trace = directory / "profiles/rank-0.json"
    assert trace.is_file()
    assert json.loads(trace.read_text())["traceEvents"]
    assert len([e for e in records if e["kind"] == "profile"]) == 1
    assert [e for e in records if e["kind"] == "metrics"][-1]["step"] == 4


def test_fused_override_rejects_cpu_and_custom_optimizer():
    from types import SimpleNamespace

    model = torch.nn.Linear(3, 2)
    config = SimpleNamespace(fused_optimizer=True, compile_mode="off")
    with pytest.raises(ValueError, match="CUDA"):
        configure_execution(model, torch.optim.AdamW(model.parameters()), config)
    with pytest.raises(ValueError, match="AdamW"):
        configure_execution(model, torch.optim.SGD(model.parameters(), lr=0.1), config)


@pytest.mark.parametrize("refresh", [False, True])
def test_resume_runtime_refresh_is_explicit_snapshotted_and_restores_state(
    tmp_path, monkeypatch, refresh
):
    service, source, stage = make_hook_run(tmp_path)
    runtime_path = source / "weave_support/runtime.py"
    runtime_path.write_text(runtime_path.read_text() + "\n# previous runtime marker\n")
    execute(source)
    checkpoint = events(source, kind="checkpoint")[-1]
    expected = torch.load(source / checkpoint["payload"]["path"], weights_only=True)
    info = json.loads((source / "run.json").read_text())
    atomic_json(source / "run.json", {**info, "execution": "local"})
    atomic_json(source / "status.json", {"state": "stopped"})
    launch = service.launch

    def prepare(request, **kwargs):
        request.stage.config.execution = "monitor"
        return launch(request, **kwargs)

    monkeypatch.setattr(service, "launch", prepare)
    original_digest = file_hash(runtime_path)
    result = service.restart(source.name, refresh_support=refresh)
    directory = service.run_dir(result["id"])
    revision = json.loads((directory / "revision.json").read_text())
    assert revision["refresh_support"] is refresh
    assert (
        "previous runtime marker"
        in (directory / "weave_support/runtime.py").read_text()
    ) is not refresh
    assert file_hash(runtime_path) == original_digest
    assert (directory / "train.py").read_text() == (source / "train.py").read_text()
    execute(directory)
    last = events(directory, kind="checkpoint")[-1]
    restored = torch.load(directory / last["payload"]["path"], weights_only=True)
    assert restored["step"] == expected["step"]
    assert restored["rank_state"][0]["loader"] == expected["rank_state"][0]["loader"]
    for name, weight in expected["model"].items():
        torch.testing.assert_close(restored["model"][name], weight, rtol=0, atol=0)
