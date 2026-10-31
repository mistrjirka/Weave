"""Real allocation/sharding correctness and bounded benchmark/profile lifecycle."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from backend.training.tuning_math import (
    allocate,
    ddp_candidates,
    split_candidates,
    graph_signature,
)
from backend.training.tuning_runtime import (
    WeightedBatchSampler,
    tune_loader,
    load_tuning,
)
from backend.training.gpu_tuning import GPUTuning, matching_profile, TuningRequest
from backend.training.service import TrainingConfig
from test_training import tiny_project


def test_ratio_search_preserves_global_batch_and_starts_with_standalone_estimate():
    assert allocate(100, [45, 55]) == [45, 55]
    candidates = ddp_candidates(32, [3, 1], 8)
    assert candidates[0][0] > candidates[0][1]
    assert all(sum(sizes) == 32 and min(sizes) >= 1 for sizes in candidates)
    assert len(candidates) == len(set(candidates)) <= 8
    with pytest.raises(ValueError):
        allocate(1, [1, 1])
    with pytest.raises(ValueError):
        allocate(8, [float("nan"), 1])
    plans = split_candidates(["a", "b", "c", "d"], [1, 8, 4, 1], [0, 1], [3, 1])
    assert plans and all(
        set(plan) == {"a", "b", "c", "d"} and set(plan.values()) == {0, 1}
        for plan in plans
    )


def test_ddp_search_expands_past_duplicate_equal_and_nearby_allocations():
    # These measured speeds round to 8/8. The old search stopped at 7/9, 9/7.
    speeds = [34717, 39460]
    assert ddp_candidates(16, speeds, 4) == [
        (8, 8),
        (7, 9),
        (9, 7),
        (6, 10),
    ]
    assert ddp_candidates(16, speeds, 5)[-1] == (10, 6)
    assert ddp_candidates(16, speeds, 0) == []


@pytest.mark.parametrize("speeds", [[1, 1], [3, 1], [1, 20]])
def test_two_gpu_search_covers_every_legal_split_when_budget_allows(speeds):
    for total in range(2, 21):
        candidates = ddp_candidates(total, speeds, 100)
        assert len(candidates) == total - 1
        assert set(candidates) == {(n, total - n) for n in range(1, total)}
        assert candidates[0] == tuple(allocate(total, speeds))


def test_graph_identity_ignores_canvas_movement_but_not_computation():
    original = tiny_project()
    moved = copy.deepcopy(original)
    moved["graph"]["nodes"][0]["position"] = {"x": 123, "y": 456}
    moved["name"] = "Renamed project"
    assert graph_signature(original) == graph_signature(moved)
    moved["config"]["width"] *= 2
    assert graph_signature(original) != graph_signature(moved)


@pytest.mark.parametrize("drop_last", [True, False])
def test_unequal_rank_batches_have_matched_updates_and_correct_record_coverage(
    drop_last,
):
    samplers = [
        WeightedBatchSampler(23, [3, 7], rank, shuffle=False, drop_last=drop_last)
        for rank in range(2)
    ]
    rows = [list(sampler) for sampler in samplers]
    assert len(rows[0]) == len(rows[1]) == (2 if drop_last else 3)
    for first, second in zip(*rows):
        assert len(first) > 0 and len(second) > 0
        assert not set(first) & set(second)
    flattened = [index for a, b in zip(*rows) for index in a + b]
    assert flattened[:20] == list(range(20))
    assert (
        flattened == list(range(20))
        if drop_last
        else flattened == list(range(23)) + [0]
    )
    original = list(WeightedBatchSampler(40, [3, 7], 0, seed=7))
    shuffled = WeightedBatchSampler(40, [3, 7], 0, seed=7)
    shuffled.set_epoch(1)
    assert list(shuffled) != original


def test_only_standard_loader_is_rebatched_and_custom_loader_can_opt_in():
    data = list(range(20))
    cfg = SimpleNamespace(rank_micro_batch_sizes=[1, 3], micro_batch_size=2)
    sampler = DistributedSampler(data, num_replicas=2, rank=1, shuffle=False)
    loader = DataLoader(data, batch_size=2, sampler=sampler)
    tuned, applied = tune_loader(loader, cfg, 1)
    assert applied and len(tuned) == 5
    assert [batch.tolist() for batch in tuned][:2] == [[1, 2, 3], [5, 6, 7]]
    custom = SimpleNamespace(weave_tuning_applied=True)
    assert tune_loader(custom, cfg, 1) == (custom, True)
    unsupported = [torch.zeros(4, 8)]
    assert tune_loader(unsupported, cfg, 0) == (unsupported, False)


def test_saved_profile_autoload_and_runtime_reject_stale_hardware(
    tmp_path, monkeypatch
):
    from backend.training import tuning_runtime

    project = tiny_project()
    gpu = {"torch": "test", "cuda": "test", "devices": []}
    directory = tmp_path / "gpu-tuning/tune-test"
    directory.mkdir(parents=True)
    (directory / "request.json").write_text("{}")
    (directory / "status.json").write_text(
        json.dumps({"state": "completed", "results": [], "message": "done"})
    )
    profile = {
        "id": "tune-test",
        "model_signature": graph_signature(project),
        "created": 1,
        "model_name": "test",
        "python": "/test/python",
        "devices": [0, 1],
        "hardware": gpu,
        "sequence_length": 16,
        "global_batch_size": 4,
        "best": {
            "ddp": {
                "precision": "fp32",
                "batch_sizes": [1, 3],
                "placement": {},
                "step_seconds": 0.2,
            }
        },
    }
    (directory / "profile.json").write_text(json.dumps(profile))
    config = TrainingConfig(
        strategy="ddp", python="/test/python", devices=[0, 1], batch_size=2
    )
    selected = matching_profile(tmp_path, project, config)
    assert selected["batch_sizes"] == {"0": 1, "1": 3}
    (tmp_path / "project.json").write_text(json.dumps(project))
    runtime = SimpleNamespace(
        use_gpu_tuning=True,
        devices=[0, 1],
        precision="fp32",
        resolved_strategy="ddp",
        micro_batch_size=2,
        placement={},
    )
    monkeypatch.setattr(tuning_runtime, "hardware_identity", lambda devices: gpu)
    assert load_tuning(runtime, {"gpu_tuning": selected}, tmp_path)[0]
    assert runtime.rank_micro_batch_sizes == [1, 3]
    monkeypatch.setattr(
        tuning_runtime, "hardware_identity", lambda devices: {**gpu, "torch": "changed"}
    )
    assert load_tuning(runtime, {"gpu_tuning": selected}, tmp_path)[0] is None
    GPUTuning(tmp_path).configure(False)
    assert matching_profile(tmp_path, project, config) is None


def test_bad_benchmark_request_rejected_before_export_or_launch(tmp_path):
    tuning = GPUTuning(tmp_path)
    request = TuningRequest(
        project=tiny_project(), python="/missing/python", devices=[0, 0]
    )
    with pytest.raises(ValueError, match="unique"):
        tuning.start(request)
    request.devices = [0, 1]
    request.global_batch_size = 1
    with pytest.raises(ValueError, match="at least one"):
        tuning.start(request)
    request.global_batch_size = 4
    with pytest.raises(ValueError, match="executable"):
        tuning.start(request)
