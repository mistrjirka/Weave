from types import SimpleNamespace

from backend.training.environment import inspect_python, _inspect_cached


def test_python_probe_reuses_checks_and_invalidates_changed_interpreters(
    tmp_path, monkeypatch
):
    executable = tmp_path / "python"
    executable.write_text("interpreter")
    calls = []
    monkeypatch.setattr("backend.training.environment.time.monotonic", lambda: 120)

    def probe(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='{"python_version":"3.12","device_count":2}')

    monkeypatch.setattr("backend.training.environment.subprocess.run", probe)
    _inspect_cached.cache_clear()
    assert inspect_python(str(executable))["device_count"] == 2
    inspect_python(str(executable))
    assert len(calls) == 1
    executable.write_text("updated interpreter")
    inspect_python(str(executable))
    assert len(calls) == 2
    monkeypatch.setattr("backend.training.environment.time.monotonic", lambda: 181)
    inspect_python(str(executable))
    assert len(calls) == 3
    _inspect_cached.cache_clear()
