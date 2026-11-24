import hashlib
import json
import sqlite3
import sys
import types
import zipfile

from backend.training import service, supervisor


def setup_run(tmp_path, monkeypatch, *, running):
    (tmp_path / 'config.json').write_text(json.dumps({'strategy': 'single', 'devices': [], 'recipe': 'custom'}))
    (tmp_path / 'train.py').write_text('pass\n')
    (tmp_path / 'commands').mkdir()
    (tmp_path / 'commands' / 'snapshot.json').write_text(json.dumps({'id': 's1', 'kind': 'snapshot', 'include_data': False}))
    (tmp_path / 'checkpoints').mkdir()
    weights = b'actual checkpoint weights'
    (tmp_path / 'checkpoints' / 'model.bin').write_bytes(weights)
    checkpoint = {'version': 1, 'id': 'checkpoint', 'kind': 'checkpoint', 'time': 1, 'step': 1, 'tokens': 1, 'channel': 'training', 'payload': {'path': 'checkpoints/model.bin', 'sha256': hashlib.sha256(weights).hexdigest(), 'metadata': {'snapshot_requests': ['s1']}}}
    (tmp_path / 'events.jsonl').write_text('')
    service.events(tmp_path)
    class Worker:
        pid = 99999999
        returncode = None if running else 0
        def poll(self):
            return self.returncode
    worker = Worker()
    class Reporter:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(supervisor, 'Run', lambda *args: Reporter())
    monkeypatch.setattr(supervisor, 'hardware', lambda: [])
    monkeypatch.setattr(supervisor, 'is_code_first', lambda _: True)
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(stdout=json.dumps({'packages': {}, 'cuda': None, 'cuda_available': False})))
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: worker)
    monkeypatch.setitem(sys.modules, 'psutil', None)
    monkeypatch.setattr(sys, 'argv', ['supervisor', str(tmp_path), sys.executable])
    return worker, checkpoint, weights


def append(path, record):
    with (path / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


def check_snapshot(path, weights):
    with zipfile.ZipFile(path / 'snapshots' / 'snapshot-s1.zip') as archive:
        assert archive.read('checkpoints/model.bin') == weights
        assert json.loads(archive.read('manifest.json'))['resumable_weights'] is True
    assert json.loads((path / 'status.json').read_text())['state'] == 'completed'


def test_snapshot_internal_index_busy_retries(tmp_path, monkeypatch):
    worker, checkpoint, weights = setup_run(tmp_path, monkeypatch, running=True)
    append(tmp_path, checkpoint)
    real_events = service.events
    locks = []
    def blocked_events(directory, **kwargs):
        if kwargs.get('timeout') == 0.2 and 'kind' not in kwargs and not locks:
            # New complete data makes the internal query require a writer.
            append(tmp_path, dict(checkpoint, id='additional', kind='metrics', payload={}))
            holder = sqlite3.connect(tmp_path / 'event-index.sqlite')
            holder.execute('BEGIN IMMEDIATE')
            locks.append(holder)
        return real_events(directory, **kwargs)
    monkeypatch.setattr(service, 'events', blocked_events)
    sleeps = []
    def tick(_):
        sleeps.append(1)
        assert not (tmp_path / 'snapshots' / 'snapshot-s1.json').exists()
        locks[0].rollback()
        locks[0].close()
        worker.returncode = 0
    monkeypatch.setattr(supervisor.time, 'sleep', tick)
    supervisor.main()
    assert len(sleeps) == 1
    check_snapshot(tmp_path, weights)


def test_checkpoint_index_busy_on_exit_retries_with_weights(tmp_path, monkeypatch):
    _, checkpoint, weights = setup_run(tmp_path, monkeypatch, running=False)
    append(tmp_path, checkpoint)
    holder = sqlite3.connect(tmp_path / 'event-index.sqlite')
    holder.execute('BEGIN IMMEDIATE')
    sleeps = []
    def tick(_):
        sleeps.append(1)
        assert not (tmp_path / 'snapshots' / 'snapshot-s1.json').exists()
        holder.rollback()
        holder.close()
    monkeypatch.setattr(supervisor.time, 'sleep', tick)
    supervisor.main()
    assert len(sleeps) == 1
    check_snapshot(tmp_path, weights)


def test_exit_success_with_persistent_index_lock_is_unknown(tmp_path, monkeypatch):
    _, checkpoint, _ = setup_run(tmp_path, monkeypatch, running=False)
    append(tmp_path, checkpoint)
    holder = sqlite3.connect(tmp_path / 'event-index.sqlite')
    holder.execute('BEGIN IMMEDIATE')
    clock = iter([0, 11])
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    try:
        supervisor.main()
    finally:
        holder.rollback()
        holder.close()
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['state'] == 'unknown'
    assert status['exit_code'] == 0
    assert 'index remained busy' in status['message']
    assert not (tmp_path / 'snapshots' / 'snapshot-s1.json').exists()
