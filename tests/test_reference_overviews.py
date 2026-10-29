from backend.templates import templates
from backend.core import validate_project
from backend.compiler import compile_project
import pytest


def test_reference_schedules_and_export_guard():
    examples = {project['id']: project for project in templates()}
    hammer = examples['hammer21-05b']
    shell = examples['lfm-shellper-350m']
    assert len([n for n in hammer['graph']['nodes'] if n['data']['kind'] == 'block']) == 24
    layers = [n for n in shell['graph']['nodes'] if n['data']['kind'] == 'block']
    assert len(layers) == 16
    assert sum('Convolution' in n['data']['label'] for n in layers) == 10
    assert any('embedded CPUs' in n['data'].get('annotation', '') and n['data'].get('annotationSource') for n in layers)
    assert validate_project(shell)['valid']
    assert validate_project(hammer)['valid']


def test_published_thimble_trunk_and_scale_have_notes():
    examples = {project['id']: project for project in templates()}
    assert 'v6' in examples['thimble']['description']
    assert next(n for n in examples['needle3']['graph']['nodes'] if n['id'] == 'scale')['data']['annotation']
