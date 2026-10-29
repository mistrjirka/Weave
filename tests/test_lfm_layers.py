import copy
import importlib.util
import torch
from torch.nn import functional as F
from backend.templates import templates
from backend.core import validate_project
from backend.compiler import compile_project
from backend.runtime import CausalConv, Attention


def test_causal_convolution_matches_depthwise_pytorch_and_gradient():
    torch.manual_seed(17)
    x = torch.randn(2, 7, 4, requires_grad=True)
    layer = CausalConv(4, 3)
    with torch.no_grad(): layer.taps.copy_(torch.randn_like(layer.taps))
    actual = layer(x)
    weights = layer.taps.flip(0).T.unsqueeze(1)
    expected = F.conv1d(F.pad(x.transpose(1, 2), (2, 0)), weights, groups=4).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0], torch.autograd.grad(expected.sum(), x)[0])
    changed = x.detach().clone(); changed[:, 4:] += 10
    torch.testing.assert_close(layer(changed)[:, :4], actual[:, :4])


def test_grouped_attention_matches_explicit_head_reference():
    torch.manual_seed(12)
    layer = Attention(16, 4, 2, theta=1_000_000, qk_norm=True, qk_norm_eps=1e-5)
    x = torch.randn(2, 5, 16, requires_grad=True)
    actual = layer(x)
    q = layer.rotate(layer.q_norm(layer.q(x).view(2, 5, 4, 4)).transpose(1, 2))
    k = layer.rotate(layer.k_norm(layer.k(x).view(2, 5, 2, 4)).transpose(1, 2)).repeat_interleave(2, 1)
    v = layer.v(x).view(2, 5, 2, 4).transpose(1, 2).repeat_interleave(2, 1)
    scores = (q @ k.transpose(-2, -1)) / 2
    mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
    expected = layer.o((scores.masked_fill(mask, float('-inf')).softmax(-1) @ v).transpose(1, 2).contiguous().view(2, 5, 16))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_shellper_export_runs_forward_and_backward(tmp_path, monkeypatch):
    project = copy.deepcopy(next(p for p in templates() if p['id'] == 'lfm-shellper-350m'))
    project['config'].update(width=16, heads=4, ffn_width=32, vocab=32)
    for layer in project['graph']['nodes']:
        for node in layer['data'].get('body', {}).get('nodes', []):
            if node['data']['kind'] == 'attention': node['data']['params'].update(heads=4, kv_heads=2)
    report = validate_project(project)
    assert report['valid'], report['errors']
    for name, source in compile_project(project).items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location('lfm_test_model', tmp_path / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.Model()
    output = model(torch.randint(0, 32, (2, 5)))
    assert output.shape == (2, 5, 32)
    assert sum(p.numel() for p in model.parameters()) == report['parameters']
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
