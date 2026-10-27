import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
import torch
from backend.core import validate_project
from backend.compiler import compile_project, export_zip
from backend.templates import templates, node, edge
from backend.runtime import Attention, NGram, EngramHash, Monarch, Routing, Functional

ROOT=Path(__file__).parent.parent


def small(project):
    p=copy.deepcopy(project);p['config'].update(width=32,heads=4,ffn_width=64,vocab=64)
    def visit(g):
        for n in g['nodes']:
            k=n['data']['kind'];q=n['data']['params']
            if k in ('attention','cross_attention'):q.update(heads=4,kv_heads=2,head_dim=8,value_dim=8)
            if k in ('engram_hash','engram_lookup'):q.update(hash_heads=1,buckets=64,sub_dim=16)
            if k=='scale':q['factor']=32**.5
            if n['data'].get('body'):visit(n['data']['body'])
    visit(p['graph']);return p


@pytest.mark.parametrize('index',range(8))
def test_export_forward_and_parameter_accounting(index,tmp_path):
    project=small(templates()[index]);report=validate_project(project)
    assert report['valid'],report['errors']
    for name,content in compile_project(project).items():
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
    script='''
import torch,json
from model import Model
torch.set_num_threads(1)
model=Model().eval()
args=[torch.zeros((1,8),dtype=torch.long)]*INPUTS
with torch.no_grad(): y=model(*args)
assert torch.isfinite(y).all()
print(json.dumps({'shape':list(y.shape),'parameters':sum(p.numel() for p in model.parameters())}))
'''.replace('INPUTS',str(sum(n['data']['kind']=='input' for n in project['graph']['nodes'])))
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    runtime=json.loads(result.stdout)
    assert runtime['parameters']==report['parameters']
    assert runtime['shape'][:2]==[1,24 if index in (0,3) else 8]


def test_shape_error_points_to_residual_port():
    p=templates()[0];body=p['graph']['nodes'][2]['data']['body']['nodes'][1]['data']['body']
    next(n for n in body['nodes'] if n['id']=='memory')['data']['params']['width']=768
    report=validate_project(p)
    assert not report['valid']
    assert any(e['node']=='denoise/predict/fusion' and '768' in e['message'] for e in report['errors'])
    with pytest.raises(ValueError):compile_project(p)


def test_loop_rejects_state_shape_change():
    p=templates()[0];body=p['graph']['nodes'][2]['data']['body']
    body['edges'][-1]['source']='predict'
    assert any('Loop state' in e['message'] for e in validate_project(p)['errors'])


def test_explicit_sharing_inside_independent_layers():
    p=templates()[1]
    body=p['graph']['nodes'][2]['data']['body']
    for n in body['nodes']:
        if n['data']['kind']=='norm':n['data']['share']='all_norms'
    shared=validate_project(p)
    assert shared['valid'],shared['errors']
    # Two layer norms, six layers; a single explicit norm module replaces all twelve.
    expected=validate_project(templates()[1])['parameters']-11*128
    assert shared['parameters']==expected


def test_sharing_rejects_different_feature_widths():
    p={'version':1,'config':{'width':32,'vocab':64},'graph':{'nodes':[
        node('x','input','X',0,0,{'shape':['B','T',32],'dtype':'float32'}),
        node('norm1','norm','Norm',300,0,{},share='n'),
        node('proj','linear','Project',600,0,{'out':64}),
        node('norm2','norm','Norm',900,0,{},share='n'),node('out','output','Out',1200,0)],
        'edges':[edge('x','norm1'),edge('norm1','proj'),edge('proj','norm2'),edge('norm2','out')]}}
    assert any('Shared module' in e['message'] for e in validate_project(p)['errors'])


def test_cycle_and_ambiguous_connection():
    p=templates()[1];p['graph']['edges'].append(edge('head','emb'))
    assert any('cycle' in e['message'] for e in validate_project(p)['errors'])
    p=templates()[1];p['graph']['edges'].append(edge('ids','head'))
    assert any('multiple connections' in e['message'] for e in validate_project(p)['errors'])


def test_hash_boundaries_and_engram_upstream_formula():
    import numpy as np
    ids=torch.tensor([[1,2,3]])
    gram=NGram([2],101)(ids)
    assert gram[0,:,0].tolist()==[2,(2*257+3)%101,(3*257+4)%101]
    actual=EngramHash([2,3],2,8192)(ids).numpy()
    u=ids.numpy().astype(np.uint32);expected=[]
    with np.errstate(over='ignore'):
        for oi,order in enumerate([2,3]):
            for head in range(2):
                acc=np.full_like(u,(0x9E3779B9*(oi*2+head+1))&0xffffffff)
                for j in range(order):
                    shifted=np.pad(u,((0,0),(j,0)))[:,:u.shape[1]]
                    acc=(acc^shifted)*np.uint32(0x01000193)
                expected.append((acc^(acc>>15))%8192)
    assert np.array_equal(actual,np.stack(expected,-1))


def test_routing_is_doubly_stochastic():
    routing=Routing(16,4,2)(torch.randn(2,5,4,16))
    matrix=routing[...,8:].reshape(2,5,4,4)
    # The upstream recipe uses a fixed 20 iterations, so row sums are approximate.
    assert torch.allclose(matrix.sum(-1),torch.ones(2,5,4),atol=5e-4)
    assert torch.allclose(matrix.sum(-2),torch.ones(2,5,4),atol=1e-5)


def test_monarch_matches_upstream_numpy_equations():
    import numpy as np
    torch.manual_seed(4);m=Monarch(12);x=torch.randn(1,3,12)
    actual=m(x).detach().numpy()
    cv,cu=m.cv.detach().numpy(),m.cu.detach().numpy()
    h=x.numpy()@cv;h=np.exp(h-h.max(-1,keepdims=True));cond=1+(h/h.sum(-1,keepdims=True))@cu
    def kron(z,i):
        a,b=[v.detach().numpy() for v in m.factors[2*i:2*i+2]]
        return np.einsum('...ij,ik,jl->...kl',z.reshape(*z.shape[:-1],a.shape[0],b.shape[0]),a,b).reshape(*z.shape[:-1],m.n)
    d=[v.detach().numpy() for v in m.d]
    z=np.pad(x.numpy(),((0,0),(0,0),(0,m.n-m.width)))
    z=kron(d[0]*z,0)[...,m.perm0.numpy()]
    z=d[1]*cond*z+m.bias.detach().numpy();z=z/(1+np.exp(-z))
    z=kron(z,1)[...,m.perm1.numpy()]
    expected=(d[3]*kron(d[2]*z,2))[...,:m.width]
    assert np.allclose(actual,expected,rtol=1e-5,atol=1e-6)


def test_attention_matches_thimble_source():
    spec=importlib.util.spec_from_file_location('upstream_thimble',ROOT/'references/thimble_model.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    cfg=mod.Config(d_model=32,n_heads=4,n_kv=2)
    upstream=mod.Attn(cfg);ours=Attention(32,4,2,qk_norm=True,gated=True)
    ours.load_state_dict(upstream.state_dict())
    x=torch.randn(2,7,32)
    inv=1/(10000**(torch.arange(0,8,2).float()/8));freq=torch.outer(torch.arange(7),inv)
    cos=torch.cat((freq.cos(),freq.cos()),-1)[None,None]
    sin=torch.cat((freq.sin(),freq.sin()),-1)[None,None]
    assert torch.allclose(ours(x),upstream(x,cos,sin),rtol=1e-5,atol=1e-6)


def test_expanded_attention_preserves_parameters_and_behavior(tmp_path):
    from backend.attention import expand_attention
    p=templates()[1];p=small(p)
    body=p['graph']['nodes'][2]['data']['body'];attn=next(n for n in body['nodes'] if n['id']=='attn')
    settings={'heads':4,'kv_heads':2,'head_dim':8,'value_dim':8,'causal':True,'qk_norm':True,'gated':True,'rope':True,'conv_taps':3,'window':4}
    project={'version':1,'name':'Expanded attention','config':{'width':32,'heads':4,'vocab':64},'graph':expand_attention(settings,32)}
    for name,content in compile_project(project).items():
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
    script='''
import torch
from model import Model
from runtime import Attention
torch.set_num_threads(1)
model=Model();primitive=Attention(32,4,2,8,8,True,4,True,10000.,True,True,3)
# Match the named layers exposed by the exported PyTorch module.
ops = {
 'q': model.q_projection, 'k': model.k_projection, 'v': model.v_projection,
 'o': model.output_projection,
 'qnorm': model.q_rmsnorm, 'knorm': model.k_rmsnorm,
 'gateproj': model.output_gate_projection,
 'qconv': model.q_causal_convolution, 'kconv': model.k_causal_convolution,
 'vconv': model.v_causal_convolution,
}
for key in ['q','k','v','o']:ops[key].load_state_dict(getattr(primitive,key).state_dict())
for key in ['q','k']:ops[key+'norm'].load_state_dict(getattr(primitive,key+'_norm').state_dict())
ops['gateproj'].load_state_dict(primitive.gate.state_dict())
for i,key in enumerate(['q','k','v']):ops[key+'conv'].taps.data.copy_(primitive.taps[i])
x=torch.randn(2,7,32)
assert torch.allclose(model(x),primitive(x),atol=1e-6,rtol=1e-5)
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr


def test_diffusion_preserves_prompt_and_resolves_masks():
    op=Functional('select',{'strategy':'confidence'},{'mask_id':3})
    ids=torch.tensor([[1,2,3,3,3,3]])
    logits=torch.randn(1,6,16)
    for step in range(3):ids=op(logits,ids,step=step,steps=3)
    assert ids[0,:2].tolist()==[1,2]
    assert not ids.eq(3).any()


def test_custom_python_never_runs_in_compiler(tmp_path):
    p=templates()[1]
    n=p['graph']['nodes'][2]
    n['data']={'kind':'custom','label':'Custom','params':{'input_shape':['B','T','$width'],'output_shape':['B','T','$width'],'source':f'raise RuntimeError("DO NOT EXECUTE")\nclass CustomBlock(nn.Module):\n    pass'}}
    files=compile_project(p)
    assert any('DO NOT EXECUTE' in v for k,v in files.items() if k.startswith('custom_'))


def test_tokenizer_capacity_and_bundle(tmp_path):
    from tokenizers import Tokenizer,models,pre_tokenizers
    from backend.tokenizers import TokenizerService
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1,'[MASK]':2},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    path=tmp_path/'tokenizer.json';tokenizer.save(str(path))
    service=TokenizerService(tmp_path/'cache');info=service.load(path)
    assert info['capacity']==3 and service.preview(info['id'],'hello')['ids']==[1]
    p=templates()[1];p['tokenizer']={**info,'capacity':2048}
    assert not validate_project(p)['valid']
    p['tokenizer']=info
    archive=zipfile.ZipFile(io.BytesIO(export_zip(p,tmp_path/'cache'/info['id'])))
    assert 'tokenizer/tokenizer.json' in archive.namelist()


def test_api_persistence_validation_export_and_tokenizer_upload(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from backend import app as module
    monkeypatch.setattr(module,'DATA',tmp_path)
    client=TestClient(module.app);p=templates()[0]
    assert client.put('/api/projects/test',json=p).status_code==200
    assert client.get('/api/projects/test').json()['name']==p['name']
    assert client.post('/api/validate',json=p).json()['valid']
    assert client.post('/api/export',json=p).content.startswith(b'PK')
    assert client.delete('/api/projects/test').json()['deleted']
    assert client.get('/api/projects/test').status_code==404
