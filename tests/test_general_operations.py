import copy
import importlib.util
import math
import pytest
import torch
from backend.templates import node, edge, templates
from backend.core import validate_project
from backend.compiler import compile_project
from backend.runtime import Attention


def make_project(kind, params, shape):
    return {'version':1,'id':'ops','name':'Ops','description':'','config':{'width':shape[-1], 'heads':2,'ffn_width':16,'vocab':16},'graph':{'nodes':[node('x','input','Input',0,0,{'shape':shape,'index':0,'dtype':'float32'}),node('op',kind,kind,300,0,params),node('y','output','Output',600,0)],'edges':[edge('x','op'),edge('op','y')]}}


@pytest.mark.parametrize('kind,params,shape,outshape',[
 ('gelu',{'approximate':'none'},[2,7,8],[2,7,8]),('relu',{},[2,7,8],[2,7,8]),
 ('layer_norm',{'eps':1e-5,'affine':True,'bias':True},[2,7,8],[2,7,8]),
 ('dropout',{'probability':0.2},[2,7,8],[2,7,8]),
 ('position_embedding',{'max_length':10},[2,7,8],[2,7,8]),
 ('conv1d',{'out':6,'kernel_size':3,'stride':2,'padding':1,'bias':True},[2,7,8],[2,4,6]),
 ('pool1d',{'kernel_size':2,'stride':2,'mode':'average'},[2,8,8],[2,4,8]),
 ('conv2d',{'out_channels':5,'kernel_size':3,'stride':2,'padding':1,'bias':True},[2,3,8,8],[2,5,4,4]),
 ('pool2d',{'kernel_size':2,'stride':2,'mode':'max'},[2,3,8,8],[2,3,4,4]),
])
def test_exported_operations_shape_parameters_and_backward(kind,params,shape,outshape,tmp_path,monkeypatch):
    project=make_project(kind,params,shape); report=validate_project(project)
    assert report['valid'],report['errors']
    assert report['shapes']['op']['shape']==outshape
    for name, text in compile_project(project).items():
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec=importlib.util.spec_from_file_location('ops_model',tmp_path/'model.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.Model(); value=torch.randn(*shape,requires_grad=True); result=model(value)
    assert list(result.shape)==outshape
    assert sum(p.numel() for p in model.parameters())==report['parameters']
    result.sum().backward();assert torch.isfinite(value.grad).all()
    if kind=='dropout':
        model.eval();torch.testing.assert_close(model(value),value)


def test_invalid_shapes_and_position_limit():
    p=make_project('position_embedding',{'max_length':4},[2,7,8]);assert not validate_project(p)['valid']
    p=make_project('pool1d',{'kernel_size':2,'stride':2},[2,'T',8]);assert not validate_project(p)['valid']
    p=make_project('conv2d',{'kernel_size':9},[2,3,4,4]);assert not validate_project(p)['valid']


def test_yarn_and_qkv_bias_against_independent_formula():
    layer=Attention(64,4,2,qkv_bias=True,rope_scaling='yarn',rope_factor=4,rope_original_context=32768,theta=1_000_000)
    assert layer.q.bias is not None and layer.k.bias is not None and layer.v.bias is not None and layer.o.bias is None
    x=torch.randn(1,4,5,16)
    dim=16;theta=1_000_000;factor=4;context=32768
    correction=lambda n: dim*math.log(context/(n*2*math.pi))/(2*math.log(theta))
    low=max(math.floor(correction(32)),0);high=min(math.ceil(correction(1)),dim-1)
    if low==high:high+=0.001
    ramp=((torch.arange(dim//2)-low)/(high-low)).clamp(0,1)
    inv=theta**(-torch.arange(0,dim,2).float()/dim)*((1-ramp)+ramp/factor)
    angles=torch.outer(torch.arange(5).float(),inv);angles=torch.cat((angles,angles),-1)
    a,b=x.chunk(2,-1); expected=(x*angles.cos()+torch.cat((-b,a),-1)*angles.sin())*(1+0.1*math.log(4))
    torch.testing.assert_close(layer.rotate(x),expected)


def test_gpt2_reference_export_and_dropout(tmp_path,monkeypatch):
    p=copy.deepcopy(next(p for p in templates() if p['id']=='gpt2'))
    assert validate_project(p)['parameters']==124439808
    p['config'].update(width=16,heads=4,ffn_width=32,vocab=32)
    for layer in p['graph']['nodes']:
        if layer['id']=='pos': layer['data']['params']['max_length']=16
        for child in layer['data'].get('body',{}).get('nodes',[]):
            if child['data']['kind']=='attention':child['data']['params'].update(heads=4,kv_heads=4)
    report=validate_project(p);assert report['valid'],report['errors']
    for name,source in compile_project(p).items():
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec=importlib.util.spec_from_file_location('gpt2_test_model',tmp_path/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.Model();assert sum(v.numel() for v in model.parameters())==report['parameters']
    ids=torch.randint(0,32,(1,6));model.eval()
    a=model(ids);torch.testing.assert_close(a,model(ids));assert a.shape==(1,6,32)
    model.train(); b=model(ids); assert not torch.equal(a,b)
    b.square().mean().backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())


def test_attention_dropout_train_eval_behavior():
    layer=Attention(16,4,2,attention_dropout=0.5)
    x=torch.randn(1,6,16)
    layer.eval();a=layer(x);torch.testing.assert_close(a,layer(x))
    layer.train();assert not torch.equal(layer(x),layer(x))
