import copy
import sys
import pytest
import torch
from backend import runtime, export_helpers, sparse_runtime
from backend.core import validate_project
from backend.compiler import compile_project
from backend.templates import templates,node,edge
from backend.operations import unroll_loop
from backend.sparse_runtime import AttentionPattern,attend
from backend.attention import expand_attention


def build(project,monkeypatch):
    report=validate_project(project)
    assert report['valid'],report['errors']
    monkeypatch.setitem(sys.modules,'runtime',runtime)
    monkeypatch.setitem(sys.modules,'export_helpers',export_helpers)
    monkeypatch.setitem(sys.modules,'sparse_runtime',sparse_runtime)
    namespace={};exec(compile_project(project)['model.py'],namespace)
    return namespace['Model']().eval()


def compact_loop(shared=True):
    body={'nodes':[node('x','input','X',0,0,{'shape':['B','T',8],'dtype':'float32','index':0}),node('linear','linear','Learned',300,0,{'out':8}),node('out','output','Out',600,0)],'edges':[edge('x','linear'),edge('linear','out')]}
    return {'version':1,'id':'test','name':'Test','config':{'width':8,'heads':2,'ffn_width':16,'vocab':16},'graph':{'nodes':[node('x','input','X',0,0,{'shape':['B','T',8],'dtype':'float32'}),node('loop','loop','Repeat',300,0,{'iterations':3,'shared':shared},body),node('out','output','Out',600,0)],'edges':[edge('x','loop'),edge('loop','out')]}}


@pytest.mark.parametrize('shared',[True,False])
def test_unroll_preserves_computation_and_parameter_accounting(shared,monkeypatch):
    p=compact_loop(shared);expanded=copy.deepcopy(p);expanded['graph']=unroll_loop(p['graph'],'loop',p['config'])['graph']
    torch.manual_seed(7);a=build(p,monkeypatch)
    torch.manual_seed(7);b=build(expanded,monkeypatch)
    x=torch.randn(2,5,8)
    torch.testing.assert_close(a(x),b(x))
    assert validate_project(p)['parameters']==validate_project(expanded)['parameters']==sum(v.numel() for v in b.parameters())
    applications=list(b._modules.values())
    assert (applications[0] is applications[1])==shared
    assert (applications[1] is applications[2])==shared


def test_independent_layer_between_shared_applications_has_own_gradients(monkeypatch):
    p=compact_loop();p['graph']=unroll_loop(p['graph'],'loop',p['config'])['graph']
    apps=[n for n in p['graph']['nodes'] if n['data']['kind']=='block']
    first,second=apps[:2];g=p['graph'];wire=next(e for e in g['edges'] if e['source']==first['id'] and e['target']==second['id']);g['edges'].remove(wire)
    g['nodes'].append(node('adapter','linear','Independent',0,0,{'out':8}))
    g['edges'].extend([edge(first['id'],'adapter'),edge('adapter',second['id'])])
    model=build(p,monkeypatch);model(torch.randn(2,5,8)).sum().backward()
    assert sum(v.numel() for v in model.parameters())==2*8*8
    modules=list(model._modules.values());shared=[m for m in modules if not isinstance(m,torch.nn.Linear)];adapter=next(m for m in modules if isinstance(m,torch.nn.Linear))
    assert shared[0] is shared[1] is shared[2]
    assert adapter.weight.grad is not None
    assert next(shared[0].parameters()).grad is not None
    assert adapter.weight is not next(shared[0].parameters())


def test_sparse_patterns_compose_and_enforce_causality():
    torch.manual_seed(1);q=torch.randn(2,3,8,4);k=torch.randn_like(q);v=torch.randn_like(q)
    local=AttentionPattern('local',q,k,{'window':2});global_=AttentionPattern('global',q,k,{'global_tokens':[0]})
    union=AttentionPattern('union',q,k,children=[local,global_]);intersection=AttentionPattern('intersection',q,k,children=[union,AttentionPattern('causal',q,k)])
    mask=intersection.dense();assert mask[7,0] and mask[7,6] and not mask[7,3] and not mask[0,7]
    actual=attend(q,k,v,union,causal=True)
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    torch.testing.assert_close(actual,expected)
    a=AttentionPattern('random',q,k,{'density':.3,'seed':42}).dense();b=AttentionPattern('random',q,k,{'density':.3,'seed':42}).dense()
    assert torch.equal(a,b)
    four=mask.expand(2,3,8,8);torch.testing.assert_close(attend(q,k,v,four),expected)


def test_content_topk_is_per_batch_per_head_and_causal():
    q=torch.randn(2,3,8,4);mask=AttentionPattern('content_topk',q,p={'top_k':2,'causal':True}).dense(2,3)
    assert mask.shape==(2,3,8,8)
    assert mask.sum(-1).max()==2
    assert not mask.triu(1).any()
    assert mask[:,:,0,:].sum()==6


def test_expanding_attention_preserves_external_mask(monkeypatch):
    p=compact_loop();g=p['graph'];g['nodes']=[g['nodes'][0],node('attn','attention','Attention',300,0,{'heads':2,'kv_heads':1,'rope':True,'causal':True}),node('pattern','attention_pattern','Local',100,300,{'pattern':'local','window':2}),g['nodes'][-1]];g['edges']=[edge('x','attn'),edge('x','pattern'),edge('pattern','attn','b'),edge('attn','out')]
    compact=build(p,monkeypatch);expanded=copy.deepcopy(p);n=next(n for n in expanded['graph']['nodes'] if n['id']=='attn');n['data'].update(kind='block',body=expand_attention(n['data']['params'],8),params={})
    lowered=build(expanded,monkeypatch)
    # Copy parameter tensors by computation branch; generated traversal order differs.
    a=next(m for m in compact.children() if isinstance(m,runtime.Attention));b=next(lowered.children())
    for key, attribute in [('q','q_projection'),('k','k_projection'),('v','v_projection'),('o','output_projection')]:
        getattr(b,attribute).load_state_dict(getattr(a,key).state_dict())
    x=torch.randn(2,8,8);torch.testing.assert_close(compact(x),lowered(x))


def test_custom_masks_reject_sequence_mismatch_and_flex_has_cpu_message():
    p=compact_loop();g=p['graph'];g['nodes'].insert(1,node('pattern','attention_pattern','Custom',0,0,{'pattern':'custom','matrix':[[True]*4]*4}));g['edges'].append(edge('x','pattern'))
    assert any('fixed query/key lengths' in e['message'] for e in validate_project(p)['errors'])
    q=torch.randn(1,2,4,4)
    with pytest.raises(RuntimeError,match='GPU'):attend(q,q,q,backend='flex')


def test_sparse_and_explicit_presets_export(monkeypatch):
    for id in ('sparse','looped'):
        p=next(t for t in templates() if t['id']==id);model=build(p,monkeypatch)
        with torch.no_grad():result=model(torch.zeros(1,5,dtype=torch.long))
        assert result.shape==(1,5,p['config']['vocab'])
        assert validate_project(p)['parameters']==sum(v.numel() for v in model.parameters())


def test_diffusion_unroll_preserves_sampling_schedule(monkeypatch):
    p=next(t for t in templates() if t['id']=='diffusion');p['config'].update(width=16,heads=4,ffn_width=32,vocab=32)
    def shrink(g):
        for n in g['nodes']:
            if n['data']['kind'] in ('attention','cross_attention'):n['data']['params'].update(heads=4,kv_heads=2,head_dim=4,value_dim=4)
            if n['data'].get('body'):shrink(n['data']['body'])
    shrink(p['graph']);p['graph']['nodes'][2]['data']['params']['iterations']=3
    expanded=copy.deepcopy(p);expanded['graph']=unroll_loop(p['graph'],'denoise',p['config'])['graph']
    torch.manual_seed(5);compact=build(p,monkeypatch)
    torch.manual_seed(5);unrolled=build(expanded,monkeypatch)
    x=torch.tensor([[1,2]])
    with torch.no_grad():assert torch.equal(compact(x),unrolled(x))
    assert validate_project(p)['parameters']==validate_project(expanded)['parameters']
