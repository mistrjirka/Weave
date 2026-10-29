"""Weight-matched comparisons against published upstream model implementations."""
import copy
import os
import importlib.util
import sys
from pathlib import Path
import torch
import pytest

# The tokenizer service intentionally disables ML runtimes; load Transformers
# with PyTorch during test collection before that service is imported.
os.environ["USE_TORCH"] = "1"
import transformers

from backend.templates import templates
from test_codegen import instantiate, reduced

torch.set_num_threads(1)


def load_thimble():
    path=Path(__file__).resolve().parents[1]/'references'/'thimble_model.py'
    spec=importlib.util.spec_from_file_location('weave_upstream_thimble',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def test_thimble_full_trunk_matches_upstream(monkeypatch):
    upstream=load_thimble();cfg=upstream.Config(vocab_size=32,d_model=16,n_layers=2,n_heads=4,n_kv=2,ffn_mult=2.)
    source=upstream.ToolTransformer(cfg).eval()
    project=reduced(next(p for p in templates() if p['id']=='thimble'))
    next(n for n in project['graph']['nodes'] if n['id']=='stack')['data']['params']['iterations']=2
    actual,_=instantiate(project,monkeypatch);actual.eval()
    with torch.no_grad():
        actual.token_embedding.weight.copy_(source.embed.weight)
        actual.final_rmsnorm.weight.copy_(source.norm.weight)
        for ours,theirs in zip(actual.transformer_layers,source.blocks):
            for a,b in [(ours.pre_attention_norm,theirs.n1),(ours.post_attention_norm,theirs.n2),(ours.pre_ffn_norm,theirs.n3),(ours.post_ffn_norm,theirs.n4)]:a.weight.copy_(b.weight)
            ours.grouped_query_attention.load_state_dict(theirs.attn.state_dict())
            ours.swi_glu_feed_forward.gate.weight.copy_(theirs.w1.weight)
            ours.swi_glu_feed_forward.down.weight.copy_(theirs.w2.weight)
            ours.swi_glu_feed_forward.up.weight.copy_(theirs.w3.weight)
    ids=torch.tensor([[1,2,3,4,5],[5,3,7,1,0]])
    with torch.no_grad():
        expected,_=source(ids)
        torch.testing.assert_close(actual(ids),expected,rtol=2e-5,atol=2e-6)


def attention_copy(dst, src, qnorm=False):
    for a,b in [(dst.q,src.q_proj),(dst.k,src.k_proj),(dst.v,src.v_proj),(dst.o,getattr(src,'o_proj',getattr(src,'out_proj',None)))]:
        a.weight.copy_(b.weight)
        if a.bias is not None: a.bias.copy_(b.bias)
    if qnorm:
        dst.q_norm.weight.copy_(src.q_layernorm.weight)
        dst.k_norm.weight.copy_(src.k_layernorm.weight)


def ffn_copy(dst, gate, up, down):
    dst.gate.weight.copy_(gate.weight);dst.up.weight.copy_(up.weight);dst.down.weight.copy_(down.weight)


def test_gpt2_full_logits_match_transformers(monkeypatch):
    from transformers import GPT2Config,GPT2LMHeadModel
    cfg=GPT2Config(vocab_size=32,n_positions=16,n_embd=16,n_layer=2,n_head=4,n_inner=32)
    source=GPT2LMHeadModel(cfg).eval()
    p=reduced(next(p for p in templates() if p['id']=='gpt2'))
    for n in p['graph']['nodes']:
        if n['id']=='pos': n['data']['params']['max_length']=16
        if n['id']=='stack':
            n['data']['params']['iterations']=2
            for child in n['data']['body']['nodes']:
                if child['data']['kind']=='attention': child['data']['params'].update(heads=4,kv_heads=4)
    model,_=instantiate(p,monkeypatch);model.eval()
    with torch.no_grad():
        model.token_embeddings.weight.copy_(source.transformer.wte.weight);model.learned_positions_1024.table.weight.copy_(source.transformer.wpe.weight)
        model.final_layer_norm.load_state_dict(source.transformer.ln_f.state_dict())
        for a,b in zip(model.layer_12_decoder_layers,source.transformer.h):
            a.attention_layer_norm.load_state_dict(b.ln_1.state_dict());a.feed_forward_layer_norm.load_state_dict(b.ln_2.state_dict())
            qkv=b.attn.c_attn.weight.T.chunk(3);bias=b.attn.c_attn.bias.chunk(3)
            for x,w,z in zip((a.causal_multi_head_attention.q,a.causal_multi_head_attention.k,a.causal_multi_head_attention.v),qkv,bias):x.weight.copy_(w);x.bias.copy_(z)
            a.causal_multi_head_attention.o.weight.copy_(b.attn.c_proj.weight.T);a.causal_multi_head_attention.o.bias.copy_(b.attn.c_proj.bias)
            for x,y in [(a.expand_features,b.mlp.c_fc),(a.project_to_model_width,b.mlp.c_proj)]:x.weight.copy_(y.weight.T);x.bias.copy_(y.bias)
        ids=torch.tensor([[1,2,3,4,5]])
        torch.testing.assert_close(model(ids),source(ids).logits,atol=2e-6,rtol=2e-5)


def test_hammer_full_logits_match_transformers(monkeypatch):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=24,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32768,rope_theta=1000000.,rope_scaling={'rope_type':'yarn','factor':4.,'original_max_position_embeddings':32768},tie_word_embeddings=True)
    source=Qwen2ForCausalLM(cfg).eval()
    p=reduced(next(p for p in templates() if p['id']=='hammer21-05b'))
    model,_=instantiate(p,monkeypatch);model.eval()
    with torch.no_grad():
        model.token_embedding.weight.copy_(source.model.embed_tokens.weight);model.final_rmsnorm.weight.copy_(source.model.norm.weight)
        for i,b in enumerate(source.model.layers,1):
            a=getattr(model,f'layer_{i}_gqa_decoder');a.operator_rmsnorm.weight.copy_(b.input_layernorm.weight);a.feed_forward_rmsnorm.weight.copy_(b.post_attention_layernorm.weight)
            attention_copy(a.causal_grouped_query_attention,b.self_attn);ffn_copy(a.swi_glu_feed_forward,b.mlp.gate_proj,b.mlp.up_proj,b.mlp.down_proj)
        ids=torch.tensor([[1,2,3,4,5]])
        torch.testing.assert_close(model(ids),source(ids).logits,atol=2e-6,rtol=2e-5)


def test_shellper_full_logits_match_transformers(monkeypatch):
    transformers=pytest.importorskip("transformers")
    if not hasattr(transformers,"Lfm2ForCausalLM"): pytest.skip("LFM2 requires a newer Transformers release")
    Lfm2Config,Lfm2ForCausalLM=transformers.Lfm2Config,transformers.Lfm2ForCausalLM
    p=reduced(next(p for p in templates() if p['id']=='lfm-shellper-350m'))
    schedule=['full_attention' if 'GQA attention' in n['data']['label'] else 'conv' for n in p['graph']['nodes'] if n['data']['kind']=='block']
    cfg=Lfm2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=16,num_attention_heads=4,num_key_value_heads=2,layer_types=schedule,block_auto_adjust_ff_dim=False,conv_L_cache=3,rope_parameters={'rope_theta':1000000.,'rope_type':'default'},tie_word_embeddings=True,norm_eps=1e-5)
    source=Lfm2ForCausalLM(cfg).eval()
    model,_=instantiate(p,monkeypatch);model.eval()
    with torch.no_grad():
        model.token_embedding.weight.copy_(source.model.embed_tokens.weight);model.final_rmsnorm.weight.copy_(source.model.embedding_norm.weight)
        for i,b in enumerate(source.model.layers,1):
            suffix='gqa_attention' if schedule[i-1]=='full_attention' else 'convolution'
            a=getattr(model,f'layer_{i}_{suffix}');a.operator_rmsnorm.weight.copy_(b.operator_norm.weight);a.feed_forward_rmsnorm.weight.copy_(b.ffn_norm.weight)
            ffn_copy(a.swi_glu_feed_forward,b.feed_forward.w1,b.feed_forward.w3,b.feed_forward.w2)
            if suffix=='gqa_attention':attention_copy(a.causal_grouped_query_attention,b.self_attn,True)
            else:
                for dst,weight in zip((a.input_gate_b,a.output_gate_c,a.input_projection),b.conv.in_proj.weight.chunk(3)):dst.weight.copy_(weight)
                a.output_projection.weight.copy_(b.conv.out_proj.weight);a.depthwise_causal_convolution.taps.copy_(b.conv.conv.weight.squeeze(1).T.flip(0))
        ids=torch.tensor([[1,2,3,4,5]])
        torch.testing.assert_close(model(ids),source(ids).logits,atol=2e-6,rtol=2e-5)


def fixture(version):
    import numpy as np
    path=Path(__file__).parent/'fixtures'/'reference'/f'needle{version}_small.npz'
    return {key:torch.from_numpy(value.copy()) for key,value in np.load(path).items()}


def copy_dense(module, values, base, index=None):
    def tensor(name):
        v=values[base+'/'+name]
        return v if index is None else v[index]
    for field,up in [('q','q_proj'),('k','k_proj'),('v','v_proj'),('o','out_proj')]:getattr(module,field).weight.copy_(tensor(up+'/kernel').T)
    module.q_norm.weight.copy_(1+tensor('q_norm/scale'));module.k_norm.weight.copy_(1+tensor('k_norm/scale'))
    if module.gate is not None:module.gate.weight.copy_(tensor('gate_proj/kernel').T)
    if module.taps:
        for j,key in enumerate(('q_taps','k_taps','v_taps')):module.taps[j].copy_(tensor(key))


def test_needle1_full_logits_match_original_jax(monkeypatch):
    vals=fixture(1);p=reduced(next(p for p in templates() if p['id']=='needle1'))
    for n in p['graph']['nodes']:
        if n['id'] in ('encoder','decoder'):n['data']['params']['iterations']=2
    model,_=instantiate(p,monkeypatch);model.eval()
    with torch.no_grad():
        model.shared_embedding_encoder.weight.copy_(vals['params/embedding/embedding'])
        model.encoder_final_rmsnorm.weight.copy_(1+vals['params/encoder/final_norm/scale'])
        model.final_rmsnorm.weight.copy_(1+vals['params/decoder/ZCRMSNorm_0/scale'])
        for i,a in enumerate(model.encoder_12_attention_layers):
            base='params/encoder/layers/EncoderBlock_0';a.pre_attention_norm.weight.copy_(1+vals[base+'/ZCRMSNorm_0/scale'][i]);a.residual_connection.logit.copy_(vals[base+'/attn_gate'][i]);copy_dense(a.grouped_query_attention,vals,base+'/self_attn',i)
        for i,a in enumerate(model.decoder_8_attention_layers):
            base='params/decoder/layers/DecoderBlock_0';a.pre_attention_norm.weight.copy_(1+vals[base+'/ZCRMSNorm_0/scale'][i]);a.cross_attention_norm.weight.copy_(1+vals[base+'/ZCRMSNorm_1/scale'][i]);a.residual_connection.logit.copy_(vals[base+'/self_attn_gate'][i]);a.cross_residual.logit.copy_(vals[base+'/cross_attn_gate'][i]);copy_dense(a.grouped_query_attention,vals,base+'/self_attn',i);copy_dense(a.cross_attention,vals,base+'/cross_attn',i)
        ids=torch.tensor([[1,2,3,4,5]])
        torch.testing.assert_close(model(ids,ids),vals['logits'],atol=2e-5,rtol=2e-4)


def small_needle(version):
    from backend.templates import needle_network
    conf={'hidden_size':16,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2,'qk_head_dim':4,'value_head_dim':6 if version==3 else 4,'rope_theta':100000.,'sliding_window':2 if version==3 else 0,'extras':{'mhc_lanes':4,'engram':{'sites':[0],'orders':[2,3],'slots':32,'sub_dim':8 if version==2 else 4},'global_attention_layers':[1] if version==3 else [],'qkv_conv_taps':3 if version==3 else 0}}
    p=copy.deepcopy(next(p for p in templates() if p['id']==f'needle{version}'));p['graph']=needle_network(conf);p['config'].update(width=16,heads=4,ffn_width=32,vocab=32)
    if version==2:
        stack=next(n for n in p['graph']['nodes'] if n['id']=='stack')['data']['body']
        for layer in stack['nodes']:
            for child in layer['data'].get('body',{}).get('nodes',[]):
                if child['data']['kind']=='monarch':child['data'].update(kind='hadamard',label='Two-stage Hadamard MLP',params={'variant':'two_stage'})
    return p


def test_needle2_and_3_full_logits_match_original_jax(monkeypatch):
    for version in (2,3):
        vals=fixture(version);model,_=instantiate(small_needle(version),monkeypatch);model.eval()
        with torch.no_grad():
            model.token_embedding.weight.copy_(vals['params/embedding/embedding']);model.final_rmsnorm.weight.copy_(1+vals['params/stack/final_norm/scale'])
            stack=model.layer_2_layer_simple_attention_network
            for i,a in enumerate((stack.layer_1_engram,stack.layer_2)):
                route=a.input_dependent_lane_routing
                for dst,src in [('pre','phi_pre'),('post','phi_post'),('res','phi_res'),('bp','b_pre'),('bo','b_post'),('br','b_res')]:getattr(route,dst).copy_(vals[f'params/stack/mhc_{src}'][i])
                for j,name in enumerate(('pre','post','res')):route.a[j].copy_(vals[f'params/stack/mhc_a_{name}'][i])
                base='params/stack/layers/block'
                for dst,src in [(a.pre_attention_rmsnorm,'ZCRMSNorm_0'),(a.post_attention_rmsnorm,'post_attn_norm'),(a.pre_mixer_rmsnorm,'pre_hada_norm')]:dst.weight.copy_(1+vals[f'{base}/{src}/scale'][i])
                a.gated_attention_residual.logit.copy_(vals[base+'/attn_gate'][i])
                attn=getattr(a,'global_gqa_attention',getattr(a,'local_gqa_attention',None));copy_dense(attn,vals,base+'/self_attn',i)
                if version==2:
                    mix=a.two_stage_hadamard_mlp
                    for name in ('d1','d2','d3'):getattr(mix,name).copy_(vals[f'{base}/hadamard_mlp/{name}'][i])
                else:
                    mix=a.monarch_hadamard_mlp
                    for j,name in enumerate(('w1a','w1b','w2a','w2b','w3a','w3b')):mix.factors[j].copy_(vals[f'{base}/hadamard_mlp/{name}'][i])
                    for j in range(4):mix.d[j].copy_(vals[f'{base}/hadamard_mlp/d{j+1}'][i])
                    mix.bias.copy_(vals[base+'/hadamard_mlp/b2'][i]);mix.cv.copy_(vals[base+'/hadamard_mlp/cond_v'][i]);mix.cu.copy_(vals[base+'/hadamard_mlp/cond_u'][i])
                if i==0:
                    eng=a.engram_key_value_tables
                    for j,t in enumerate(eng.tables):t.weight.copy_(vals['params/engrams_0/embedding'][j])
                    eng.key.weight.copy_(vals['params/engrams_0/key_proj/kernel'].T);eng.value.weight.copy_(vals['params/engrams_0/value_proj/kernel'].T);eng.taps.copy_(vals['params/engrams_0/taps'])
            ids=torch.tensor([[1,2,3,4,5]])
            torch.testing.assert_close(model(ids),vals['logits'],atol=2e-5,rtol=2e-4,msg=f'Needle {version} original JAX parity')


def test_references_use_corrected_original_needle_components():
    examples={p['id']:p for p in templates()}
    one=examples['needle1']['graph']
    assert {'se_scale','te_scale','encoder_norm'} <= {n['id'] for n in one['nodes']}
    decoder=next(n for n in one['nodes'] if n['id']=='decoder')['data']['body']
    assert next(n for n in decoder['nodes'] if n['id']=='cross')['data']['params']['qk_norm'] is True
    two=examples['needle2']['graph'];stack=next(n for n in two['nodes'] if n['id']=='stack')['data']['body']
    mixers=[child['data'] for layer in stack['nodes'] for child in layer['data'].get('body',{}).get('nodes',[]) if child['id']=='monarch']
    assert len(mixers)==27 and all(m['kind']=='hadamard' and m['params']['variant']=='two_stage' for m in mixers)
