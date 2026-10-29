"""Editable compositions; all templates use the same graph operations."""
from copy import deepcopy
import json
from pathlib import Path


def node(id, kind, label, x, y, p=None, body=None, **extra):
    return {'id':id,'type':'architecture','position':{'x':x,'y':y},'data':{'kind':kind,'label':label,'params':p or {},**({'body':body} if body else {}),**extra}}


def edge(a,b,port='a'):
    return {'id':f'{a}-{b}-{port}','source':a,'target':b,'sourceHandle':'out','targetHandle':port,'type':'smoothstep'}


def transformer(causal=True, sandwich=False, ffn=True, cross=False, heads='$heads', kv=4):
    ns=[node('x','input','Hidden state',0,120,{'index':0,'shape':['B','T','$width'],'dtype':'float32'}),
        node('n1','norm','Pre-attention norm',260,120),
        node('attn','attention','Grouped-query attention',520,120,{'heads':heads,'kv_heads':kv,'causal':causal,'qk_norm':True,'gated':sandwich,'rope':True,'window':0,'backend':'dense'}),
        node('r1','gate' if not sandwich else 'add','Residual connection',1040,120)]
    es=[edge('x','n1'),edge('n1','attn'),edge('x','r1'),edge('attn','r1','b')]
    last='r1'
    if sandwich:
        ns.append(node('n2','norm','Post-attention norm',780,120)); es[-1]=edge('n2','r1','b');es.append(edge('attn','n2'))
    if cross:
        ns.extend([node('memory','input','Encoder memory',260,460,{'index':1,'shape':['B','S','$width'],'dtype':'float32'}),node('cn','norm','Cross-attention norm',1300,120),node('cross','cross_attention','Cross attention',1560,120,{'heads':heads,'kv_heads':kv,'causal':False,'rope':False,'qk_norm':True}),node('cr','gate','Cross residual',1820,120)])
        es.extend([edge(last,'cn'),edge('cn','cross'),edge('memory','cross','b'),edge(last,'cr'),edge('cross','cr','b')]);last='cr'
    if ffn:
        start=1300 if not cross else 2080
        ns.extend([node('fn','norm','Pre-FFN norm',start,120),node('ffn','ffn','SwiGLU feed-forward',start+260,120,{'hidden':'$ffn_width'}),node('r2','add','Residual connection',start+780,120)])
        es.extend([edge(last,'fn'),edge('fn','ffn'),edge(last,'r2')])
        if sandwich:
            ns.append(node('fn2','norm','Post-FFN norm',start+520,120));es.extend([edge('ffn','fn2'),edge('fn2','r2','b')])
        else: es.append(edge('ffn','r2','b'))
        last='r2'
    ns.append(node('y','output','Hidden state out',max(n['position']['x'] for n in ns)+280,120));es.append(edge(last,'y'))
    return {'nodes':ns,'edges':es}


def predictor(causal=False):
    return {'nodes':[
        node('ids','input','Token IDs',0,200,{'index':0,'shape':['B','T'],'dtype':'int64'}),
        node('emb','embedding','Token embedding',280,60,{'width':'$width','vocab':'$vocab'}),
        node('grams','ngram','N-gram windows',280,360,{'orders':[2,3],'buckets':2048}),
        node('memory','lookup','Hashed n-gram memory',560,360,{'buckets':2048,'width':'$width'}),
        node('fusion','add','Feature fusion',840,160),
        node('refine','loop','Recurrent transformer',1120,160,{'iterations':3,'shared':True},transformer(causal=causal,kv=2)),
        node('norm','norm','Final RMSNorm',1460,160),
        node('head','head','Vocabulary projection',1740,160,{'out':'$vocab','tie':'emb'}),
        node('out','output','Token logits',2020,160)],
        'edges':[edge('ids','emb'),edge('ids','grams'),edge('grams','memory'),edge('emb','fusion'),edge('memory','fusion','b'),edge('fusion','refine'),edge('refine','norm'),edge('norm','head'),edge('head','out')]}


def denoise_body():
    return {'nodes':[
        node('state','input','Current tokens',0,150,{'index':0,'shape':['B','T'],'dtype':'int64'}),
        node('predict','block','Token predictor',300,50,{},predictor()),
        node('select','select','Confidence token selection',660,150,{'strategy':'confidence','temperature':0}),
        node('out','output','Next token state',960,150)],
        'edges':[edge('state','predict'),edge('predict','select'),edge('state','select','b'),edge('select','out')]}


def combined():
    return {'nodes':[
        node('ids','input','Prompt tokens',0,100,{'index':0,'shape':['B','T'],'dtype':'int64'}),
        node('mask','mask','Append masked tokens',320,100,{'length':16,'mask_id':'$mask_id'}),
        node('denoise','loop','Iterative denoising',680,100,{'iterations':8,'shared':True},denoise_body()),
        node('out','output','Resolved token IDs',1060,100)],
        'edges':[edge('ids','mask'),edge('mask','denoise'),edge('denoise','out')]}


def language_model(body=None, layers=6, width=128):
    return {'nodes':[node('ids','input','Token IDs',0,150,{'index':0,'shape':['B','T'],'dtype':'int64'}),node('emb','embedding','Token embedding',280,150,{'width':'$width','vocab':'$vocab'}),node('stack','loop','Transformer layers',600,150,{'iterations':layers,'shared':False},body or transformer(kv=2)),node('norm','norm','Final RMSNorm',940,150),node('head','head','Tied language head',1220,150,{'out':'$vocab','tie':'emb'}),node('out','output','Token logits',1500,150)],'edges':[edge('ids','emb'),edge('emb','stack'),edge('stack','norm'),edge('norm','head'),edge('head','out')]}


def explicit_looped():
    g=language_model();g['nodes']=[n for n in g['nodes'] if n['id']!='stack'];g['edges']=[e for e in g['edges'] if e['source']!='stack' and e['target']!='stack']
    body=transformer(kv=2);previous='emb'
    for i in range(3):
        id=f'transformer_{i+1}'
        g['nodes'].append(node(id,'block',f'Transformer · application {i+1}',640+i*760,150,{},deepcopy(body),share='transformer_core',role='transformer'))
        g['edges'].append(edge(previous,id));previous=id
        if i<2:
            adapter=f'adapter_{i+1}'
            g['nodes'].append(node(adapter,'linear',f'Independent learned layer {i+1}',1020+i*760,150,{'out':'$width'},role='adapter'))
            g['edges'].append(edge(previous,adapter));previous=adapter
    g['edges'].append(edge(previous,'norm'))
    for n in g['nodes']:
        if n['id'] in ('norm','head','out'):n['position']['x']={'norm':2920,'head':3240,'out':3560}[n['id']]
    return g



def expert_graph():
    return {'nodes':[
        node('input','input','Expert input',0,120,{'index':0,'shape':['B','T','$width'],'dtype':'float32'}),
        node('ffn','ffn','Expert feed-forward',300,120,{'hidden':'$ffn_width'}),
        node('output','output','Expert output',600,120),
    ], 'edges':[edge('input','ffn'),edge('ffn','output')]}


def moe_body(experts=4):
    nodes=[node('input','input','Token features',0,180,{'index':0,'shape':['B','T','$width'],'dtype':'float32'}),
           node('router','moe_router','Token router',320,0,{'experts':experts}),
           node('combine','moe_combine','Weighted combine',1100,180,{'experts':experts}),
           node('output','output','Mixed features',1400,180)]
    edges=[edge('input','router'),edge('router','combine'),edge('combine','output')]
    for index in range(experts):
        expert_id=f'expert_{index+1}'
        nodes.append(node(expert_id,'block',f'Expert {index+1}',700,index*230,{},expert_graph(),role='moe-expert'))
        edges.extend([edge('input',expert_id),edge(expert_id,'combine',f'e{index}')])
    return {'nodes':nodes,'edges':edges}


def moe_transformer():
    """Expose the attention and MoE residual branch on the main canvas."""
    graph=language_model(transformer(kv=2, ffn=False), layers=1)
    attention=next(n for n in graph['nodes'] if n['id']=='stack')
    attention['data'].update(kind='block', label='Causal attention block', params={}, role='transformer')
    graph['nodes'].extend([
        node('moe_norm','norm','Pre-expert RMSNorm',950,120),
        node('moe','moe','Mixture of experts',1250,120,{'experts':4,'top_k':2,'balance_weight':0.01,'router':'softmax'},moe_body(),role='moe'),
        node('moe_residual','add','Expert residual',1600,250),
    ])
    graph['edges']=[e for e in graph['edges'] if not (e['source']=='stack' and e['target']=='norm')]
    graph['edges'].extend([edge('stack','moe_norm'),edge('moe_norm','moe'),edge('stack','moe_residual'),edge('moe','moe_residual','b'),edge('moe_residual','norm')])
    for n in graph['nodes']:
        if n['id'] in ('norm','head','out'):
            n['position']['x']={'norm':1950,'head':2250,'out':2550}[n['id']]
    return graph


def embedding_example():
    return {'nodes':[node('ids','input','Token IDs',0,120,{'index':0,'shape':['B','T'],'dtype':'int64'}),
                     node('embedding','embedding','Embedding table',320,120,{'width':'$width','vocab':'$vocab'}),
                     node('output','output','Token vectors',640,120)],
            'edges':[edge('ids','embedding'),edge('embedding','output')]}

def sparse_transformer():
    g=language_model();body=g['nodes'][2]['data']['body']
    body['nodes'].extend([node('local','attention_pattern','Local attention window',0,500,{'pattern':'local','window':32,'causal':False}),node('global','attention_pattern','Global anchor tokens',0,800,{'pattern':'global','global_tokens':[0],'global_stride':0,'causal':False}),node('combine','pattern_union','Local ∪ global attention',320,600)])
    body['edges'].extend([edge('x','local'),edge('x','global'),edge('local','combine'),edge('global','combine','b'),edge('combine','attn','b')])
    return g


def needle_layer(index, conf):
    """An mHC layer is an ordinary editable subgraph, not a separate canvas mode."""
    width=conf['hidden_size'];heads=conf['num_attention_heads'];kv=conf['num_key_value_heads']
    lanes=conf['extras']['mhc_lanes'];engram=conf['extras']['engram']
    global_layer=index in conf['extras'].get('global_attention_layers',[])
    nodes=[node('x','input','Residual lanes',0,160,{'index':0,'shape':['B','T',lanes,'$width'],'dtype':'float32'}),
        node('ids','input','Token IDs',0,600,{'index':1,'shape':['B','T'],'dtype':'int64'}),
        node('routing','routing','Input-dependent lane routing',310,0,{'lanes':lanes,'layer':index}),
        node('mix','lane_mix','Gather residual lanes',640,160,{'lanes':lanes}),
        node('n1','norm','Pre-attention RMSNorm',1260,160),
        node('attn','attention',('Global' if global_layer else 'Local')+' GQA attention',1550,160,{'heads':heads,'kv_heads':kv,'head_dim':conf.get('qk_head_dim',width//heads),'value_dim':conf.get('value_head_dim',width//heads),'causal':True,'qk_norm':True,'gated':True,'theta':conf['rope_theta'],'window':0 if global_layer else conf.get('sliding_window',0),'conv_taps':conf['extras'].get('qkv_conv_taps',0),'conv_respects_window':True}),
        node('n2','norm','Post-attention RMSNorm',1840,160),node('attn_res','gate','Gated attention residual',2130,160),
        node('n3','norm','Pre-mixer RMSNorm',2420,160),node('monarch','monarch','Monarch Hadamard MLP',2710,160,{'split':False}),node('mlp_res','add','Mixer residual',3000,160),
        node('delta','subtract','Block update Δ',3300,160),node('merge','lane_merge','Sinkhorn residual merge',3620,160,{'lanes':lanes}),node('out','output','Updated residual lanes',3930,160)]
    edges=[edge('x','routing'),edge('x','mix'),edge('routing','mix','b'),edge('n1','attn'),edge('attn','n2'),edge('n2','attn_res','b'),edge('attn_res','n3'),edge('n3','monarch'),edge('attn_res','mlp_res'),edge('monarch','mlp_res','b'),edge('mlp_res','delta'),edge('mix','delta','b'),edge('x','merge'),edge('delta','merge','b'),edge('routing','merge','c'),edge('merge','out')]
    upstream='mix'
    if index in engram['sites']:
        sub_dim=engram['sub_dim'];hash_heads=max(1,width//(len(engram['orders'])*sub_dim))
        common={'orders':engram['orders'],'buckets':engram['slots'],'hash_heads':hash_heads}
        nodes.extend([node('hash','engram_hash','Engram n-gram hashing',310,600,common),node('lookup','engram_lookup','Engram key/value tables',640,600,{**common,'sub_dim':sub_dim,'width':'$width','dilation':engram.get('conv_dilation',3)}),node('fuse','memory_fuse','Gated memory retrieval',960,160)])
        edges.extend([edge('ids','hash'),edge('hash','lookup'),edge('mix','fuse'),edge('lookup','fuse','b')]);upstream='fuse'
    edges.extend([edge(upstream,'n1'),edge(upstream,'attn_res')])
    return {'nodes':nodes,'edges':edges}


def needle_network(conf):
    lanes=conf['extras']['mhc_lanes'];layers=conf['num_hidden_layers'];width=conf['hidden_size']
    stack={'nodes':[node('x','input','Residual lanes',0,150,{'index':0,'shape':['B','T',lanes,'$width'],'dtype':'float32'}),node('ids','input','Token IDs',0,600,{'index':1,'shape':['B','T'],'dtype':'int64'})],'edges':[]}
    last='x'
    for i in range(layers):
        id=f'layer_{i}';stack['nodes'].append(node(id,'block',f'Layer {i+1}'+(' · Engram' if i in conf['extras']['engram']['sites'] else ''),340*(i+1),150,{},needle_layer(i,conf)))
        stack['edges'].extend([edge(last,id),edge('ids',id,'b')]);last=id
    stack['nodes'].append(node('out','output','Residual lanes out',340*(layers+1),150));stack['edges'].append(edge(last,'out'))
    return {'nodes':[node('ids','input','Token IDs',0,150,{'index':0,'shape':['B','T'],'dtype':'int64'}),node('emb','embedding','Token embedding',280,150,{'width':'$width','vocab':'$vocab'}),node('scale','scale','Scale embeddings · √D',570,150,{'factor':'$sqrt_width'}),node('lanes','lane_expand','Expand residual lanes',850,150,{'lanes':lanes}),node('stack','block',f'{layers}-layer Simple Attention Network',1180,150,{},stack),node('reduce','lane_reduce','Mean residual lanes',1520,150,{'lanes':lanes}),node('norm','norm','Final RMSNorm',1800,150),node('head','head','Tied vocabulary head',2080,150,{'out':'$vocab','tie':'emb'}),node('out','output','Token logits',2360,150)],'edges':[edge('ids','emb'),edge('emb','scale'),edge('scale','lanes'),edge('lanes','stack'),edge('ids','stack','b'),edge('stack','reduce'),edge('reduce','norm'),edge('norm','head'),edge('head','out')]}



def model_overview(layers, layer_description, layer_notes):
    """Source-based diagrams with explicit placeholders for unsupported operators."""
    g = {'nodes': [node('ids','input','Token IDs',0,150,{'index':0,'shape':['B','T'],'dtype':'int64'}),
                   node('emb','embedding','Token embedding',300,150,{'width':'$width','vocab':'$vocab'})],
         'edges': [edge('ids','emb')]}
    previous = 'emb'
    annotated_kinds = set()
    for index, kind in enumerate(layers):
        key = f'layer_{index + 1}'
        g['nodes'].append(node(key,'reference',f'Layer {index + 1} · {kind}',600 + index*290,150,{},
            description=layer_description, **({'annotation':layer_notes[kind]} if kind not in annotated_kinds else {})))
        annotated_kinds.add(kind)
        g['edges'].append(edge(previous,key)); previous = key
    x = 600 + len(layers)*290
    g['nodes'].extend([node('norm','norm','Final RMSNorm',x,150),node('head','head','Tied vocabulary head',x+290,150,{'out':'$vocab','tie':'emb'}),node('out','output','Token logits',x+580,150)])
    g['edges'].extend([edge(previous,'norm'),edge('norm','head'),edge('head','out')])
    return g



def lfm_layer(attention):
    """Editable LFM2 layer; separate projections are equivalent to chunks of in_proj."""
    ns=[node('x','input','Hidden state',0,160,{'index':0,'shape':['B','T','$width'],'dtype':'float32'}),
        node('norm','norm','Operator RMSNorm',300,160,{'eps':1e-5})]
    es=[edge('x','norm')]
    if attention:
        ns.append(node('operator','attention','Causal grouped-query attention',620,160,
            {'heads':16,'kv_heads':8,'causal':True,'rope':True,'theta':1000000.0,'qk_norm':True,'qk_norm_eps':1e-5,'gated':False,'window':0},
            annotation='Six of the 16 layers use attention. Multiple query heads share each key/value head.'))
        es.append(edge('norm','operator')); last='operator'
    else:
        ns.extend([node('b','linear','Input gate B',620,0,{'out':'$width','bias':False}),
                   node('c','linear','Output gate C',620,320,{'out':'$width','bias':False}),
                   node('v','linear','Input projection',620,160,{'out':'$width','bias':False}),
                   node('gate_in','multiply','Gate input · B × x',920,100),
                   node('conv','causal_conv','Depthwise causal convolution',1220,100,{'taps':3},annotation='A short learned filter mixes each feature with its previous two positions; future tokens are never used.'),
                   node('gate_out','multiply','Gate output · C × conv',1520,150),
                   node('operator','linear','Output projection',1820,150,{'out':'$width','bias':False})])
        es.extend([edge('norm','b'),edge('norm','c'),edge('norm','v'),edge('b','gate_in'),edge('v','gate_in','b'),edge('gate_in','conv'),edge('conv','gate_out'),edge('c','gate_out','b'),edge('gate_out','operator')]); last='operator'
    x=2120 if not attention else 940
    ns.extend([node('res1','add','Operator residual',x,160),node('fn','norm','Feed-forward RMSNorm',x+300,160,{'eps':1e-5}),
               node('ffn','ffn','SwiGLU feed-forward',x+600,160,{'hidden':'$ffn_width'}),node('res2','add','Feed-forward residual',x+900,160),node('y','output','Hidden state out',x+1200,160)])
    es.extend([edge('x','res1'),edge(last,'res1','b'),edge('res1','fn'),edge('fn','ffn'),edge('res1','res2'),edge('ffn','res2','b'),edge('res2','y')])
    return {'nodes':ns,'edges':es}



def gpt2_model():
    body={'nodes':[node('x','input','Hidden state',0,120,{'index':0,'shape':['B','T','$width'],'dtype':'float32'}),
        node('n1','layer_norm','Attention LayerNorm',300,120,{'eps':1e-5,'affine':True,'bias':True}),
        node('attn','attention','Causal multi-head attention',600,120,{'heads':12,'kv_heads':12,'causal':True,'rope':False,'qkv_bias':True,'output_bias':True,'attention_dropout':0.1}),
        node('drop1','dropout','Attention residual dropout',900,120,{'probability':0.1}),node('r1','add','Attention residual',1200,120),
        node('n2','layer_norm','Feed-forward LayerNorm',1500,120,{'eps':1e-5,'affine':True,'bias':True}),
        node('up','linear','Expand features',1800,120,{'out':'$ffn_width','bias':True}),
        node('gelu','gelu','GELU (tanh approximation)',2100,120,{'approximate':'tanh'},annotation='GPT-2 uses the tanh approximation to GELU in its feed-forward network.'),
        node('down','linear','Project to model width',2400,120,{'out':'$width','bias':True}),
        node('drop2','dropout','Feed-forward residual dropout',2700,120,{'probability':0.1}),node('r2','add','Feed-forward residual',3000,120),node('y','output','Hidden state out',3300,120)],
        'edges':[edge('x','n1'),edge('n1','attn'),edge('attn','drop1'),edge('x','r1'),edge('drop1','r1','b'),edge('r1','n2'),edge('n2','up'),edge('up','gelu'),edge('gelu','down'),edge('down','drop2'),edge('r1','r2'),edge('drop2','r2','b'),edge('r2','y')]}
    nodes=[node('ids','input','Token IDs',0,120,{'index':0,'shape':['B','T'],'dtype':'int64'}),node('emb','embedding','Token embeddings',300,120,{'vocab':'$vocab','width':'$width'}),
        node('pos','position_embedding','Learned positions · 1024',600,120,{'max_length':1024}),node('drop','dropout','Embedding dropout',900,120,{'probability':0.1}),
        node('stack','loop','12 decoder layers',1200,120,{'iterations':12,'shared':False},body),node('norm','layer_norm','Final LayerNorm',1500,120,{'eps':1e-5,'affine':True,'bias':True}),
        node('head','head','Tied vocabulary head',1800,120,{'out':'$vocab','tie':'emb'}),node('out','output','Token logits',2100,120)]
    return {'nodes':nodes,'edges':[edge('ids','emb'),edge('emb','pos'),edge('pos','drop'),edge('drop','stack'),edge('stack','norm'),edge('norm','head'),edge('head','out')]}


def templates():
    base={'version':1,'config':{'width':128,'heads':8,'ffn_width':256,'vocab':1024,'mask_id':3},'tokenizer':None,'references':[]}
    def project(id,name,description,graph,**extra):
        return {**deepcopy(base),'id':id,'name':name,'description':description,'graph':graph,**extra}
    items=[project('combined','Composed language model','N-gram memory + recurrent attention + masked denoising. One graph, built from reusable parts.',combined()),
           project('transformer','Decoder transformer','A causal decoder with grouped-query attention, rotary positions and SwiGLU.',language_model()),
           project('looped','Looped transformer','Three applications share one transformer’s weights, with independently learned layers between them. Click any connection to insert more layers.',explicit_looped()),
           project('diffusion','Masked text diffusion','LLaDA-style bidirectional prediction with iterative confidence-based unmasking.',combined(),references=[{'title':'LLaDA sampling reference','url':'https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py','revision':'9182493720ed723ef8031210d85959364e51cbe0','notes':'Composable LLaDA-style example, not an 8B checkpoint reproduction. Linear remaining-token schedule; no classifier-free guidance.'}])]
    # Needle 1 has a real encoder-decoder topology and no feed-forward network.
    n1={'nodes':[node('source','input','Source token IDs',0,80,{'index':0,'shape':['B','S'],'dtype':'int64'}),node('target','input','Decoder token IDs',0,440,{'index':1,'shape':['B','T'],'dtype':'int64'}),node('se','embedding','Shared embedding · encoder',290,80,{'width':'$width','vocab':'$vocab'},share='token_embedding'),node('te','embedding','Shared embedding · decoder',290,440,{'width':'$width','vocab':'$vocab'},share='token_embedding'),node('encoder','loop','Encoder · 12 attention layers',620,80,{'iterations':12,'shared':False},transformer(causal=False,ffn=False)),node('decoder','loop','Decoder · 8 attention layers',1000,440,{'iterations':8,'shared':False},transformer(causal=True,ffn=False,cross=True)),node('norm','norm','Final RMSNorm',1360,440),node('head','head','Tied language head',1640,440,{'out':'$vocab','tie':'te'}),node('out','output','Decoder logits',1920,440)],'edges':[edge('source','se'),edge('target','te'),edge('se','encoder'),edge('te','decoder'),edge('encoder','decoder','b'),edge('decoder','norm'),edge('norm','head'),edge('head','out')]}
    # Original Needle scales both token embeddings and normalizes the encoder output.
    n1['nodes'].extend([node('se_scale','scale','Scale source embeddings · √D',460,80,{'factor':'$sqrt_width'}),
                        node('te_scale','scale','Scale decoder embeddings · √D',460,440,{'factor':'$sqrt_width'}),
                        node('encoder_norm','norm','Encoder final RMSNorm',810,80)])
    n1['edges']=[e for e in n1['edges'] if (e['source'],e['target']) not in {('se','encoder'),('te','decoder'),('encoder','decoder')}]
    n1['edges'].extend([edge('se','se_scale'),edge('se_scale','encoder'),edge('te','te_scale'),edge('te_scale','decoder'),edge('encoder','encoder_norm'),edge('encoder_norm','decoder','b')])
    items.append(project('needle1','Needle 1','Encoder–decoder, pure attention, 12 encoder + 8 decoder layers. No FFN.',n1,config={'width':512,'heads':8,'ffn_width':1024,'vocab':8192,'mask_id':3},references=[{'title':'Needle 1 original implementation','url':'https://github.com/cactus-compute/needle/blob/1807b1d230ba/needle/model/architecture.py'},{'title':'Needle 1 model card and config','url':'https://huggingface.co/Cactus-Compute/needle/tree/5f89b4307696d669c3df1d38ae057e6e1728b107','revision':'5f89b4307696d669c3df1d38ae057e6e1728b107','notes':'Attention-only trunk including scaled embeddings, encoder/decoder output normalization and query/key normalization. Float32 construction; padding masks, QAT, pretrained weights and KV caching are not reproduced.'}]))
    for version,layers,width,heads,kv,rev in [(2,27,512,8,4,'32e9e3a93b205f786929697446ae669cf0a84579'),(3,20,768,12,2,'d6c7d6a2e9655505e0f8e23339356dd1705900eb')]:
        conf=json.loads((Path(__file__).parent.parent/'references'/f'needle{version}_config.json').read_text())
        g=needle_network(conf)
        if version==2:
            for layer in next(n for n in g['nodes'] if n['id']=='stack')['data']['body']['nodes']:
                for child in layer['data'].get('body',{}).get('nodes',[]):
                    if child['data']['kind']=='monarch':child['data'].update(kind='hadamard',label='Two-stage Hadamard MLP',params={'variant':'two_stage'})
        items.append(project(f'needle{version}',f'Needle {version}',f'{layers} layers · {heads}/{kv} GQA · Engram memory · 4 residual lanes.',g,config={'width':width,'heads':heads,'ffn_width':width*2,'vocab':8192,'mask_id':3},references=[{'title':f'Needle {version} published configuration','url':f'https://huggingface.co/Cactus-Compute/needle{version}/tree/{rev}','revision':rev,'config':conf,'notes':'Published backbone settings with a float32 PyTorch port of Engram, learned Monarch mixing and mHC routing. Exports contain random weights, not checkpoint conversion. Quantization, padding-aware masks, auxiliary probe heads and trained ladder subnetworks are not reproduced.'},{'title':'Upstream architecture source','url':('https://github.com/cactus-compute/needle/blob/2e19bab90ca4/needle/model/architecture.py' if version==2 else 'https://github.com/cactus-compute/needle/blob/b4fc11b04f1852888d375c8c6dfbaca44572d014/needle/model/architecture.py'),'revision':('2e19bab90ca4' if version==2 else 'b4fc11b04f1852888d375c8c6dfbaca44572d014'),'notes':('Historical two-stage Hadamard mixer. This graph applies the published 256-token deployment window; the historical JAX forward defaults to full causal attention unless given a window mask. Checkpoint conversion is not implemented.' if version==2 else 'Needle 3 uses the learned Monarch mixer. Checkpoint conversion is not implemented.')}]))
    items.append(project('thimble','Thimble','Deep-thin gated transformer · sandwich RMSNorm · GQA 8/4 · tied embeddings.',language_model(transformer(sandwich=True,kv=4),20),config={'width':448,'heads':8,'ffn_width':896,'vocab':16384,'mask_id':3},references=[{'title':'Thimble PyTorch trunk','url':'https://github.com/nikshepsvn/thimble/blob/bb7d5768651d525516f3647d551005a4411845af/src/tiny_toolcall/model.py','revision':'bb7d5768651d525516f3647d551005a4411845af','notes':'Trunk computation reproduced with editable primitives. Grammar dispatcher, retrieval and auxiliary name/pointer heads are outside this trunk graph. Release vocabulary is 16K; source Config defaults to 8192.'}]))
    # These overviews expose the published layer schedule without pretending the
    # unsupported architecture-specific operators are executable implementations.
    hammer = model_overview(['GQA decoder']*24,
        'Reference-only Qwen2 decoder layer: Weave does not yet reproduce its QKV biases and YaRN rotary settings. Replace this placeholder to export.',
        {'GQA decoder':'Qwen2.5-Coder uses grouped-query causal attention and a SwiGLU feed-forward network. Hammer fine-tunes this backbone for function calling; this diagram does not include trained weights.'})
    for layer in hammer['nodes']:
        if layer['data']['kind'] == 'reference':
            body = lfm_layer(True)
            for inner in body['nodes']:
                if inner['data']['kind'] == 'norm': inner['data']['params']['eps']=1e-6
                if inner['data']['kind'] == 'attention':
                    inner['data']['params'].update(heads=14,kv_heads=2,qk_norm=False,qkv_bias=True,rope_scaling='yarn',rope_factor=4.0,rope_original_context=32768)
            layer['data'].update(kind='block',body=body,params={},role='transformer')
            layer['data'].pop('description',None)
    items.append(project('hammer21-05b','Hammer 2.1 · 0.5B','Qwen2.5-Coder-based function-calling model · 24 editable decoder layers; random-weight backbone.',hammer,
        config={'width':896,'heads':14,'ffn_width':4864,'vocab':151665,'mask_id':3},
        references=[{'title':'Hammer 2.1 0.5B model and configuration','url':'https://huggingface.co/MadeAgents/Hammer2.1-0.5b/blob/main/config.json','notes':'Random-weight Qwen2 backbone with QKV biases and YaRN, full-sequence inference. Tokenizer, fine-tuned weights, padding masks, KV cache and tool prompting are not implemented.'},{'title':'Hammer model card','url':'https://huggingface.co/MadeAgents/Hammer2.1-0.5b'}]))
    schedule=['Convolution','Convolution','GQA attention','Convolution','Convolution','GQA attention','Convolution','Convolution','GQA attention','Convolution','GQA attention','Convolution','GQA attention','Convolution','GQA attention','Convolution']
    shellper=model_overview(schedule,
        'Reference-only LFM2 layer: its architecture-specific convolution or attention and feed-forward computation are not implemented in Weave. Replace this placeholder to export.',
        {'Convolution':'LFM2 uses a short, causal, double-gated convolution for local sequence mixing. Each layer also has a feed-forward network and residual connections.',
         'GQA attention':'This layer uses causal grouped-query attention for sequence mixing, followed by a feed-forward network. The published schedule interleaves six attention and ten convolution layers.'})
    for layer in shellper['nodes']:
        if layer['data']['kind'] == 'reference':
            is_attention = 'GQA attention' in layer['data']['label']
            layer['data'].update(kind='block', body=lfm_layer(is_attention), params={}, role='transformer' if is_attention else 'composition')
            layer['data'].pop('description', None)
    next(n for n in shellper['nodes'] if n['id'] == 'norm')['data']['params']['eps'] = 1e-5
    items.append(project('lfm-shellper-350m','LFM2.5 SHELLper · 350M','Shell tool-calling fine-tune · 16 editable hybrid convolution and attention layers; random-weight backbone.',shellper,
        config={'width':1024,'heads':16,'ffn_width':4608,'vocab':65536,'mask_id':3},
        references=[{'title':'SHELLper model card','url':'https://huggingface.co/distil-labs/distil-lfm25-shellper','notes':'Random-weight backbone with full-sequence inference. Padding masks, KV/convolution caches, tokenizer, LoRA/fine-tuned weights and tool formatting are not implemented. The published 6656 FFN setting is adjusted to 4608 by LFM2 auto-sizing.'},{'title':'SHELLper configuration','url':'https://huggingface.co/distil-labs/distil-lfm25-shellper/blob/main/config.json'},{'title':'LFM2 implementation','url':'https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py'}]))
    # Source-linked design callouts describe what the authors say, or label
    # straightforward mechanics without attributing speculative motives.
    source_urls={
        'needle1':'https://github.com/cactus-compute/needle/blob/1807b1d230ba/needle/model/architecture.py',
        'needle2':'https://github.com/cactus-compute/needle/blob/2e19bab90ca4/needle/model/architecture.py',
        'needle3':'https://huggingface.co/Cactus-Compute/needle3',
        'thimble':'https://huggingface.co/flashvenom/thimble',
        'lfm-shellper-350m':'https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models',
        'hammer21-05b':'https://huggingface.co/MadeAgents/Hammer2.1-0.5b',
    }
    for item in items:
        if item['id'] in source_urls:
            for n in item['graph']['nodes']:
                if n['data'].get('annotation'):n['data']['annotationSource']=source_urls[item['id']]
        if item['id']=='needle1':
            decoder=next(n for n in item['graph']['nodes'] if n['id']=='decoder')
            decoder['data'].update(annotation='The decoder predicts target tokens while cross-attention reads the separately encoded source.', annotationSource=source_urls['needle1'])
        if item['id']=='needle2':
            stack=next(n for n in item['graph']['nodes'] if n['id']=='stack')
            stack['data'].update(annotation='The historical Needle 2 trunk uses two-stage Hadamard mixing in place of a conventional feed-forward network.', annotationSource=source_urls['needle2'])
        if item['id']=='needle3':
            stack=next(n for n in item['graph']['nodes'] if n['id']=='stack')
            stack['data'].update(annotation='The authors place much of Needle 3’s capacity in n-gram memory. Open this block, then a layer marked Engram, to see its key/value embedding tables. Ladder training and auxiliary heads are outside this graph.', annotationSource=source_urls['needle3'])
            site=next(n for n in stack['data']['body']['nodes'] if n['id']=='layer_3')
            site['data'].update(annotation='Engram memory is inserted at selected layers, rather than every layer. Open this layer to see the n-gram hashing and learned key/value tables.',annotationSource=source_urls['needle3'])
        if item['id']=='lfm-shellper-350m':
            conv=next(n for n in item['graph']['nodes'] if n['id']=='layer_1')
            conv['data']['annotation']='The LFM2 authors chose short gated convolutions with embedded CPUs and their optimized kernels in mind. SHELLper fine-tunes this hybrid backbone.'
            conv['data']['annotationSource']=source_urls[item['id']]
    # Remaining notes describe mechanics and are not inferred author claims.
    for item in items:
        if item['id'] in ('needle2','needle3'):
            next(n for n in item['graph']['nodes'] if n['id']=='scale')['data']['annotation'] = 'Multiplies each embedding by √width, changing its magnitude before the residual stream. This does not change the shape or add learned parameters.'
            next(n for n in item['graph']['nodes'] if n['id']=='lanes')['data']['annotation'] = 'The model carries multiple residual lanes, allowing later routing and mixing between streams.'
        if item['id']=='thimble':
            item['description'] = 'Thimble v6 transformer trunk · 20 layers, width 448, GQA 8/4. Tool decoder and specialized heads are outside this graph.'
            next(n for n in item['graph']['nodes'] if n['id']=='stack')['data']['annotation'] = 'The published v6 trunk is deep and narrow: 20 layers at width 448. Its tool-calling behavior also relies on a specialized tokenizer and constrained decoder, which are outside this graph.'
            next(n for n in item['graph']['nodes'] if n['id']=='head')['data']['annotation'] = 'The vocabulary head reuses the token embedding weights, reducing the number of separate parameters.'
            item['references'].append({'title':'Thimble v6 model card','url':'https://huggingface.co/flashvenom/thimble','notes':'The v6 release describes the same trunk configuration. The complete tool-calling system is not represented.'})
    for item in items:
        if item['id'] in source_urls:
            for n in item['graph']['nodes']:
                if n['data'].get('annotation'):n['data']['annotationSource']=source_urls[item['id']]
    items.append(project('gpt2','GPT-2 · 124M','GPT-2 small: learned positions, LayerNorm and GELU, with attention and residual dropout.',gpt2_model(),
        config={'width':768,'heads':12,'ffn_width':3072,'vocab':50257,'mask_id':3},
        references=[{'title':'GPT-2 model configuration','url':'https://huggingface.co/openai-community/gpt2/blob/main/config.json','notes':'Random-weight architecture; checkpoint loading, padding masks and KV caching are not implemented.'},{'title':'GPT-2 implementation','url':'https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/gpt2/modeling_gpt2.py'}]))
    gpt=next(item for item in items if item['id']=='gpt2')
    position=next(n for n in gpt['graph']['nodes'] if n['id']=='pos')
    position['data'].update(annotation='GPT-2 adds a learned vector for each position to the token embedding before its decoder layers.',annotationSource='https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/gpt2/modeling_gpt2.py')
    items.append(project('sparse','Sparse attention composition','Causal attention with a union of local windows and global anchor tokens. Patterns are editable graph nodes.',sparse_transformer()))
    items.append(project('embedding-table','Embedding table','Token IDs index a learned table of vectors.',embedding_example()))
    items.append(project('moe-transformer','MoE transformer','A causal attention block followed by four editable experts and top-2 token routing.',moe_transformer(),references=[{'title':'Switch Transformer: top-1 routing and balancing','url':'https://arxiv.org/abs/2101.03961'},{'title':'Training MoEs at Scale with PyTorch','url':'https://pytorch.org/blog/training-moes/'}]))
    items.append(project('blank','Blank project','Start with token input and build your own architecture.',{'nodes':[node('ids','input','Token IDs',80,150,{'index':0,'shape':['B','T'],'dtype':'int64'}),node('out','output','Output',700,150)],'edges':[]}))
    return items
