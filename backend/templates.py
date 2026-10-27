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
        ns.extend([node('memory','input','Encoder memory',260,460,{'index':1,'shape':['B','S','$width'],'dtype':'float32'}),node('cn','norm','Cross-attention norm',1300,120),node('cross','cross_attention','Cross attention',1560,120,{'heads':heads,'kv_heads':kv,'causal':False,'rope':False}),node('cr','gate','Cross residual',1820,120)])
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
        node('attn','attention',('Global' if global_layer else 'Local')+' GQA attention',1550,160,{'heads':heads,'kv_heads':kv,'head_dim':conf.get('qk_head_dim',width//heads),'value_dim':conf.get('value_head_dim',width//heads),'causal':True,'qk_norm':True,'gated':True,'theta':conf['rope_theta'],'window':0 if global_layer else conf.get('sliding_window',0),'conv_taps':conf['extras'].get('qkv_conv_taps',0)}),
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
    items.append(project('needle1','Needle 1','Encoder–decoder, pure attention, 12 encoder + 8 decoder layers. No FFN.',n1,config={'width':512,'heads':8,'ffn_width':1024,'vocab':8192,'mask_id':3},references=[{'title':'Needle 1 model card and config','url':'https://huggingface.co/Cactus-Compute/needle/tree/5f89b4307696d669c3df1d38ae057e6e1728b107','revision':'5f89b4307696d669c3df1d38ae057e6e1728b107','notes':'Documented attention-only topology. PyTorch float32 construction; no QAT, pretrained weights, KV caching, or original zero-centered initialization.'}]))
    for version,layers,width,heads,kv,rev in [(2,27,512,8,4,'32e9e3a93b205f786929697446ae669cf0a84579'),(3,20,768,12,2,'d6c7d6a2e9655505e0f8e23339356dd1705900eb')]:
        conf=json.loads((Path(__file__).parent.parent/'references'/f'needle{version}_config.json').read_text())
        g=needle_network(conf)
        items.append(project(f'needle{version}',f'Needle {version}',f'{layers} layers · {heads}/{kv} GQA · Engram memory · 4 residual lanes.',g,config={'width':width,'heads':heads,'ffn_width':width*2,'vocab':8192,'mask_id':3},references=[{'title':f'Needle {version} published configuration','url':f'https://huggingface.co/Cactus-Compute/needle{version}/tree/{rev}','revision':rev,'config':conf,'notes':'Published backbone settings with a float32 PyTorch port of Engram, learned Monarch mixing and mHC routing. Exports contain random weights, not checkpoint conversion. Quantization, padding-aware masks, auxiliary probe heads and trained ladder subnetworks are not reproduced.'},{'title':'Upstream architecture source','url':'https://github.com/cactus-compute/needle/blob/b4fc11b04f1852888d375c8c6dfbaca44572d014/needle/model/architecture.py','revision':'b4fc11b04f1852888d375c8c6dfbaca44572d014','notes':'Needle 2 uses its published configuration with the current shared SAN building blocks; historical checkpoint parity is not asserted.'}]))
    items.append(project('thimble','Thimble','Deep-thin gated transformer · sandwich RMSNorm · GQA 8/4 · tied embeddings.',language_model(transformer(sandwich=True,kv=4),20),config={'width':448,'heads':8,'ffn_width':896,'vocab':16384,'mask_id':3},references=[{'title':'Thimble PyTorch trunk','url':'https://github.com/nikshepsvn/thimble/blob/bb7d5768651d525516f3647d551005a4411845af/src/tiny_toolcall/model.py','revision':'bb7d5768651d525516f3647d551005a4411845af','notes':'Trunk computation reproduced with editable primitives. Grammar dispatcher, retrieval and auxiliary name/pointer heads are outside this trunk graph. Release vocabulary is 16K; source Config defaults to 8192.'}]))
    items.append(project('sparse','Sparse attention composition','Causal attention with a union of local windows and global anchor tokens. Patterns are editable graph nodes.',sparse_transformer()))
    items.append(project('blank','Blank project','Start with token input and build your own architecture.',{'nodes':[node('ids','input','Token IDs',80,150,{'index':0,'shape':['B','T'],'dtype':'int64'}),node('out','output','Output',700,150)],'edges':[]}))
    return items
