"""Lower an attention node into the same editable primitive graph used everywhere."""
from .templates import node, edge


def expand_attention(p, width, cross=False):
    heads=p.get('heads',8);kv=p.get('kv_heads',heads);hd=p.get('head_dim',width//heads);vd=p.get('value_dim',width//heads)
    ns=[node('x','input','Queries / hidden state',0,180,{'index':0,'shape':['B','T',width],'dtype':'float32'})]
    es=[]
    if cross:ns.append(node('memory','input','Key/value memory',0,600,{'index':1,'shape':['B','S',width],'dtype':'float32'}))
    mask_index=2 if cross else 1
    ns.append(node('mask','input','Attention pattern · optional',1750,-320,{'index':mask_index,'shape':['T','S' if cross else 'T'],'dtype':'attention_pattern','optional':True}))
    branches={}
    for i,(key,count,dim) in enumerate([('q',heads,hd),('k',kv,hd),('v',kv,vd)]):
        y=i*280;src='memory' if cross and key!='q' else 'x'
        ns.append(node(key,'linear',key.upper()+' projection',300,y,{'out':count*dim}));es.append(edge(src,key));last=key
        if p.get('conv_taps',0):
            ns.append(node(key+'conv','causal_conv',key.upper()+' causal convolution',600,y,{'taps':p['conv_taps']}));es.append(edge(last,key+'conv'));last=key+'conv'
        ns.append(node(key+'heads','split_heads',key.upper()+' · split heads',900,y,{'heads':count}));es.append(edge(last,key+'heads'));last=key+'heads'
        if key!='v' and p.get('qk_norm',False):
            ns.append(node(key+'norm','norm',key.upper()+' RMSNorm',1200,y));es.append(edge(last,key+'norm'));last=key+'norm'
        if key!='v' and p.get('rope',True):
            ns.append(node(key+'rope','rope',key.upper()+' rotary positions',1500,y,{'theta':p.get('theta',10000.)}));es.append(edge(last,key+'rope'));last=key+'rope'
        if key!='q' and heads!=kv:
            ns.append(node(key+'repeat','repeat_heads',key.upper()+' · share KV heads',1800,y,{'repeats':heads//kv}));es.append(edge(last,key+'repeat'));last=key+'repeat'
        branches[key]=last
    ns.extend([node('sdpa','sdpa','Scaled dot-product attention',2100,180,{'causal':p.get('causal',not cross),'window':p.get('window',0),'backend':p.get('backend','dense')}),node('merge','merge_heads','Merge attention heads',2420,180)])
    es.extend([edge(branches['q'],'sdpa'),edge(branches['k'],'sdpa','b'),edge(branches['v'],'sdpa','c'),edge('mask','sdpa','d'),edge('sdpa','merge')]);last='merge'
    if p.get('gated',False):
        ns.extend([node('gateproj','linear','Output gate projection',1800,900,{'out':heads*vd}),node('sigmoid','sigmoid','Sigmoid gate',2120,900),node('gated','multiply','Gate attention values',2740,180)])
        es.extend([edge('x','gateproj'),edge('gateproj','sigmoid'),edge('merge','gated'),edge('sigmoid','gated','b')]);last='gated'
    ns.extend([node('o','linear','Output projection',3060,180,{'out':width}),node('out','output','Attention output',3360,180)])
    es.extend([edge(last,'o'),edge('o','out')])
    return {'nodes':ns,'edges':es}
