"""Graph contracts and shape analysis. This module never imports user code."""
from __future__ import annotations

import ast
import copy
import re
import math
from .patterns import validate_pattern,validate_mask
from .advice import design_advice

VERSION = 1


class ContractError(ValueError):
    """A contract failure at a specific input port."""

    def __init__(self, message, port):
        super().__init__(message)
        self.port = port


def valid_dimension(value):
    return (type(value) is int and value > 0 or
            isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(?:\+[1-9][0-9]*)*', value) is not None)
PORTS = {
    'input': [], 'output': ['a'], 'embedding': ['a'], 'linear': ['a'],
    'norm': ['a'], 'silu': ['a'], 'sigmoid': ['a'], 'add': ['a', 'b'],
    'multiply': ['a', 'b'], 'concat': ['a', 'b'], 'attention': ['a','b'],
    'cross_attention': ['a', 'b','c'], 'ffn': ['a'], 'ngram': ['a'],
    'lookup': ['a'], 'mask': ['a'], 'select': ['a', 'b'],
    'loop': ['a'], 'block': ['a'], 'custom': ['a'], 'reference': ['a'],
    'gate': ['a', 'b'], 'scale': ['a'], 'hadamard': ['a'],
    'mhc': ['a'], 'engram': ['a', 'b'], 'head': ['a'],
    'lane_expand':['a'], 'lane_reduce':['a'], 'routing':['a'],
    'lane_mix':['a','b'], 'lane_merge':['a','b','c'], 'subtract':['a','b'],
    'engram_hash':['a'], 'engram_lookup':['a'], 'memory_fuse':['a','b'], 'monarch':['a'],
    'split_heads':['a'], 'merge_heads':['a'], 'repeat_heads':['a'], 'rope':['a'],
    'sdpa':['a','b','c','d'], 'causal_conv':['a'], 'softmax':['a'], 'matmul':['a','b'], 'transpose':['a'],
    'attention_pattern':['a','b'], 'pattern_union':['a','b'], 'pattern_intersection':['a','b'],
}


def optional_ports(node):
    d=node['data'];kind=d['kind']
    if d.get('body'):
        return [chr(97+int(n['data'].get('params',{}).get('index',0))) for n in d['body']['nodes'] if n['data']['kind']=='input' and n['data'].get('params',{}).get('optional')]
    return {'attention':['b'],'cross_attention':['c'],'sdpa':['d'],'attention_pattern':['b']}.get(kind,[])


def definition_signature(node,config):
    d=node['data'];p=params(node,config)
    signature={'kind':d['kind'],'params':p}
    if d.get('body'):
        body=d['body'];scope={**config,**p}
        signature['body']={'nodes':[(n['id'],definition_signature(n,scope),n['data'].get('share')) for n in sorted(body['nodes'],key=lambda n:n['id'])],
            'edges':sorted((e['source'],e.get('sourceHandle') or 'out',e['target'],e.get('targetHandle') or 'a') for e in body['edges'])}
    return signature


def resolve(value, config):
    if value == '$sqrt_width':
        return config.get('width',128)**.5
    if isinstance(value, str) and value.startswith('$'):
        return config.get(value[1:])
    if isinstance(value, dict):
        return {k: resolve(v, config) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, config) for v in value]
    return value


def params(node, config):
    return resolve(node['data'].get('params', {}), config)


def port_names(node):
    d = node['data']
    if d['kind']=='custom' and d.get('params',{}).get('input_shapes'):
        return [chr(97+i) for i in range(len(d['params']['input_shapes']))]
    if d['kind'] in ('block', 'loop') and d.get('body'):
        return [chr(97 + i) for i in range(len([n for n in d['body']['nodes'] if n['data']['kind'] == 'input']))]
    return PORTS.get(d['kind'], ['a'])


def ordered(graph):
    nodes = {n['id']: n for n in graph.get('nodes', [])}
    if len(nodes) != len(graph.get('nodes', [])):
        raise ValueError('Duplicate node IDs.')
    deps = {k: set() for k in nodes}
    for e in graph.get('edges', []):
        if e['source'] not in nodes or e['target'] not in nodes:
            raise ValueError('Connection references a missing node.')
        deps[e['target']].add(e['source'])
    result = []
    while deps:
        ready = [k for k, v in deps.items() if not v]
        if not ready:
            raise ValueError('A direct cycle was found. Put feedback inside a Loop container.')
        for k in ready:
            result.append(nodes[k]); del deps[k]
        for v in deps.values():
            v.difference_update(ready)
    return result


def validate_project(project):
    errors, warnings, shapes, counts, raw_counts = [], [], {}, {}, {}
    cfg = project.get('config', {})
    for field in ('width','vocab','heads','ffn_width'):
        if field in cfg and (not isinstance(cfg[field],int) or isinstance(cfg[field],bool) or cfg[field]<1):
            errors.append({'node':None,'message':f'Model {field} must be a positive integer.'})
    shared = {}
    if project.get('version') != VERSION:
        errors.append({'node': None, 'message': 'Unsupported project version.'})
    tok = project.get('tokenizer')
    if tok and cfg.get('vocab', 0) < tok.get('capacity', 0):
        errors.append({'node': None, 'message': f"Tokenizer needs {tok['capacity']} embedding rows; vocabulary capacity is {cfg.get('vocab')} ."})

    def visit(graph, inputs=None, prefix='', config=None):
        cfg=config or project.get('config',{})
        local = {}
        total = 0
        outputs = []
        def fail(n, message, port=None):
            errors.append({'node': prefix + n['id'], 'port': port, 'message': message})
        try:
            sequence = ordered(graph)
        except (ValueError, KeyError) as exc:
            errors.append({'node': prefix.rstrip('/') or None, 'message': str(exc)})
            return [], 0
        for n in sequence:
            key = prefix + n['id']; d = n['data']; kind = d.get('kind'); p = params(n, cfg)
            incoming = {}
            for e in graph.get('edges', []):
                if e['target'] == n['id']:
                    port = e.get('targetHandle') or 'a'
                    if port in incoming:
                        fail(n, f'Input {port} has multiple connections.', port)
                    if port not in port_names(n):
                        fail(n, f'Unknown input port {port}.', port)
                    incoming[port] = local.get(e['source'])
                    if e.get('sourceHandle') not in (None,'out'):
                        fail(n,'Connection uses an unknown source port.',port)
            args = [incoming.get(k) for k in port_names(n)]
            out = None; count = 0
            try:
                if d.get('iteration'):
                    schedule=d['iteration'];index=schedule.get('index');steps=schedule.get('total')
                    if not isinstance(index,int) or not isinstance(steps,int) or not 0<=index<steps:raise ValueError('Application schedule needs 0 ≤ index < total.')
                if kind not in PORTS:
                    raise ValueError(f'Unknown operation: {kind}')
                required=[a for port,a in zip(port_names(n),args) if port not in optional_ports(n)]
                for port, arg in zip(port_names(n), args):
                    if arg is None and port not in optional_ports(n):
                        connected = next((e for e in graph.get('edges', []) if e['target'] == n['id'] and (e.get('targetHandle') or 'a') == port), None)
                        message = (f"Input {port} is connected to {connected['source']}, whose shape is unresolved. Fix the upstream error first."
                                   if connected else f'Input {port} is unconnected. Connect a tensor to this required port.')
                        fail(n, message, port)
                x = args[0] if args else None
                width = p.get('width', cfg.get('width', 128))
                if kind == 'input':
                    idx = int(p.get('index', 0))
                    out = inputs[idx] if inputs is not None and idx < len(inputs) else None if p.get('optional') else {'shape': resolve(p.get('shape', ['B', 'T']), cfg), 'dtype': p.get('dtype', 'int64')}
                    if inputs is not None and idx >= len(inputs) and not p.get('optional'):
                        raise ValueError(f'Input index {idx} is outside the container interface.')
                    if out and out['dtype'] not in ('int64','float32','bool','attention_pattern'):raise ValueError('Unsupported input dtype.')
                elif any(a is None for a in required):
                    pass
                elif kind == 'output':
                    out = x; outputs.append((p.get('index', 0), out))
                elif kind in ('block', 'loop'):
                    if not d.get('body'):
                        raise ValueError('Container has no internal graph.')
                    nested, count = visit(d['body'], args, key + '/', {**cfg,**p})
                    if len(nested) != 1:
                        raise ValueError('A container must have exactly one output.')
                    out = nested[0]
                    if kind == 'loop':
                        if not isinstance(p.get('iterations', 1), int) or p.get('iterations', 1) < 1:
                            raise ValueError('Loop iterations must be a positive integer.')
                        if out != x:
                            raise ValueError(f'Loop state must preserve shape and dtype: {x} → {out}.')
                        if not p.get('shared', True):
                            count *= p.get('iterations', 1)
                elif kind in ('embedding', 'ngram', 'engram_hash', 'mask'):
                    if x['dtype'] != 'int64' or len(x['shape']) != 2:
                        raise ValueError('Expected token IDs with shape [B, T] and dtype int64.')
                    if kind == 'embedding':
                        if not isinstance(width,int) or width<1:raise ValueError('Embedding width must be a positive integer.')
                        capacity = p.get('vocab', cfg.get('vocab'))
                        if not isinstance(capacity, int) or capacity < 1:
                            raise ValueError('Embedding vocabulary capacity must be positive.')
                        out = {'shape': x['shape'] + [width], 'dtype': 'float32'}
                        count = capacity * width
                        if tok and capacity < tok.get('capacity',0):
                            raise ValueError(f'Tokenizer needs {tok["capacity"]} rows, but this embedding has {capacity}.')
                    elif kind in ('ngram','engram_hash'):
                        orders = p.get('orders', [2, 3]); buckets = p.get('buckets', 8192)
                        if not orders or any(not isinstance(o, int) or o < 1 for o in orders) or not isinstance(buckets,int) or buckets < 1:
                            raise ValueError('N-gram orders and hash bucket count must be positive integers.')
                        if kind=='engram_hash' and (not isinstance(p.get('hash_heads',1),int) or p.get('hash_heads',1)<1):raise ValueError('Hash head count must be a positive integer.')
                        out = {'shape': x['shape'] + [len(orders)*(p.get('hash_heads',1) if kind=='engram_hash' else 1)], 'dtype': 'int64'}
                        out.update(buckets=buckets,orders=orders,hash_heads=p.get('hash_heads',1))
                    else:
                        mask = p.get('mask_id', cfg.get('mask_id'))
                        if not isinstance(mask, int) or not 0 <= mask < cfg.get('vocab', 0):
                            raise ValueError('Set a mask token ID within the vocabulary capacity.')
                        shape = list(x['shape']); length = p.get('length', 0)
                        if not isinstance(length,int) or length < 0:
                            raise ValueError('Generated token length must be a non-negative integer.')
                        if length:
                            shape[1] = shape[1] + length if isinstance(shape[1], int) else f'{shape[1]}+{length}'
                        out = {'shape': shape, 'dtype': 'int64'}
                elif kind == 'lookup':
                    if x['dtype'] != 'int64' or len(x['shape']) != 3 or not isinstance(x['shape'][-1], int):
                        raise ValueError('Expected hashed n-gram indices [B, T, orders].')
                    norders = x['shape'][-1]; buckets = p.get('buckets', 8192)
                    if not isinstance(buckets,int) or buckets<1 or not isinstance(width,int) or width<1:raise ValueError('Lookup buckets and feature width must be positive integers.')
                    if x.get('buckets') is not None and x['buckets']!=buckets:raise ValueError('Hash and lookup bucket counts differ.')
                    # A lookup must use the same bucket count as the hash operation.
                    edge = next(e for e in graph['edges'] if e['target'] == n['id'] and (e.get('targetHandle') or 'a') == 'a')
                    upstream = next(v for v in graph['nodes'] if v['id'] == edge['source'])
                    if upstream['data']['kind'] == 'ngram' and params(upstream, cfg).get('buckets', 8192) != buckets:
                        raise ValueError('Hash and lookup bucket counts differ.')
                    out = {'shape': x['shape'][:-1] + [width], 'dtype': 'float32'}
                    count = norders * buckets * width
                elif kind in ('lane_expand','lane_reduce','routing','lane_mix','lane_merge','engram_lookup','memory_fuse'):
                    lanes=p.get('lanes',4)
                    if kind=='lane_expand':
                        if x['dtype']!='float32' or len(x['shape'])!=3:raise ValueError('Lane expansion expects [B,T,D] features.')
                        out={'shape':x['shape'][:-1]+[lanes,x['shape'][-1]],'dtype':'float32'}
                    elif kind in ('lane_reduce','routing','lane_mix','lane_merge'):
                        if x['dtype']!='float32' or len(x['shape'])!=4 or x['shape'][-2]!=lanes:raise ValueError(f'Expected [B,T,{lanes},D] residual lanes.')
                        dim=x['shape'][-1]
                        out={'shape':x['shape'][:2]+[dim],'dtype':'float32'}
                        routing_shape=x['shape'][:2]+[lanes*(lanes+2)]
                        if kind=='routing':
                            out['shape']=routing_shape;count=dim*lanes*lanes*(lanes+2)+2*lanes+lanes*lanes+3
                        if kind=='lane_mix' and args[1]['shape']!=routing_shape:raise ValueError('Routing coefficients do not match residual lanes.')
                        if kind=='lane_merge':
                            if args[1]['shape']!=out['shape'] or args[2]['shape']!=routing_shape:raise ValueError('Lane merge needs matching update and routing tensors.')
                            out=x
                    elif kind=='engram_lookup':
                        orders=p.get('orders',[2,3]);heads=p.get('hash_heads',2);sub=p.get('sub_dim',128);slots=p.get('buckets',8192)
                        if x['dtype']!='int64' or len(x['shape'])!=3 or x['shape'][-1]!=len(orders)*heads:raise ValueError('Engram lookup table count does not match hashed indices.')
                        if x.get('buckets') is not None and x['buckets']!=slots:raise ValueError('Engram hash and lookup bucket counts differ.')
                        if x.get('orders') is not None and (x['orders']!=orders or x.get('hash_heads')!=heads):raise ValueError('Engram hash orders or head count differ from lookup tables.')
                        out={'shape':x['shape'][:2]+[2*width],'dtype':'float32'}
                        count=len(orders)*heads*slots*sub + 2*len(orders)*heads*sub*width+4*width
                    else:
                        if x['dtype']!='float32' or args[1]['shape']!=x['shape'][:-1]+[2*x['shape'][-1]]:raise ValueError('Engram fusion needs features [B,T,D] and packed key/value [B,T,2D].')
                        out=x
                elif kind in ('add', 'multiply', 'subtract', 'gate', 'concat'):
                    y = args[1]
                    if x['dtype'] != y['dtype'] or len(x['shape']) != len(y['shape']):
                        raise ContractError(f'Inputs have incompatible shape or dtype: {x} and {y}.', 'b')
                    if kind == 'concat':
                        if x['shape'][:-1] != y['shape'][:-1]:
                            raise ContractError(f'Concatenation requires matching leading dimensions: {x["shape"]} and {y["shape"]}.', 'b')
                        out = {'shape': x['shape'][:-1] + [x['shape'][-1] + y['shape'][-1]], 'dtype': x['dtype']}
                    else:
                        if x != y:
                            raise ContractError(f'Expected {x["shape"]}, received {y["shape"]} at input b. Both inputs must have matching shape and dtype.', 'b')
                        out = x
                        if kind == 'gate': count = 1
                elif kind == 'select':
                    y = args[1]
                    if x['dtype'] != 'float32' or y['dtype'] != 'int64' or x['shape'][:-1] != y['shape'] or x['shape'][-1] != cfg.get('vocab'):
                        raise ValueError('Selection requires logits [B,T,V] and current token IDs [B,T], with V matching the vocabulary.')
                    if not isinstance(cfg.get('mask_id'), int) or not 0 <= cfg['mask_id'] < cfg.get('vocab', 0):
                        raise ValueError('Selection requires a valid mask token ID.')
                    out = y
                elif kind == 'custom':
                    source = p.get('source', '')
                    tree = ast.parse(source)
                    if not any(isinstance(t, ast.ClassDef) and t.name == 'CustomBlock' for t in tree.body):
                        raise ValueError('Custom Python must define class CustomBlock(nn.Module).')
                    expected = resolve(p.get('input_shape', ['B', 'T', '$width']), cfg)
                    contracts=resolve(p.get('input_shapes') or [expected],cfg)
                    if any(a['shape']!=s for a,s in zip(args,contracts)):
                        raise ValueError(f'Custom shape contract expects {contracts}, received {[a["shape"] for a in args]}.')
                    if p.get('input_dtypes') and [a['dtype'] for a in args]!=p['input_dtypes']:raise ValueError('Custom input dtypes do not match the declared contract.')
                    out = {'shape': resolve(p.get('output_shape', expected), cfg), 'dtype': p.get('output_dtype', x['dtype'])}
                    count=p.get('parameter_count',0)
                    if not isinstance(count,int) or count<0:raise ValueError('Custom parameter count must be a non-negative integer.')
                    warnings.append({'node': key, 'message': 'Custom Python shapes and parameter counts are declared contracts; code is not executed by the server.'})
                elif kind == 'reference':
                    raise ValueError(d.get('description', 'Reference-only operation: exact implementation is not available.'))
                elif kind=='attention_pattern':
                    validate_pattern(p)
                    y=args[1] or x
                    if x['dtype']!='float32' or y['dtype']!='float32' or len(x['shape']) not in (3,4) or len(y['shape'])!=len(x['shape']):raise ValueError('Pattern inputs must be features [B,T,D] or heads [B,H,T,D].')
                    if x['shape'][:-2]!=y['shape'][:-2]:raise ValueError('Pattern query/key batch and head dimensions must match.')
                    if p.get('pattern')=='content_topk' and x['shape'][-1]!=y['shape'][-1]:raise ValueError('Content top-k requires matching query/key feature widths.')
                    out={'shape':[x['shape'][-2],y['shape'][-2]],'dtype':'attention_pattern'}
                    if p.get('pattern')=='content_topk':out['shape']=[x['shape'][0],x['shape'][1] if len(x['shape'])==4 else 1]+out['shape']
                    if p.get('pattern')=='custom':
                        matrix_shape=[len(p['matrix']),len(p['matrix'][0])]
                        if out['shape']!=matrix_shape:raise ValueError('Custom matrix requires fixed query/key lengths matching its dimensions. Set explicit input sequence lengths.')
                        out['shape']=matrix_shape
                elif kind in ('pattern_union','pattern_intersection'):
                    y=args[1]
                    if x['dtype'] not in ('attention_pattern','bool') or y['dtype'] not in ('attention_pattern','bool') or x['shape'][-2:]!=y['shape'][-2:]:raise ValueError('Combine patterns with matching query/key dimensions.')
                    out={'shape':x['shape'] if len(x['shape'])>=len(y['shape']) else y['shape'],'dtype':'attention_pattern'}
                    if len(x['shape'])==len(y['shape'])==4:
                        prefix_dims=[]
                        for a,b in zip(x['shape'][:2],y['shape'][:2]):
                            if a!=b and a!=1 and b!=1:raise ValueError('Combined content pattern batch/head dimensions cannot broadcast.')
                            prefix_dims.append(b if a==1 else a)
                        out['shape']=prefix_dims+x['shape'][-2:]
                elif kind in ('split_heads','merge_heads','repeat_heads','rope','sdpa','causal_conv','softmax','matmul','transpose'):
                    if x['dtype']!='float32':raise ValueError('Tensor computation expects floating point inputs.')
                    out=copy.deepcopy(x);s=x['shape']
                    if kind=='split_heads':
                        h=p.get('heads',cfg.get('heads',8))
                        if len(s)!=3 or not isinstance(h,int) or h<1 or s[-1]%h:raise ValueError('Split heads requires [B,T,D] and a head count dividing D.')
                        out['shape']=[s[0],h,s[1],s[2]//h]
                    elif kind=='merge_heads':
                        if len(s)!=4:raise ValueError('Merge heads expects [B,H,T,D].')
                        out['shape']=[s[0],s[2],s[1]*s[3]]
                    elif kind=='repeat_heads':
                        repeats=p.get('repeats',1)
                        if len(s)!=4 or not isinstance(repeats,int) or repeats<1:raise ValueError('Head repetition needs [B,H,T,D] and a positive repeat count.')
                        out['shape'][1]*=repeats
                    elif kind=='rope':
                        if len(s)!=4 or s[-1]%2:raise ValueError('RoPE expects [B,H,T,D] with even D.')
                    elif kind=='sdpa':
                        k,v=args[1:3]
                        if len(s)!=4 or len(k['shape'])!=4 or len(v['shape'])!=4 or s[:2]!=k['shape'][:2] or k['shape'][:3]!=v['shape'][:3] or s[-1]!=k['shape'][-1] or k['dtype']!='float32' or v['dtype']!='float32':raise ValueError('SDPA requires Q[B,H,T,D], K[B,H,S,D], V[B,H,S,Dv].')
                        out['shape'][-1]=v['shape'][-1]
                        if args[3]:
                            try: validate_mask(args[3],s,k['shape'])
                            except ValueError as exc: raise ContractError(str(exc), 'd') from exc
                        if p.get('backend','dense') not in ('dense','flex','auto'):raise ValueError('Attention backend must be dense, flex, or auto.')
                    elif kind=='causal_conv':
                        taps=p.get('taps',3)
                        if len(s)!=3 or not isinstance(taps,int) or taps<1:raise ValueError('Causal convolution expects [B,T,D] and a positive tap count.')
                        count=taps*s[-1]
                    elif kind=='matmul':
                        if args[1]['dtype'] != 'float32':
                            raise ContractError('Matmul input b must be floating point, matching input a.', 'b')
                        y=args[1]['shape']
                        if len(s)<2 or len(y)<2 or s[:-2]!=y[:-2] or s[-1]!=y[-2]:raise ValueError(f'Matmul dimensions are incompatible: {s} × {y}.')
                        out['shape']=s[:-1]+[y[-1]]
                    elif kind=='transpose':
                        a,b=p.get('dim0',-2),p.get('dim1',-1)
                        if not -len(s)<=a<len(s) or not -len(s)<=b<len(s):raise ValueError('Transpose axis is outside tensor rank.')
                        out['shape'][a],out['shape'][b]=s[b],s[a]
                    elif kind=='softmax':
                        if not -len(s)<=p.get('dim',-1)<len(s):raise ValueError('Softmax axis is outside tensor rank.')
                else:
                    if x['dtype'] != 'float32' or len(x['shape']) < 2:
                        raise ValueError('Expected floating point features.')
                    dim = x['shape'][-1]
                    if not isinstance(dim, int) or dim < 1:
                        raise ValueError('Feature width is unresolved or invalid.')
                    out = copy.deepcopy(x)
                    if kind in ('linear', 'head'):
                        outdim = p.get('out', cfg.get('vocab') if kind == 'head' else width)
                        if not isinstance(outdim, int) or outdim < 1:
                            raise ValueError('Output width must be a positive integer.')
                        out['shape'][-1] = outdim; count = dim * outdim + (outdim if p.get('bias', False) else 0)
                        if kind=='head' and tok and outdim<tok.get('capacity',0):raise ValueError('Vocabulary output head cannot represent all tokenizer IDs.')
                        if p.get('tie'):
                            target = next((v for v in graph['nodes'] if v['id'] == p['tie']), None)
                            if not target or target['data']['kind'] != 'embedding':
                                raise ValueError('Tied output must reference an embedding in this graph.')
                            tp = params(target, cfg)
                            if tp.get('width', cfg.get('width')) != dim or tp.get('vocab', cfg.get('vocab')) != outdim:
                                raise ValueError('Tied output dimensions must match the embedding table.')
                            count = 0
                    elif kind in ('attention', 'cross_attention'):
                        if len(x['shape']) != 3:
                            raise ValueError('Attention expects [B,T,D].')
                        heads = p.get('heads', cfg.get('heads', 8)); kv = p.get('kv_heads', heads)
                        if not isinstance(heads, int) or not isinstance(kv, int) or heads < 1 or kv < 1 or heads % kv:
                            raise ValueError('Query heads must be a positive multiple of KV heads.')
                        hd = p.get('head_dim', dim // heads); vd = p.get('value_dim', dim // heads)
                        if 'head_dim' not in p and dim % heads:
                            raise ValueError(f'Width {dim} is not divisible by {heads} heads.')
                        if not isinstance(hd,int) or not isinstance(vd,int) or hd < 2 or hd % 2 or vd < 1:
                            raise ValueError('Q/K head width must be a positive even integer; value width must be positive.')
                        if any(not isinstance(p.get(v,0),int) or p.get(v,0)<0 for v in ('window','conv_taps')):raise ValueError('Attention window and convolution taps must be non-negative integers.')
                        if kind == 'cross_attention' and (len(args[1]['shape']) != 3 or args[1]['shape'][0] != x['shape'][0] or args[1]['shape'][-1] != dim or args[1]['dtype'] != 'float32'):
                            raise ContractError(f'Cross-attention memory must be [B,S,D] with batch {x["shape"][0]} and width {dim}; received {args[1]}.', 'b')
                        mask=args[2] if kind=='cross_attention' else args[1]
                        if mask:
                            qshape=[x['shape'][0],heads,x['shape'][1],hd]
                            memory=args[1] if kind=='cross_attention' else x
                            try: validate_mask(mask,qshape,[memory['shape'][0],heads,memory['shape'][1],hd])
                            except ValueError as exc: raise ContractError(str(exc), 'c' if kind == 'cross_attention' else 'b') from exc
                        if p.get('backend','dense') not in ('dense','flex','auto'):raise ValueError('Attention backend must be dense, flex, or auto.')
                        count = dim * (heads * hd + kv * hd + kv * vd + heads * vd)
                        if p.get('qk_norm', False): count += 2 * hd
                        if p.get('gated', False): count += dim * heads * vd
                        count += p.get('conv_taps',0)*(heads*hd+kv*hd+kv*vd)
                    elif kind == 'ffn':
                        hidden = p.get('hidden', dim * 4)
                        if not isinstance(hidden, int) or hidden < 1: raise ValueError('FFN hidden width must be positive.')
                        count = 3 * dim * hidden
                    elif kind == 'norm':
                        if not isinstance(p.get('eps',1e-6),(int,float)) or p.get('eps',1e-6)<=0:raise ValueError('Normalization epsilon must be positive.')
                        count = dim
                    elif kind=='scale':
                        factor=p.get('factor',1)
                        if not isinstance(factor,(int,float)) or not math.isfinite(factor):raise ValueError('Scale factor must be a finite number.')
                    elif kind == 'hadamard': count = 4 * (1 << (dim - 1).bit_length())
                    elif kind == 'monarch':
                        size=1 << (dim-1).bit_length();ba=1 << ((size-1).bit_length()//2);bb=size//ba
                        count=3*(ba*ba+bb*bb)+5*size+8*(dim+size)
                    elif kind in ('mhc', 'engram'):
                        raise ValueError('This reference component requires its architecture-specific implementation before export.')
                if out and (not isinstance(out.get('shape'), list) or not out['shape'] or any(not valid_dimension(v) for v in out['shape'])):
                    raise ValueError(f"Invalid or unresolved dimensions {out.get('shape')}. Use positive integers or symbols such as B, T, S; check $parameter bindings.")
                if out and out['dtype'] not in ('float32','int64','bool','attention_pattern'):raise ValueError('Unsupported output dtype.')
                share = d.get('share')
                raw_counts[key]=count
                if share:
                    config_key={k:cfg.get(k) for k in ('width','heads','ffn_width','vocab','mask_id')}
                    feature_dim=None if kind=='embedding' else (x['shape'][-1] if x else None)
                    signature = (definition_signature(n,cfg), config_key, feature_dim)
                    if share in shared and shared[share] != signature:
                        raise ValueError('Shared module instances must have identical operation, parameters, and output shape.')
                    if share in shared: count = 0
                    else: shared[share] = signature
            except (ValueError, TypeError, KeyError, IndexError, SyntaxError) as exc:
                fail(n, str(exc), getattr(exc, 'port', None)); out = None
            local[n['id']] = out
            shapes[key] = out; counts[key] = count; total += count
        if not any(n['data']['kind']=='output' for n in graph['nodes']):
            errors.append({'node': prefix.rstrip('/') or None, 'message': 'Graph needs an Output node.'})
        indices = sorted(int(params(n, cfg).get('index', 0)) for n in graph['nodes'] if n['data']['kind'] == 'input')
        if indices != list(range(len(indices))):
            errors.append({'node': prefix.rstrip('/') or None, 'message': 'Input indices must be unique and consecutive from zero.'})
        return [s for _, s in sorted(outputs, key=lambda item: item[0])], total

    try:
        outputs, total = visit(project.get('graph', {}))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append({'node': None, 'message': f'Malformed graph: {exc}'})
        outputs, total = [], 0
    if not errors:
        allocated=set()
        def allocation(graph,prefix='',multiple=1,config=None):
            config=config or project.get('config',{})
            result=0
            for n in ordered(graph):
                d=n['data'];key=prefix+n['id'];share=d.get('share');p=params(n,config)
                if share and share in allocated:
                    count=0
                else:
                    if share:allocated.add(share)
                    factor=1 if share else multiple
                    if d.get('body'):
                        repetitions=p.get('iterations',1) if d['kind']=='loop' and not p.get('shared',True) else 1
                        count=allocation(d['body'],key+'/',factor*repetitions,{**config,**p})
                    else:count=raw_counts.get(key,0)*factor
                counts[key]=count;result+=count
            return result
        total=allocation(project['graph'])
    # Advice is independent of export validity, and only inspects resolved nodes.
    # Skip structurally malformed graphs rather than hiding the original error.
    if not any(e.get('node') is None for e in errors):
        warnings.extend(design_advice(project, shapes, counts, resolve))
    return {'valid': not errors, 'errors': errors, 'warnings': warnings, 'shapes': shapes, 'counts': counts, 'parameters': total, 'outputs': outputs}
