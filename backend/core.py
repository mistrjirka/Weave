"""Graph contracts and shape analysis. This module never imports user code."""
from __future__ import annotations

import ast
import copy
import re
import math
from .patterns import validate_pattern,validate_mask
from .advice import design_advice
from .tokenizer_bindings import bind_input_tokenizers

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
    'norm': ['a'], 'layer_norm': ['a'], 'dropout': ['a'], 'position_embedding': ['a'], 'conv1d': ['a'], 'pool1d': ['a'], 'conv2d': ['a'], 'pool2d': ['a'], 'gelu': ['a'], 'relu': ['a'], 'silu': ['a'], 'sigmoid': ['a'], 'add': ['a', 'b'],
    'multiply': ['a', 'b'], 'concat': ['a', 'b'], 'attention': ['a','b'],
    'cross_attention': ['a', 'b','c'], 'ffn': ['a'], 'ngram': ['a'],
    'lookup': ['a'], 'mask': ['a'], 'select': ['a', 'b'],
    'loop': ['a'], 'block': ['a'], 'moe': ['a'], 'moe_router': ['a'], 'moe_combine': ['a'], 'custom': ['a'], 'reference': ['a'],
    'gate': ['a', 'b'], 'scale': ['a'], 'hadamard': ['a'],
    'mhc': ['a'], 'engram': ['a', 'b'], 'head': ['a'],
    'lane_expand':['a'], 'lane_reduce':['a'], 'routing':['a'],
    'lane_mix':['a','b'], 'lane_merge':['a','b','c'], 'subtract':['a','b'],
    'engram_hash':['a'], 'engram_lookup':['a'], 'memory_fuse':['a','b'], 'monarch':['a'],
    'split_heads':['a'], 'merge_heads':['a'], 'repeat_heads':['a'], 'rope':['a'],
    'sdpa':['a','b','c','d'], 'causal_conv':['a'], 'softmax':['a'], 'matmul':['a','b'], 'transpose':['a'],
    'attention_pattern':['a','b'], 'pattern_union':['a','b'], 'pattern_intersection':['a','b'],
}



def validate_rope_settings(p):
    theta=p.get('theta',10000.)
    if not isinstance(theta,(int,float)) or isinstance(theta,bool) or not math.isfinite(theta) or theta<=1:raise ValueError('Rotary theta must be finite and greater than 1.')
    if p.get('rope_scaling','none') not in ('none','yarn'):raise ValueError('RoPE scaling must be none or yarn.')
    if p.get('rope_scaling','none')=='yarn':
        factor=p.get('rope_factor',1)
        if not isinstance(factor,(int,float)) or isinstance(factor,bool) or not math.isfinite(factor) or factor<1:raise ValueError('YaRN factor must be finite and at least 1.')
        if type(p.get('rope_original_context',32768)) is not int or p.get('rope_original_context',32768)<1:raise ValueError('YaRN original context must be positive.')


def optional_ports(node):
    d=node['data'];kind=d['kind']
    if d.get('body'):
        return [chr(97+int(n['data'].get('params',{}).get('index',0))) for n in d['body']['nodes'] if n['data']['kind']=='input' and n['data'].get('params',{}).get('optional')]
    return {'attention':['b'],'cross_attention':['c'],'sdpa':['d'],'attention_pattern':['b']}.get(kind,[])


def definition_signature(node,config):
    d=node['data'];p=params(node,config)
    signature={'kind':d['kind'],'params':p}
    if d.get('body'):
        body=d['body'];scope={**config,**p,**d['body'].get('variables', {})}
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


def variable_names(value):
    if isinstance(value, str) and re.fullmatch(r'\$[A-Za-z_][A-Za-z_0-9]*', value):
        return {value[1:]}
    if isinstance(value, dict):
        return set().union(*(variable_names(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(variable_names(item) for item in value))
    return set()


def params(node, config):
    return resolve(node['data'].get('params', {}), config)



def graph_context(project, path):
    """The same lexical variables are used by compilation and graph rewrites."""
    graph = project['graph']
    config = {**project.get('config', {}), **graph.get('variables', {})}
    for node_id in path:
        node = next((node for node in graph['nodes'] if node['id'] == node_id), None)
        if not node or not node['data'].get('body'):
            raise ValueError('The containing block is no longer available.')
        config = {**config, **params(node, config)}
        graph = node['data']['body']
        config.update(graph.get('variables', {}))
    return graph, config


def port_names(node):
    d = node['data']
    if d['kind']=='moe_combine':
        count=d.get('params',{}).get('experts',0)
        return ['a'] + [f'e{i}' for i in range(count)] if type(count) is int and 0 <= count <= 256 else ['a']
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



def moe_structure(node, settings):
    """Validate the editable routing scaffold used by the sparse dispatcher."""
    body=node['data'].get('body')
    if not body: raise ValueError('MoE needs an editable expert graph.')
    nodes=body.get('nodes',[])
    def one(kind):
        matches=[item for item in nodes if item['data'].get('kind')==kind]
        if len(matches)!=1: raise ValueError(f'MoE needs exactly one {kind.replace("_", " ")} node.')
        return matches[0]
    source,router,combine,output=(one(kind) for kind in ('input','moe_router','moe_combine','output'))
    experts=[n for n in nodes if n['data'].get('role')=='moe-expert']
    count=settings.get('experts')
    if type(count) is not int or not 1<=count<=256: raise ValueError('Expert count must be an integer between 1 and 256.')
    if len(experts)!=count or len(nodes)!=count+4 or any(n['data'].get('kind')!='block' or not n['data'].get('body') for n in experts):
        raise ValueError('The MoE graph must contain exactly its configured editable expert blocks. Change the expert count on the outer MoE block.')
    if router['data'].get('params',{}).get('experts')!=count or combine['data'].get('params',{}).get('experts')!=count:
        raise ValueError('Router and combine expert counts must match the MoE container.')
    expected={(source['id'],router['id'],'a'),(router['id'],combine['id'],'a'),(combine['id'],output['id'],'a')}
    for index, expert in enumerate(experts):
        expected.add((source['id'],expert['id'],'a'))
        expected.add((expert['id'],combine['id'],f'e{index}'))
    edges=body.get('edges',[])
    actual=[(e.get('source'),e.get('target'),e.get('targetHandle') or 'a') for e in edges]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Connect the router and every expert to the weighted combine as shown in the MoE scaffold; edit expert internals inside each expert block.')
    return router, experts

def validate_project(project):
    project, token_domains, binding_errors = bind_input_tokenizers(project, ordered)
    errors, warnings, shapes, counts, raw_counts = binding_errors, [], {}, {}, {}
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

    def visit(graph, inputs=None, prefix='', config=None, moe_body=False):
        cfg={**(config if config is not None else project.get('config',{})), **graph.get('variables', {})}
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
            tok = token_domains.get(key)
            vocabulary = tok['capacity'] if tok else cfg.get('vocab', 0)
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
                missing = sorted(variable_names(d.get('params', {})) - set(cfg) - {'sqrt_width'})
                if missing:
                    raise ValueError('Unknown variable ' + ', '.join('$' + name for name in missing) + '. Add it in Graph variables or enter a number in this block.')
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
                elif kind == 'moe':
                    if x['dtype'] != 'float32' or len(x['shape']) != 3 or type(x['shape'][-1]) is not int:
                        raise ValueError('MoE expects float features [B, T, D] with a known width.')
                    router, experts = moe_structure(n,p)
                    top_k=p.get('top_k',2); weight=p.get('balance_weight',0.01)
                    if p.get('router', 'softmax') not in ('softmax', 'sigmoid'):
                        raise ValueError('Router scoring must be softmax or sigmoid.')
                    if type(top_k) is not int or not 1<=top_k<=len(experts):
                        raise ValueError(f'top k must be between 1 and {len(experts)} experts.')
                    if not isinstance(weight,(int,float)) or isinstance(weight,bool) or not math.isfinite(weight) or weight<0:
                        raise ValueError('Balance weight must be a finite non-negative number.')
                    def warn_token_mixing(graph, prefix):
                        for child in graph.get('nodes', []):
                            child_data = child['data']
                            child_key = prefix + child['id']
                            if child_data.get('kind') in ('attention', 'cross_attention', 'sdpa', 'causal_conv', 'ngram', 'engram_hash', 'engram_lookup'):
                                warnings.append({'node': child_key, 'message': 'Expert routing packs selected tokens into one sequence. This operation may mix tokens from different original sequences or lose their positions; use token-wise layers unless this is intended.'})
                            if child_data.get('body'):
                                warn_token_mixing(child_data['body'], child_key + '/')
                    for expert in experts:
                        warn_token_mixing(expert['data']['body'], key + '/' + expert['id'] + '/')
                    nested, count = visit(d['body'], args, key + '/', {**cfg,**p}, True)
                    if len(nested)!=1 or nested[0]!=x:
                        raise ValueError(f'Each expert and the combined output must preserve the input shape and dtype: {x}.')
                    out=nested[0]
                elif kind == 'moe_router':
                    if not moe_body: raise ValueError('A token router belongs inside an MoE container.')
                    dim=x['shape'][-1]; count_experts=p.get('experts')
                    if x['dtype']!='float32' or len(x['shape'])!=3 or type(dim) is not int or type(count_experts) is not int or count_experts<1:
                        raise ValueError('Router expects [B, T, D] float features and a positive expert count.')
                    out={'shape':x['shape'][:-1]+[count_experts],'dtype':'float32'}
                    count=dim*count_experts
                elif kind == 'moe_combine':
                    if not moe_body: raise ValueError('A weighted expert combine belongs inside an MoE container.')
                    if not args[1:] or any(arg != args[1] for arg in args[1:]):
                        raise ValueError('All experts must return matching shapes and dtypes.')
                    out=args[1]
                    if out['dtype']!='float32' or x['dtype']!='float32' or x['shape'][:-1]!=out['shape'][:-1] or x['shape'][-1]!=len(args)-1:
                        raise ValueError('Routing scores and expert outputs must agree on token and expert dimensions.')
                elif kind in ('block', 'loop'):
                    if not d.get('body'):
                        raise ValueError('Container has no internal graph.')
                    error_start = len(errors)
                    nested, count = visit(d['body'], args, key + '/', {**cfg,**p})
                    declared_outputs = sum(child['data'].get('kind') == 'output' for child in d['body']['nodes'])
                    if declared_outputs != 1:
                        raise ValueError('A container must have exactly one output.')
                    if not nested:
                        cause = errors[error_start]['message'] if len(errors) > error_start else 'an internal output shape could not be determined.'
                        raise ValueError(f'Cannot determine this container’s output. {cause} Open the container to inspect its graph.')
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
                        if not isinstance(mask, int) or not 0 <= mask < vocabulary:
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
                    if x['dtype'] != 'float32' or y['dtype'] != 'int64' or x['shape'][:-1] != y['shape'] or x['shape'][-1] != vocabulary:
                        raise ValueError('Selection requires logits [B,T,V] and current token IDs [B,T], with V matching the vocabulary.')
                    if not isinstance(p.get('mask_id', cfg.get('mask_id')), int) or not 0 <= p.get('mask_id', cfg.get('mask_id')) < vocabulary:
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
                        validate_rope_settings(p)
                    elif kind=='sdpa':
                        k,v=args[1:3]
                        if len(s)!=4 or len(k['shape'])!=4 or len(v['shape'])!=4 or s[:2]!=k['shape'][:2] or k['shape'][:3]!=v['shape'][:3] or s[-1]!=k['shape'][-1] or k['dtype']!='float32' or v['dtype']!='float32':raise ValueError('SDPA requires Q[B,H,T,D], K[B,H,S,D], V[B,H,S,Dv].')
                        out['shape'][-1]=v['shape'][-1]
                        if args[3]:
                            try: validate_mask(args[3],s,k['shape'])
                            except ValueError as exc: raise ContractError(str(exc), 'd') from exc
                        if p.get('backend','dense') not in ('dense','flex','auto'):raise ValueError('Attention backend must be dense, flex, or auto.')
                        if not isinstance(p.get('attention_dropout',0),(int,float)) or not 0<=p.get('attention_dropout',0)<=1:raise ValueError('Attention dropout must be between 0 and 1.')
                        if p.get('attention_dropout',0) and p.get('backend','dense')=='flex':raise ValueError('Attention dropout requires the dense or auto backend.')
                    elif kind=='causal_conv':
                        taps=p.get('taps',3)
                        if len(s)!=3 or not isinstance(taps,int) or taps<1:raise ValueError('Causal convolution expects [B,T,D] and a positive tap count.')
                        active=p.get('active_taps',taps)
                        if type(active) is not int or not 1<=active<=taps:raise ValueError('Active convolution taps must be between 1 and the tap count.')
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
                elif kind in ('conv1d','pool1d','conv2d','pool2d'):
                    shape=x['shape']; is_2d=kind.endswith('2d'); pooling=kind.startswith('pool')
                    if x['dtype']!='float32' or len(shape)!=(4 if is_2d else 3):
                        raise ValueError('Expected float features [B,C,H,W] for 2D or [B,T,C] for 1D.')
                    kernel=p.get('kernel_size',2 if pooling else 3); stride=p.get('stride',2 if pooling else 1); padding=0 if pooling else p.get('padding',0)
                    if any(type(v) is not int or v<1 for v in (kernel,stride)) or type(padding) is not int or padding<0:
                        raise ValueError('Kernel and stride must be positive integers; padding must be a non-negative integer.')
                    if pooling and p.get('mode','max') not in ('max','average'):raise ValueError('Pooling mode must be max or average.')
                    channel=shape[1] if is_2d else shape[-1]
                    if type(channel) is not int or channel<1:raise ValueError('Input channel count must be a known positive integer.')
                    output=copy.deepcopy(shape)
                    for axis in ((2,3) if is_2d else (1,)):
                        length=shape[axis]
                        if type(length) is int: output[axis]=(length+2*padding-kernel)//stride+1
                        elif stride==1 and 2*padding==kernel-1: output[axis]=length
                        else: raise ValueError('Set a fixed input spatial or sequence length to infer this kernel, stride and padding.')
                        if type(output[axis]) is int and output[axis]<1:raise ValueError('Kernel is too large for the input.')
                    if not pooling:
                        out_channels=p.get('out_channels',32) if is_2d else p.get('out',channel)
                        if type(out_channels) is not int or out_channels<1:raise ValueError('Output channels must be a positive integer.')
                        output[1 if is_2d else -1]=out_channels
                        count=channel*out_channels*(kernel**(2 if is_2d else 1))+(out_channels if p.get('bias',True) else 0)
                    out={'shape':output,'dtype':'float32'}
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
                            mha = kv == heads
                            candidates = sorted((h for h in range(1, min(dim // 2, 256) + 1) if dim % h == 0 and (dim // h) % 2 == 0 and (mha or h % kv == 0)), key=lambda h: (abs(h - heads), h))[:3]
                            if candidates and mha:
                                examples = f' For example, use {candidates[0]} query heads and {candidates[0]} KV heads for multi-head attention.'
                            elif candidates:
                                examples = f" With {kv} KV heads, examples include {', '.join(map(str, candidates))} query heads."
                            else:
                                examples = ' You may also need to change the KV head count.'
                            raise ValueError(f'Width {dim} cannot be split evenly across {heads} query heads ({dim} ÷ {heads}). Choose a head count that divides the width.{examples} Query heads must also be a multiple of KV heads.')
                        if not isinstance(hd,int) or not isinstance(vd,int) or hd < 2 or hd % 2 or vd < 1:
                            raise ValueError('Q/K head width must be a positive even integer; value width must be positive.')
                        if any(not isinstance(p.get(v,0),int) or p.get(v,0)<0 for v in ('window','conv_taps')):raise ValueError('Attention window and convolution taps must be non-negative integers.')
                        if not isinstance(p.get('qk_norm_eps',1e-6),(int,float)) or not math.isfinite(p.get('qk_norm_eps',1e-6)) or p.get('qk_norm_eps',1e-6)<=0:raise ValueError('Q/K normalization epsilon must be finite and positive.')
                        if p.get('rope_scaling','none')!='none' and not p.get('rope',True):raise ValueError('RoPE scaling requires rotary positions.')
                        if p.get('rope',True):validate_rope_settings(p)
                        if kind == 'cross_attention' and (len(args[1]['shape']) != 3 or args[1]['shape'][0] != x['shape'][0] or args[1]['shape'][-1] != dim or args[1]['dtype'] != 'float32'):
                            raise ContractError(f'Cross-attention memory must be [B,S,D] with batch {x["shape"][0]} and width {dim}; received {args[1]}.', 'b')
                        mask=args[2] if kind=='cross_attention' else args[1]
                        if mask:
                            qshape=[x['shape'][0],heads,x['shape'][1],hd]
                            memory=args[1] if kind=='cross_attention' else x
                            try: validate_mask(mask,qshape,[memory['shape'][0],heads,memory['shape'][1],hd])
                            except ValueError as exc: raise ContractError(str(exc), 'c' if kind == 'cross_attention' else 'b') from exc
                        if p.get('backend','dense') not in ('dense','flex','auto'):raise ValueError('Attention backend must be dense, flex, or auto.')
                        if not isinstance(p.get('attention_dropout',0),(int,float)) or not 0<=p.get('attention_dropout',0)<=1:raise ValueError('Attention dropout must be between 0 and 1.')
                        if p.get('attention_dropout',0) and p.get('backend','dense')=='flex':raise ValueError('Attention dropout requires the dense or auto backend.')
                        count = dim * (heads * hd + kv * hd + kv * vd + heads * vd)
                        if p.get('qkv_bias', False): count += heads*hd+kv*hd+kv*vd
                        if p.get('output_bias', False): count += dim
                        if p.get('qk_norm', False): count += 2 * hd
                        if p.get('gated', False): count += dim * heads * vd
                        count += p.get('conv_taps',0)*(heads*hd+kv*hd+kv*vd)
                    elif kind == 'ffn':
                        hidden = p.get('hidden', dim * 4)
                        if not isinstance(hidden, int) or hidden < 1: raise ValueError('FFN hidden width must be positive.')
                        count = 3 * dim * hidden
                    elif kind == 'layer_norm':
                        if not isinstance(p.get('eps',1e-5),(int,float)) or not math.isfinite(p.get('eps',1e-5)) or p.get('eps',1e-5)<=0:raise ValueError('Normalization epsilon must be finite and positive.')
                        count=dim*(1+int(p.get('bias',True))) if p.get('affine',True) else 0
                    elif kind == 'dropout':
                        probability=p.get('probability',0.1)
                        if not isinstance(probability,(int,float)) or not 0<=probability<=1:raise ValueError('Dropout probability must be between 0 and 1.')
                    elif kind == 'position_embedding':
                        limit=p.get('max_length',2048)
                        if len(x['shape'])!=3 or type(limit) is not int or limit<1:raise ValueError('Position embeddings require [B,T,D] and a positive table length.')
                        if type(x['shape'][1]) is int and x['shape'][1]>limit:raise ValueError('Sequence exceeds the learned position table length.')
                        count=limit*dim
                    elif kind == 'gelu':
                        if p.get('approximate','none') not in ('none','tanh'):raise ValueError('GELU approximation must be none or tanh.')
                    elif kind == 'norm':
                        if not isinstance(p.get('eps',1e-6),(int,float)) or p.get('eps',1e-6)<=0:raise ValueError('Normalization epsilon must be positive.')
                        count = dim
                    elif kind=='scale':
                        factor=p.get('factor',1)
                        if not isinstance(factor,(int,float)) or not math.isfinite(factor):raise ValueError('Scale factor must be a finite number.')
                    elif kind == 'hadamard':
                        if p.get('variant','three_stage') not in ('three_stage','two_stage'):raise ValueError('Hadamard variant must be two_stage or three_stage.')
                        count = (3 if p.get('variant')=='two_stage' else 4) * (1 << (dim - 1).bit_length())
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
                    if share in shared and shared[share][0]['kind'] != kind:
                        raise ValueError(f'Weight group {share} belongs to {shared[share][0]["kind"]} nodes; choose a group for {kind} nodes.')
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
            config={**(config if config is not None else project.get('config',{})), **graph.get('variables', {})}
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
    return {'valid': not errors, 'errors': errors, 'warnings': warnings, 'shapes': shapes, 'counts': counts, 'parameters': total, 'outputs': outputs, 'tokenizers': token_domains}
