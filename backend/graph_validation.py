"""Graph validation and shape propagation."""
import ast
import copy
import math
from .core import (VERSION, PORTS, ContractError, valid_dimension, optional_ports,
    definition_signature, resolve, variable_names, params, mixed_matrix_groups,
    output_port_names, port_names, ordered, moe_structure, validate_rope_settings)
from .lanes import MULTI_OUTPUT_LANES, lane_count, port_label
from .patterns import validate_pattern, validate_mask
from .advice import design_advice
from .tokenizer_bindings import bind_input_tokenizers
from .feature_contracts import feature_contract


def validate_project(project, *, input_contracts=None, moe_body=False, include_advice=True):
    project, token_domains, binding_errors = bind_input_tokenizers(project, ordered)
    errors, warnings, shapes, counts, raw_counts = binding_errors, [], {}, {}, {}
    port_shapes = {}
    cfg = project.get('config', {})
    for field in ('width','vocab','heads','ffn_width'):
        if field in cfg and (not isinstance(cfg[field],int) or isinstance(cfg[field],bool) or cfg[field]<1):
            errors.append({'node':None,'message':f'Model {field} must be a positive integer.'})
    shared = {}
    mixed_groups = mixed_matrix_groups(project.get('graph', {}))
    if project.get('version') != VERSION:
        errors.append({'node': None, 'message': 'Unsupported project version.'})
    tok = project.get('tokenizer')
    if tok and cfg.get('vocab', 0) < tok.get('capacity', 0):
        errors.append({'node': None, 'message': f"Tokenizer needs {tok['capacity']} embedding rows; vocabulary capacity is {cfg.get('vocab')} ."})

    def visit(graph, inputs=None, prefix='', config=None, moe_body=False):
        cfg={**(config if config is not None else project.get('config',{})), **graph.get('variables', {})}
        local = {}
        nodes_by_id = {node['id']: node for node in graph['nodes']}
        local_outputs = {}
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
                    if port not in port_names(n, cfg):
                        fail(n, f'Unknown input port {port}.', port)
                    handle = e.get('sourceHandle') or 'out'
                    source = nodes_by_id.get(e['source'])
                    incoming[port] = local_outputs.get(e['source'], {}).get(handle)
                    if source and handle not in output_port_names(source, cfg):
                        fail(n, f'Connection uses an unknown source port {handle}.', port)
            args = [incoming.get(k) for k in port_names(n, cfg)]
            out = None; count = 0
            try:
                missing = sorted(variable_names(d.get('params', {})) - set(cfg) - {'sqrt_width'})
                if missing:
                    raise ValueError('Unknown variable ' + ', '.join('$' + name for name in missing) + '. Add it in Graph variables or enter a number in this block.')
                if d.get('iteration'):
                    schedule=d['iteration'];index=schedule.get('index');steps=schedule.get('total')
                    if not isinstance(index,int) or not isinstance(steps,int) or not 0<=index<steps:raise ValueError('Application schedule needs 0 ≤ index < total.')
                if kind in ('lane_fork', 'lane_split', 'lane_join'):
                    lane_count(p)
                if kind == 'lane_join':
                    known = [(port, arg) for port, arg in zip(port_names(n, cfg), args) if arg is not None]
                    for port, arg in known[1:]:
                        first = known[0][1]
                        if arg['shape'] != first['shape'] or arg['dtype'] != first['dtype']:
                            raise ContractError(f'Every joined lane must have matching shape and dtype. Expected {first}, received {arg}.', port)
                if kind not in PORTS:
                    raise ValueError(f'Unknown operation: {kind}')
                required=[a for port,a in zip(port_names(n, cfg),args) if port not in optional_ports(n)]
                for port, arg in zip(port_names(n, cfg), args):
                    if arg is None and port not in optional_ports(n):
                        connected = next((e for e in graph.get('edges', []) if e['target'] == n['id'] and (e.get('targetHandle') or 'a') == port), None)
                        message = (f"Input {port_label(port)} is connected to {connected['source']}, whose shape is unresolved. Fix the upstream error first."
                                   if connected else f'Input {port_label(port)} is unconnected. Connect a tensor to this required port.')
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
                elif kind in ('lane_fork', 'lane_split', 'lane_join'):
                    lanes = lane_count(p)
                    if kind == 'lane_fork':
                        if not x['shape'] or x['dtype'] == 'attention_pattern':
                            raise ValueError('Fork lanes expects a tensor with a feature dimension.')
                        out = copy.deepcopy(x)
                    elif kind == 'lane_split':
                        if len(x['shape']) < 2 or x['shape'][-2] != lanes or x['dtype'] == 'attention_pattern':
                            raise ValueError(f'Split lanes expects a tensor with {lanes} lanes in its second-to-last dimension.')
                        out = {'shape': x['shape'][:-2] + x['shape'][-1:], 'dtype': x['dtype']}
                    else:
                        mode = p.get('mode', 'stack')
                        if mode not in ('stack', 'mean', 'sum'):
                            raise ValueError('Join mode must be stack, mean or sum.')
                        if x['dtype'] == 'attention_pattern' or (mode != 'stack' and x['dtype'] != 'float32'):
                            raise ValueError('Mean and sum joins require float features; stack also supports int64 and bool tensors.')
                        shape = x['shape'][:-1] + [lanes, x['shape'][-1]] if mode == 'stack' else list(x['shape'])
                        out = {'shape': shape, 'dtype': x['dtype']}
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
                    out, count, dim, outdim = feature_contract(kind, x, args, p, cfg, width, tok, graph)
                if out and (not isinstance(out.get('shape'), list) or not out['shape'] or any(not valid_dimension(v) for v in out['shape'])):
                    raise ValueError(f"Invalid or unresolved dimensions {out.get('shape')}. Use positive integers or symbols such as B, T, S; check $parameter bindings.")
                if out and out['dtype'] not in ('float32','int64','bool','attention_pattern'):raise ValueError('Unsupported output dtype.')
                share = d.get('share')
                raw_counts[key]=count
                if share and out is not None:
                    if share in mixed_groups and p.get('tie'):
                        raise ValueError('Use either this weight group or the legacy embedding tie on an output head.')
                    config_key={k:cfg.get(k) for k in ('width','heads','ffn_width','vocab','mask_id')}
                    feature_dim=None if kind=='embedding' else (x['shape'][-1] if x else None)
                    signature = (definition_signature(n,cfg), config_key, feature_dim)
                    group = shared.setdefault(share, {})
                    if kind not in group and group and share not in mixed_groups:
                        previous = next(iter(group))
                        raise ValueError(f'Weight group {share} belongs to {previous} nodes; choose a group for {kind} nodes.')
                    if kind in group and group[kind] != signature:
                        raise ValueError('Shared module instances must have identical operation, parameters, and output shape.')
                    if share in mixed_groups:
                        matrix = (capacity, width) if kind == 'embedding' else (outdim, dim)
                        if '_matrix' in group and group['_matrix'] != matrix:
                            raise ValueError(f'Weight group {share} needs matching embedding and head matrix dimensions.')
                        group['_matrix'] = matrix
                    if kind in group: count = 0
                    elif share in mixed_groups and len(group) > 1:
                        count = outdim if kind == 'head' and p.get('bias', False) else 0
                    group[kind] = signature
            except (ValueError, TypeError, KeyError, IndexError, SyntaxError) as exc:
                fail(n, str(exc), getattr(exc, 'port', None)); out = None
            local[n['id']] = out
            local_outputs[n['id']] = {handle: out for handle in output_port_names(n, cfg)}
            if kind in MULTI_OUTPUT_LANES:
                port_shapes[key] = local_outputs[n['id']]
            shapes[key] = out; counts[key] = count; total += count
        if not any(n['data']['kind']=='output' for n in graph['nodes']):
            errors.append({'node': prefix.rstrip('/') or None, 'message': 'Graph needs an Output node.'})
        indices = sorted(int(params(n, cfg).get('index', 0)) for n in graph['nodes'] if n['data']['kind'] == 'input')
        if indices != list(range(len(indices))):
            errors.append({'node': prefix.rstrip('/') or None, 'message': 'Input indices must be unique and consecutive from zero.'})

        return [s for _, s in sorted(outputs, key=lambda item: item[0])], total

    try:
        outputs, total = visit(project.get('graph', {}), inputs=input_contracts, moe_body=moe_body)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append({'node': None, 'message': f'Malformed graph: {exc}'})
        outputs, total = [], 0
    if not errors:
        allocated=set()
        allocated_matrices=set()
        def allocation(graph,prefix='',multiple=1,config=None):
            config={**(config if config is not None else project.get('config',{})), **graph.get('variables', {})}
            result=0
            for n in ordered(graph):
                d=n['data'];key=prefix+n['id'];share=d.get('share');p=params(n,config)
                slot=(share,d['kind']) if share in mixed_groups else share
                if share and slot in allocated:
                    count=0
                else:
                    if share:allocated.add(slot)
                    factor=1 if share else multiple
                    if d.get('body'):
                        repetitions=p.get('iterations',1) if d['kind']=='loop' and not p.get('shared',True) else 1
                        count=allocation(d['body'],key+'/',factor*repetitions,{**config,**p})
                    else:
                        count=raw_counts.get(key,0)*factor
                        if share in mixed_groups and d['kind'] in ('embedding','head'):
                            matrix_size=raw_counts.get(key,0)-(p.get('out',config.get('vocab')) if d['kind']=='head' and p.get('bias',False) else 0)
                            if share in allocated_matrices:count-=matrix_size
                            else:allocated_matrices.add(share)
                counts[key]=count;result+=count
            return result
        total=allocation(project['graph'])
    # Advice is independent of export validity, and only inspects resolved nodes.
    # Skip structurally malformed graphs rather than hiding the original error.
    if include_advice and not any(e.get('node') is None for e in errors):
        warnings.extend(design_advice(project, shapes, counts, resolve))
    return {'valid': not errors, 'errors': errors, 'warnings': warnings, 'shapes': shapes, 'port_shapes': port_shapes, 'counts': counts, 'parameters': total, 'outputs': outputs, 'tokenizers': token_domains}
