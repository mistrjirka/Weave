"""Graph contracts and shape analysis. This module never imports user code."""
from __future__ import annotations

import ast
import copy
import re
import math
from .lanes import MULTI_OUTPUT_LANES, lane_count, lane_ports, port_label
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
    'lane_fork':['a'], 'lane_split':['a'], 'lane_join':[],
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


def mixed_matrix_groups(graph):
    kinds = {}
    def visit(body):
        for n in body.get('nodes', []):
            data = n.get('data', {})
            group = data.get('share')
            if group:
                kinds.setdefault(group, set()).add(data.get('kind'))
            if isinstance(data.get('body'), dict):
                visit(data['body'])
    visit(graph)
    return {group for group, members in kinds.items() if members == {'embedding', 'head'}}



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


def output_port_names(node, config=None):
    d = node['data']
    if d['kind'] in MULTI_OUTPUT_LANES:
        return lane_ports(params(node, config or {}))
    return [] if d['kind'] == 'output' else ['out']


def port_names(node, config=None):
    d = node['data']
    if d['kind'] == 'lane_join':
        return lane_ports(params(node, config or {}))
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

def validate_project(project, *, input_contracts=None, moe_body=False, include_advice=True):
    from .graph_validation import validate_project as validate_graph
    return validate_graph(project, input_contracts=input_contracts, moe_body=moe_body, include_advice=include_advice)
