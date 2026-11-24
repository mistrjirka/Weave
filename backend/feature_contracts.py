"""Contracts for floating-point feature operations."""
import copy
import math
from .core import ContractError, params, validate_rope_settings
from .patterns import validate_mask


def feature_contract(kind, x, args, p, cfg, width, tok, graph):
    count = 0
    dim = outdim = None
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
            if kind != 'head':
                raise ValueError('Only an output head can tie to an embedding table.')
            target = next((v for v in graph['nodes'] if v['id'] == p['tie']), None)
            if not target or target['data']['kind'] != 'embedding':
                raise ValueError('Tied output must reference an embedding in this graph.')
            tp = params(target, cfg)
            if tp.get('width', cfg.get('width')) != dim or tp.get('vocab', cfg.get('vocab')) != outdim:
                raise ValueError('Tied output dimensions must match the embedding table.')
            count = outdim if p.get('bias', False) else 0
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
    return out, count, dim, outdim
