"""Composable predicates: dense SDPA reference path and explicit compiled FlexAttention."""
import math
import torch
from torch.nn import functional as F


class AttentionPattern:
    def __init__(self,kind,q,k=None,p=None,children=None,tensor=None):
        self.kind,self.q,self.k,self.p=kind,q,q if k is None else k,p or {}
        self.children=children or []
        self.tensor=tensor
        self.shape=tuple(tensor.shape) if tensor is not None else (self.q.shape[-2],self.k.shape[-2])
        self.device=self.q.device
        if kind=='content_topk':
            qh=q if q.ndim==4 else q.unsqueeze(1);kh=self.k if self.k.ndim==4 else self.k.unsqueeze(1)
            scores=(qh.detach()@kh.detach().transpose(-1,-2))/math.sqrt(qh.shape[-1])
            if self.p.get('causal',False):
                allowed=torch.arange(scores.shape[-1],device=q.device)[None,:]<=torch.arange(scores.shape[-2],device=q.device)[:,None]
                scores=scores.masked_fill(~allowed,-torch.inf)
            indices=scores.topk(min(self.p.get('top_k',8),scores.shape[-1]),dim=-1).indices
            self.tensor=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,indices,True)
            if self.p.get('causal',False):self.tensor &= allowed
            self.shape=tuple(self.tensor.shape)
        elif kind=='custom':
            self.tensor=torch.tensor(self.p['matrix'],device=q.device,dtype=torch.bool)
            if tuple(self.tensor.shape)!=self.shape:raise ValueError('Custom attention matrix must match query/key sequence lengths.')
        elif children and any(len(c.shape)==4 for c in children):self.shape=next(c.shape for c in children if len(c.shape)==4)

    def predicate(self,b,h,q,k):
        p=self.p;kind=self.kind
        if self.tensor is not None:
            t=self.tensor
            qi=q.clamp(0,t.shape[-2]-1);ki=k.clamp(0,t.shape[-1]-1)
            result=t[qi,ki] if t.ndim==2 else t[b%t.shape[0],h%t.shape[1],qi,ki]
            return result&(q<t.shape[-2])&(k<t.shape[-1])
        if kind=='union':return self.children[0].predicate(b,h,q,k)|self.children[1].predicate(b,h,q,k)
        if kind=='intersection':return self.children[0].predicate(b,h,q,k)&self.children[1].predicate(b,h,q,k)
        if kind=='full':result=torch.ones_like(q+k,dtype=torch.bool)
        elif kind=='causal':result=k<=q
        elif kind=='local':result=(q-k).abs()<p.get('window',32)
        elif kind=='global':
            result=torch.zeros_like(q+k,dtype=torch.bool)
            for index in p.get('global_tokens',[0]):result=result|(q==index)|(k==index)
            if p.get('global_stride',0):result=result|(q%p['global_stride']==0)|(k%p['global_stride']==0)
        elif kind=='block':result=(torch.div(q,p.get('block_size',16),rounding_mode='floor')-torch.div(k,p.get('block_size',16),rounding_mode='floor')).abs()<=p.get('neighbor_blocks',0)
        elif kind=='random':
            hashed=(q*1103515245+k*12345+p.get('seed',0)*2654435761)&0xffffffff
            hashed=((hashed^(hashed>>16))*0x45d9f3b)&0xffffffff
            hashed=((hashed^(hashed>>16))*0x45d9f3b)&0xffffffff
            hashed=(hashed^(hashed>>16))&0x7fffffff
            result=(hashed<int(p.get('density',.1)*2147483648))|(q==k)
        else:raise ValueError('Unsupported pattern: '+kind)
        if p.get('causal',False):result=result&(k<=q)
        return result

    def dense(self,batch=1,heads=1):
        q=torch.arange(self.q.shape[-2],device=self.device)
        k=torch.arange(self.k.shape[-2],device=self.device)
        if len(self.shape)==4:
            return self.predicate(torch.arange(batch,device=self.device)[:,None,None,None],torch.arange(heads,device=self.device)[None,:,None,None],q[None,None,:,None],k[None,None,None,:])
        return self.predicate(torch.tensor(0,device=self.device),torch.tensor(0,device=self.device),q[:,None],k[None,:])


def as_pattern(value,q,k):
    if isinstance(value,AttentionPattern):return value
    return AttentionPattern('tensor',q,k,tensor=value)


_compiled_flex=None


def attend(q,k,v,mask=None,causal=False,window=0,backend='dense',dropout_p=0.0):
    pattern=as_pattern(mask,q,k) if mask is not None else AttentionPattern('full',q,k)
    if window:
        pattern=AttentionPattern('intersection',q,k,children=[pattern,AttentionPattern('local',q,k,{'window':window})])
    if causal:pattern=AttentionPattern('intersection',q,k,children=[pattern,AttentionPattern('causal',q,k)])
    if dropout_p and backend=='flex':raise ValueError('FlexAttention does not support attention dropout; use dense or auto.')
    use_flex=not dropout_p and (backend=='flex' or backend=='auto' and q.is_cuda)
    if use_flex:
        if backend=='flex' and not q.is_cuda:raise RuntimeError('Compiled FlexAttention requires a compatible GPU. Select dense for CPU execution.')
        from torch.nn.attention.flex_attention import create_block_mask,flex_attention
        block_mask=create_block_mask(pattern.predicate,q.shape[0] if len(pattern.shape)==4 else None,q.shape[1] if len(pattern.shape)==4 else None,q.shape[-2],k.shape[-2],device=str(q.device))
        global _compiled_flex
        if _compiled_flex is None:_compiled_flex=torch.compile(flex_attention)
        return _compiled_flex(q,k,v,block_mask=block_mask)
    if mask is None and not window:
        return F.scaled_dot_product_attention(q,k,v,is_causal=causal,dropout_p=dropout_p)
    return F.scaled_dot_product_attention(q,k,v,attn_mask=pattern.dense(q.shape[0],q.shape[1]),is_causal=False,dropout_p=dropout_p)
