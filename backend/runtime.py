"""Portable PyTorch operations shipped with exported graphs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .sparse_runtime import AttentionPattern,as_pattern,attend
except ImportError:
    from sparse_runtime import AttentionPattern,as_pattern,attend


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads, head_dim=None, value_dim=None,
                 causal=True, window=0, rope=True, theta=10000., qk_norm=False, gated=False, conv_taps=0,backend='dense'):
        super().__init__()
        self.h, self.kv = heads, kv_heads
        self.hd, self.vd = head_dim or width // heads, value_dim or width // heads
        self.causal, self.window, self.rope, self.theta = causal, window, rope, theta
        self.backend=backend
        self.q = nn.Linear(width, heads * self.hd, bias=False)
        self.k = nn.Linear(width, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(width, kv_heads * self.vd, bias=False)
        self.o = nn.Linear(heads * self.vd, width, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()
        self.gate = nn.Linear(width, heads * self.vd, bias=False) if gated else None
        self.taps=nn.ParameterList()
        for d in [heads*self.hd,kv_heads*self.hd,kv_heads*self.vd] if conv_taps else []:
            taps=torch.zeros(conv_taps,d);taps[0]=1;self.taps.append(nn.Parameter(taps))

    def rotate(self, x):
        pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        inv = self.theta ** (-torch.arange(0, self.hd, 2, device=x.device).float() / self.hd)
        freq = torch.outer(pos, inv)
        freq = torch.cat((freq, freq), -1).to(x.dtype)
        a, b = x.chunk(2, dim=-1)
        return x * freq.cos() + torch.cat((-b, a), -1) * freq.sin()

    def forward(self, x, memory=None, mask=None, **_):
        memory = x if memory is None else memory
        b, t, _ = x.shape; s = memory.shape[1]
        q,k,v=self.q(x),self.k(memory),self.v(memory)
        if self.taps:
            q,k,v=[sum(taps[j]*shift(z,j) for j in range(taps.shape[0])) for z,taps in zip([q,k,v],self.taps)]
        q = self.q_norm(q.view(b, t, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(k.view(b, s, self.kv, self.hd)).transpose(1, 2)
        v = v.view(b, s, self.kv, self.vd).transpose(1, 2)
        if self.rope: q, k = self.rotate(q), self.rotate(k)
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        y = attend(q,k,v,mask,self.causal,self.window,self.backend)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        if self.gate is not None: y = y * torch.sigmoid(self.gate(x))
        return self.o(y)


class FFN(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x, **_):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class NGram(nn.Module):
    """Independent rolling hashes; incomplete left windows use explicit zero padding."""
    def __init__(self, orders, buckets):
        super().__init__(); self.orders, self.buckets = orders, buckets

    def forward(self, ids, **_):
        results = []
        for order in self.orders:
            h = torch.zeros_like(ids)
            for offset in range(order - 1, -1, -1):
                shifted = F.pad(ids + 1, (offset, 0))[:, :ids.shape[1]]
                h = (h * 257 + shifted) % self.buckets
            results.append(h)
        return torch.stack(results, dim=-1)


class Lookup(nn.Module):
    def __init__(self, orders, buckets, width):
        super().__init__(); self.tables = nn.ModuleList(nn.Embedding(buckets, width) for _ in range(orders))

    def forward(self, indices, **_):
        return sum(table(indices[..., i]) for i, table in enumerate(self.tables))


class Loop(nn.Module):
    def __init__(self, factory, iterations, shared=True):
        super().__init__(); self.iterations, self.shared = iterations, shared
        self.bodies = nn.ModuleList([factory() for _ in range(1 if shared else iterations)])

    def forward(self, state, *invariants, **_):
        for step in range(self.iterations):
            body = self.bodies[0 if self.shared else step]
            state = body(state, *invariants, step=step, steps=self.iterations)
        return state


class Gate(nn.Module):
    def __init__(self):
        super().__init__(); self.logit = nn.Parameter(torch.zeros(()))
    def forward(self, skip, update, **_):
        return skip + torch.sigmoid(self.logit) * update


def shift(x, offset):
    if not offset:return x
    return torch.cat((torch.zeros_like(x[:,:min(offset,x.shape[1])]),x[:,:max(0,x.shape[1]-offset)]),dim=1)


class Routing(nn.Module):
    """Four-lane input-dependent pre/post routing and log-space Sinkhorn residual mixing."""
    def __init__(self,width,lanes,layer):
        super().__init__();self.n=lanes
        self.pre=nn.Parameter(torch.randn(lanes*width,lanes)*.02)
        self.post=nn.Parameter(torch.randn(lanes*width,lanes)*.02)
        self.res=nn.Parameter(torch.randn(lanes*width,lanes*lanes)*.02)
        self.bp=nn.Parameter(torch.zeros(lanes));self.bo=nn.Parameter(torch.zeros(lanes))
        self.br=nn.Parameter(torch.eye(lanes)*4)
        self.a=nn.Parameter(torch.ones(3)*.01)
        lane=F.one_hot(torch.tensor(layer%lanes),lanes).float()
        self.register_buffer('pre_off',8*lane-4);self.register_buffer('post_off',-4*(1-lane))
    def forward(self,x,**_):
        nx=F.rms_norm(x.flatten(-2),[x.shape[-2]*x.shape[-1]],eps=1e-6)
        pre=torch.sigmoid(self.a[0]*(nx@self.pre)+self.bp+self.pre_off)
        post=2*torch.sigmoid(self.a[1]*(nx@self.post)+self.bo+self.post_off)
        res=self.a[2]*(nx@self.res).view(*x.shape[:2],self.n,self.n)+self.br
        for _ in range(20):
            res=res-torch.logsumexp(res,-1,keepdim=True)
            res=res-torch.logsumexp(res,-2,keepdim=True)
        return torch.cat((pre,post,res.exp().flatten(-2)),-1)


class EngramHash(nn.Module):
    def __init__(self,orders,heads,slots):
        super().__init__();self.orders,self.heads,self.slots=orders,heads,slots
    def forward(self,ids,**_):
        result=[]
        for oi,order in enumerate(self.orders):
            for h in range(self.heads):
                acc=torch.full_like(ids,(0x9E3779B9*(oi*self.heads+h+1))&0xffffffff)
                for j in range(order):acc=((acc^shift(ids,j))*16777619)&0xffffffff
                result.append((acc^(acc>>15))%self.slots)
        return torch.stack(result,-1)


class EngramLookup(nn.Module):
    def __init__(self,width,orders,heads,slots,sub_dim,dilation=3):
        super().__init__();self.orders,self.heads,self.dilation=orders,heads,dilation
        self.tables=nn.ModuleList(nn.Embedding(slots,sub_dim) for _ in range(len(orders)*heads))
        self.key=nn.Linear(len(orders)*heads*sub_dim,width,bias=False)
        self.value=nn.Linear(len(orders)*heads*sub_dim,width,bias=False)
        taps=torch.zeros(4,width);taps[0]=1;self.taps=nn.Parameter(taps)
    def forward(self,indices,**_):
        pos=torch.arange(indices.shape[1],device=indices.device)[None,:,None]
        e=torch.cat([table(indices[...,i])*(pos>=self.orders[i//self.heads]-1) for i,table in enumerate(self.tables)],-1)
        k,v=self.key(e),self.value(e)
        v=sum(self.taps[j]*shift(v,j*self.dilation)*(pos>=j*self.dilation) for j in range(4))
        return torch.cat((k,v),-1)


class Monarch(nn.Module):
    """Learned Kronecker factors, fixed seeded permutations and input-conditioned diagonals."""
    def __init__(self,width,split=False):
        super().__init__();import numpy as np
        self.width=width;self.n=1<<(width-1).bit_length()
        ba=1<<((self.n-1).bit_length()//2);bb=self.n//ba
        def walsh(n):
            h=torch.ones(1,1)
            while h.shape[0]<n:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
            return h/math.sqrt(n)
        self.factors=nn.ParameterList([nn.Parameter(walsh(n)) for _ in range(3) for n in [ba,bb]])
        self.d=nn.ParameterList([nn.Parameter(torch.ones(self.n)*(0.02 if i==3 else 1)) for i in range(4)])
        self.bias=nn.Parameter(torch.zeros(self.n))
        self.cv=nn.Parameter(torch.randn(width,8)*.02);self.cu=nn.Parameter(torch.zeros(8,self.n))
        for i,seed in enumerate([11,13]):
            perm=np.concatenate((np.random.RandomState(seed).permutation(self.n//2),self.n//2+np.random.RandomState(seed+977).permutation(self.n//2))) if split else np.random.RandomState(seed).permutation(self.n)
            self.register_buffer('perm'+str(i),torch.tensor(perm,dtype=torch.long))
    def kron(self,z,index):
        a,b=self.factors[2*index:2*index+2]
        return torch.einsum('...ij,ik,jl->...kl',z.reshape(*z.shape[:-1],a.shape[0],b.shape[0]),a,b).flatten(-2)
    def forward(self,x,**_):
        cond=1+(x@self.cv).softmax(-1)@self.cu
        z=F.pad(x,(0,self.n-self.width))
        z=self.kron(self.d[0]*z,0)[...,self.perm0]
        z=self.kron(F.silu(self.d[1]*cond*z+self.bias),1)[...,self.perm1]
        return (self.d[3]*self.kron(self.d[2]*z,2))[...,:self.width]


class Hadamard(nn.Module):
    """Generic fixed Walsh-Hadamard mixer; not Needle's Monarch learned mixer."""
    def __init__(self, width):
        super().__init__(); self.width = width; self.n = 1 << (width - 1).bit_length()
        self.d1 = nn.Parameter(torch.ones(self.n)); self.d2 = nn.Parameter(torch.ones(self.n))
        self.d3 = nn.Parameter(torch.ones(self.n)); self.d4 = nn.Parameter(torch.ones(self.n) * .02)
    def transform(self, x):
        size = 1
        while size < self.n:
            z = x.reshape(*x.shape[:-1], -1, 2, size)
            a, b = z.unbind(-2)
            x = torch.stack((a+b, a-b), -2).flatten(-3)
            size *= 2
        return x / math.sqrt(self.n)
    def forward(self, x, **_):
        z = F.pad(x, (0, self.n-self.width))
        z = self.transform(z*self.d1)*self.d2
        return (self.transform(F.silu(z)*self.d3)*self.d4)[..., :self.width]


class Functional(nn.Module):
    def __init__(self, kind, p, config):
        super().__init__(); self.kind, self.p, self.config = kind, p, config
    def forward(self, x, *args, step=0, steps=1):
        k, p = self.kind, self.p
        if k == 'add': return x + args[0]
        if k == 'multiply': return x * args[0]
        if k == 'subtract': return x - args[0]
        if k == 'concat': return torch.cat((x, args[0]), dim=-1)
        if k == 'silu': return F.silu(x)
        if k == 'sigmoid': return torch.sigmoid(x)
        if k == 'scale': return x * p.get('factor', 1.)
        if k == 'split_heads':
            h=p.get('heads',self.config.get('heads',8))
            return x.reshape(x.shape[0],x.shape[1],h,x.shape[-1]//h).transpose(1,2)
        if k == 'merge_heads':return x.transpose(1,2).contiguous().flatten(-2)
        if k == 'repeat_heads':return x.repeat_interleave(p.get('repeats',1),dim=1)
        if k == 'softmax':return x.softmax(p.get('dim',-1))
        if k == 'matmul':return x@args[0]
        if k == 'transpose':return x.transpose(p.get('dim0',-2),p.get('dim1',-1))
        if k == 'rope':
            d=x.shape[-1];freq=torch.outer(torch.arange(x.shape[-2],device=x.device).float(),p.get('theta',10000.)**(-torch.arange(0,d,2,device=x.device).float()/d))
            freq=torch.cat((freq,freq),-1).to(x.dtype);a,b=x.chunk(2,-1)
            return x*freq.cos()+torch.cat((-b,a),-1)*freq.sin()
        if k == 'sdpa':
            return attend(x,args[0],args[1],args[2] if len(args)>2 else None,p.get('causal',True),p.get('window',0),p.get('backend','dense'))
        if k=='attention_pattern':return AttentionPattern(p.get('pattern','local'),x,args[0] if args and args[0] is not None else x,p)
        if k in ('pattern_union','pattern_intersection'):
            a,b=x,args[0]
            base=a if isinstance(a,AttentionPattern) else b
            if not isinstance(base,AttentionPattern):
                # A pair of custom boolean masks can also be composed directly.
                return a|b if k=='pattern_union' else a&b
            return AttentionPattern('union' if k=='pattern_union' else 'intersection',base.q,base.k,children=[as_pattern(a,base.q,base.k),as_pattern(b,base.q,base.k)])
        if k == 'lane_expand':return x.unsqueeze(-2).expand(*x.shape[:-1],p.get('lanes',4),x.shape[-1])
        if k == 'lane_reduce':return x.mean(-2)
        if k == 'lane_mix':return torch.einsum('btn,btnc->btc',args[0][...,:p.get('lanes',4)],x)
        if k == 'lane_merge':
            n=p.get('lanes',4);coeff=args[1];post=coeff[...,n:2*n];res=coeff[...,2*n:].reshape(*x.shape[:2],n,n)
            return torch.einsum('btij,btjc->btic',res,x)+post.unsqueeze(-1)*args[0].unsqueeze(-2)
        if k == 'memory_fuse':
            k,v=args[0].chunk(2,-1)
            alpha=torch.sigmoid((F.rms_norm(x,[x.shape[-1]],eps=1e-6)*F.rms_norm(k,[k.shape[-1]],eps=1e-6)).sum(-1,keepdim=True)/math.sqrt(x.shape[-1]))
            return x+alpha*v
        if k == 'mask':
            length = p.get('length', 0)
            return torch.cat((x, torch.full((x.shape[0], length), p.get('mask_id', self.config['mask_id']), device=x.device, dtype=x.dtype)), -1)
        if k == 'select':
            ids = args[0]; mask_id = self.config['mask_id']; masked = ids.eq(mask_id)
            logits = x.clone(); logits[..., mask_id] = -torch.inf
            if p.get('temperature', 0) > 0:
                logits = logits / p['temperature'] - torch.empty_like(logits).exponential_().log()
            pred = logits.argmax(-1)
            confidence = logits.softmax(-1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            if p.get('strategy', 'confidence') == 'random': confidence = torch.rand_like(confidence)
            confidence = confidence.masked_fill(~masked, -torch.inf)
            result = ids.clone()
            for b in range(ids.shape[0]):
                # A linear schedule over remaining tokens; all are resolved at the final step.
                remaining = int(masked[b].sum())
                count = math.ceil(remaining / max(1, steps-step))
                if count:
                    positions = confidence[b].topk(count).indices
                    result[b, positions] = pred[b, positions]
            return result
        raise ValueError('Unsupported operation: '+k)


def make_module(kind, p, input_shape, config):
    width = input_shape[-1] if input_shape else config['width']
    if kind == 'embedding': return nn.Embedding(p.get('vocab', config['vocab']), p.get('width', config['width']))
    if kind in ('linear', 'head'): return nn.Linear(width, p.get('out', config['vocab'] if kind == 'head' else config['width']), bias=p.get('bias', False))
    if kind == 'norm': return nn.RMSNorm(width, eps=p.get('eps', 1e-6))
    if kind in ('attention', 'cross_attention'):
        heads = p.get('heads', config.get('heads', 8))
        return Attention(width, heads, p.get('kv_heads', heads), p.get('head_dim'), p.get('value_dim'), p.get('causal', kind != 'cross_attention'), p.get('window', 0), p.get('rope', True), p.get('theta', 10000.), p.get('qk_norm', False), p.get('gated', False),p.get('conv_taps',0),p.get('backend','dense'))
    if kind == 'ffn': return FFN(width, p.get('hidden', width*4))
    if kind == 'ngram': return NGram(p.get('orders', [2,3]), p.get('buckets', 8192))
    if kind == 'lookup': return Lookup(input_shape[-1], p.get('buckets', 8192), p.get('width', config['width']))
    if kind == 'gate': return Gate()
    if kind == 'hadamard': return Hadamard(width)
    if kind == 'monarch':return Monarch(width,p.get('split',False))
    if kind == 'causal_conv':return CausalConv(width,p.get('taps',3))
    if kind == 'routing':return Routing(width,p.get('lanes',4),p.get('layer',0))
    if kind == 'engram_hash':return EngramHash(p.get('orders',[2,3]),p.get('hash_heads',2),p.get('buckets',8192))
    if kind == 'engram_lookup':return EngramLookup(p.get('width',config['width']),p.get('orders',[2,3]),p.get('hash_heads',2),p.get('buckets',8192),p.get('sub_dim',128),p.get('dilation',3))
    return Functional(kind, p, config)


class CausalConv(nn.Module):
    def __init__(self,width,taps):
        super().__init__();v=torch.zeros(taps,width);v[0]=1;self.taps=nn.Parameter(v)
    def forward(self,x,**_):return sum(self.taps[j]*shift(x,j) for j in range(self.taps.shape[0]))
