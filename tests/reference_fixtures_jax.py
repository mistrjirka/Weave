"""Regenerate the small source-model fixtures using JAX 0.6.2 and Flax 0.10.7.

Run in a separate Python 3.11 environment; these libraries are not app dependencies.
The Needle sources are pinned in references/ and use their normal unquantized path.
"""
import importlib.util
from pathlib import Path
import sys
import types
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
package=types.ModuleType('upstream');package.__path__=[];sys.modules['upstream']=package
q=types.ModuleType('upstream.quantize');q.fake_quant_act=lambda x:x;q.head_weight=lambda x,*a,**k:x;q.maybe_quant_kv=lambda x,*a:x;q.maybe_quant_query=lambda x,*a:x;sys.modules['upstream.quantize']=q
ids=jnp.array([[1,2,3,4,5]],dtype=jnp.int32)
rng=np.random.default_rng(17)
for version in (1,2,3):
    spec=importlib.util.spec_from_file_location(f'upstream.needle{version}',ROOT/'references'/f'needle{version}_architecture.py')
    source=importlib.util.module_from_spec(spec);sys.modules[spec.name]=source;spec.loader.exec_module(source)
    if version==1:
        cfg=source.TransformerConfig(vocab_size=32,d_model=16,num_heads=4,num_kv_heads=2,num_encoder_layers=2,num_decoder_layers=2,dtype='float32')
        model=source.SimpleAttentionNetwork(cfg);mask=source.make_causal_mask(ids.shape[1]);args=(ids,ids);kwargs={'tgt_mask':mask}
    else:
        options=dict(vocab_size=32,d_model=16,num_heads=4,num_kv_heads=2,num_layers=2,engram_layers=(0,),engram_slots=32,engram_orders=(2,3),dtype='float32',flash=False,rope_theta=100000.)
        if version==3:options.update(engram_heads=2,qk_head_dim=4,v_head_dim=6,qkv_conv_taps=3,global_layers=(1,),sliding_window=2,remat=False)
        model=source.SimpleAttentionNetwork(source.TransformerConfig(**options));args=(ids,);kwargs={}
    initial=model.init(jax.random.PRNGKey(0),*args,**kwargs)
    paths,structure=jax.tree_util.tree_flatten_with_path(initial)
    values=[];flat={}
    for path,value in paths:
        name='/'.join(str(getattr(k,'key',k)) for k in path)
        # Vary every learned weight, including normally zero-initialized gates,
        # to exercise computations that an initialization-only check would miss.
        array=rng.normal(0,0.05,value.shape).astype(np.float32)
        values.append(jnp.asarray(array));flat[name]=array
    params=jax.tree_util.tree_unflatten(structure,values)
    flat['logits']=np.asarray(model.apply(params,*args,**kwargs))
    out=ROOT/'tests'/'fixtures'/'reference';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'needle{version}_small.npz',**flat)
    print(f'needle{version}: {flat["logits"].shape}',flush=True)
