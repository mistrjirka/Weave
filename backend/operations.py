"""Graph rewrites preserve computation while exposing editable applications."""
from copy import deepcopy
import uuid
from .core import params


def unroll_loop(graph,node_id,config):
    g=deepcopy(graph);loop=next(n for n in g['nodes'] if n['id']==node_id);d=loop['data'];p=params(loop,config)
    if d['kind']!='loop':raise ValueError('Select a loop container to expand.')
    if d.get('share'):raise ValueError('Make the outer loop independent before expanding its applications.')
    count=p.get('iterations',1)
    if not isinstance(count,int) or not 1<=count<=64:raise ValueError('Separate applications support 1–64 iterations. Larger loops can stay compact.')
    prefix='apply_'+uuid.uuid4().hex[:8];group=prefix+'_weights'
    incoming=[e for e in g['edges'] if e['target']==node_id];outgoing=[e for e in g['edges'] if e['source']==node_id]
    if not any((e.get('targetHandle') or 'a')=='a' for e in incoming):raise ValueError('Connect the loop state before expanding it.')
    nodes=[];edges=[]
    for i in range(count):
        id=f'{prefix}_{i}'
        data={**deepcopy(d),'kind':'block','label':f'{d["label"]} · {i+1}',
              'params':deepcopy(d.get('params',{})),
              'iteration':{'index':i,'total':count}}
        if p.get('shared',True):data['share']=group
        nodes.append({'id':id,'type':'architecture','position':{'x':loop['position']['x']+i*380,'y':loop['position']['y']},'data':data})
        for e in incoming:
            port=e.get('targetHandle') or 'a'
            if port=='a' and i:continue
            edges.append({**deepcopy(e),'id':uuid.uuid4().hex,'target':id})
        if i:edges.append({'id':uuid.uuid4().hex,'source':nodes[i-1]['id'],'target':id,'sourceHandle':'out','targetHandle':'a','type':'smoothstep'})
    edges.extend({**deepcopy(e),'id':uuid.uuid4().hex,'source':nodes[-1]['id']} for e in outgoing)
    g['nodes']=[n for n in g['nodes'] if n['id']!=node_id]+nodes
    g['edges']=[e for e in g['edges'] if e['target']!=node_id and e['source']!=node_id]+edges
    return {'graph':g,'first_id':nodes[0]['id'],'group':group if p.get('shared',True) else None}
