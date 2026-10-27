"""Static contracts for composable attention patterns; independent of torch."""
PATTERNS=('full','causal','local','global','block','random','content_topk','custom')


def validate_pattern(p):
    kind=p.get('pattern','local')
    if kind not in PATTERNS:raise ValueError('Unknown attention pattern.')
    for key,default in [('window',32),('block_size',16),('top_k',8)]:
        if not isinstance(p.get(key,default),int) or p.get(key,default)<1:raise ValueError(f'{key} must be a positive integer.')
    for key,default in [('neighbor_blocks',0),('global_stride',0),('seed',0)]:
        if not isinstance(p.get(key,default),int) or p.get(key,default)<0:raise ValueError(f'{key} must be a non-negative integer.')
    if not isinstance(p.get('density',.1),(int,float)) or not 0<=p.get('density',.1)<=1:raise ValueError('Random density must be between 0 and 1.')
    if any(not isinstance(i,int) or i<0 for i in p.get('global_tokens',[0])):raise ValueError('Global token positions must be non-negative integers.')
    if kind=='custom':
        matrix=p.get('matrix',[])
        if not matrix or not all(isinstance(row,list) and len(row)==len(matrix[0]) and row for row in matrix):raise ValueError('Custom pattern must be a non-empty rectangular boolean matrix.')
        if any(not isinstance(v,bool) for row in matrix for v in row):raise ValueError('Custom matrix entries must be true or false.')


def validate_mask(mask,q_shape,k_shape):
    if mask['dtype'] not in ('bool','attention_pattern'):raise ValueError('Attention mask must be a boolean tensor or an attention pattern.')
    shape=mask['shape']
    target=[q_shape[0],q_shape[1] if len(q_shape)==4 else 1,q_shape[-2],k_shape[-2]]
    if len(shape) not in (2,4) or shape[-2:]!=target[-2:]:raise ValueError(f'Attention mask {shape} must end in query/key dimensions {target[-2:]}.')
    if len(shape)==4 and any(a!=1 and a!=b for a,b in zip(shape[:2],target[:2])):raise ValueError(f'Mask batch/head dimensions {shape[:2]} cannot broadcast to attention {target[:2]}.')
