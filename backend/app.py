from __future__ import annotations
import io
import json
import logging
import shutil
import os
from pathlib import Path
import re
import sqlite3
import uuid
import zipfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from .core import validate_project, params, graph_context
from .attention import expand_attention
from .operations import unroll_loop
from .compiler import compile_project, export_zip
from .templates import templates
from .tokenizers import TokenizerService, ALLOWED_SUFFIXES
from .tokenizer_bindings import input_tokenizers

ROOT=Path(__file__).parent.parent
DATA=Path(os.environ.get('WEAVE_DATA', ROOT/'data'));DATA.mkdir(parents=True,exist_ok=True)
@asynccontextmanager
async def app_lifespan(app):
    async with mcp_server.session_manager.run():
        yield


app=FastAPI(title='Weave Neural Graph Studio',version='0.1.0',lifespan=app_lifespan)

@app.exception_handler(Exception)
async def internal_error(request, exc):
    error_id = uuid.uuid4().hex[:12]
    logging.getLogger("uvicorn.error").error(
        "Request %s failed (%s %s)", error_id, request.method, request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error (reference {error_id}). Check the server logs."},
    )

@app.middleware("http")
async def revalidate_frontend(request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache"
    return response

tokenizers=TokenizerService(DATA/'tokenizers')


def db():
    conn=sqlite3.connect(DATA/'studio.db')
    conn.execute('CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, document TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP)')
    return conn


@app.get('/api/health')
def health():return {'status':'ok','version':1}


@app.get('/api/templates')
def get_templates():return templates()


@app.get('/api/projects')
def list_projects():
    with db() as conn:return [{'id':r[0],'name':r[1],'updated':r[2]} for r in conn.execute('SELECT id,name,updated FROM projects ORDER BY updated DESC')]


@app.get('/api/projects/{project_id}')
def get_project(project_id:str):
    with db() as conn:row=conn.execute('SELECT document FROM projects WHERE id=?',(project_id,)).fetchone()
    if not row:raise HTTPException(404,'Project not found')
    return json.loads(row[0])


@app.put('/api/projects/{project_id}')
def save_project(project_id:str,project:dict):
    if project.get('version') != 1:raise HTTPException(422,'Unsupported project version')
    if not isinstance(project.get('graph',{}).get('nodes'),list):raise HTTPException(422,'Project needs graph.nodes')
    from .training.service import valid_id
    from .training.weave_training import atomic_json
    try:valid_id(project_id)
    except ValueError as exc:raise HTTPException(422,str(exc))
    project['id']=project_id
    from .training.workspaces import WorkspaceStore, WorkspaceConflict
    store=WorkspaceStore(DATA/'training')
    with store.lock:
        try:atomic_json(store.project_directory(project_id, project.get('name'))/'project.json', project)
        except WorkspaceConflict as exc:raise HTTPException(409,str(exc))
        with db() as conn:conn.execute('INSERT INTO projects(id,name,document) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document,updated=CURRENT_TIMESTAMP',(project_id,project.get('name','Untitled'),json.dumps(project)))
        stage_sync={'updated':0,'error':None}
        try:stage_sync['updated']=store.sync_project_models(project, DATA/'tokenizers')
        except (ValueError, OSError) as exc:stage_sync['error']=str(exc)
    return {'id':project_id,'saved':True,'stage_sync':stage_sync}


@app.delete('/api/projects/{project_id}')
def delete_project(project_id:str):
    with db() as conn:conn.execute('DELETE FROM projects WHERE id=?',(project_id,))
    return {'deleted':True}


@app.post('/api/validate')
def validate(project:dict):return validate_project(project)

class ConnectionOptionsRequest(BaseModel):
    project: dict
    path: list[str] = Field(default_factory=list, max_length=32)
    source: str
    handle: str = "out"
    candidates: list[dict] = Field(max_length=512)


@app.post('/api/connections/compatible')
def compatible_connections(request: ConnectionOptionsRequest):
    from .connections import connection_options
    try:
        return connection_options(request.project, request.path, request.source, request.handle, request.candidates)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise HTTPException(422, str(exc))



@app.post('/api/attention/expand')
def expand(request:dict):
    project=request['project']
    try:graph,config=graph_context(project,request.get('path',[]))
    except (KeyError,ValueError) as exc:raise HTTPException(422,str(exc))
    n=next(n for n in graph['nodes'] if n['id']==request['node_id'])
    if n['data']['kind'] not in ('attention','cross_attention'):raise HTTPException(422,'Select an attention node')
    report=validate_project(project)
    incoming=next((e for e in graph['edges'] if e['target']==n['id'] and (e.get('targetHandle') or 'a')=='a'),None)
    prefix='/'.join(request.get('path',[]));prefix=prefix+'/' if prefix else ''
    shape=report['shapes'].get(prefix+incoming['source']) if incoming else None
    width=shape['shape'][-1] if shape else config.get('width',128)
    try:return expand_attention(params(n,config),width,n['data']['kind']=='cross_attention')
    except (ValueError,ZeroDivisionError,TypeError) as exc:raise HTTPException(422,str(exc))


@app.post('/api/graph/unroll')
def unroll(request:dict):
    project=request['project']
    try:
        graph,config=graph_context(project,request.get('path',[]))
        return unroll_loop(graph,request['node_id'],config)
    except (KeyError,ValueError,StopIteration) as exc:raise HTTPException(422,str(exc))


@app.post('/api/code')
def code(project:dict):
    report=validate_project(project)
    if not report['valid']:return {'report':report,'source':'# Resolve graph errors to generate runnable PyTorch.\n'+ '\n'.join('# '+e['message'] for e in report['errors'])}
    try:return {'report':report,'source':compile_project(project)['model.py']}
    except Exception as exc:raise HTTPException(422,str(exc))


@app.post('/api/export')
def export(project:dict):
    tok=project.get('tokenizer') or {};token_id=tok.get('id','')
    directory=DATA/'tokenizers'/token_id if re.fullmatch('[a-f0-9]{20}',token_id) else None
    directories={info["id"]: DATA/"tokenizers"/info["id"] for info in input_tokenizers(project).values() if re.fullmatch(r"[a-f0-9]{20}",info.get("id", ""))}
    try:result=export_zip(project,directory,directories)
    except ValueError as exc:raise HTTPException(422,exc.args[0])
    name=re.sub(r'[^a-zA-Z0-9_-]','_',project.get('name','model'))
    return Response(result,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{name}.zip"'})


class TokenizerRequest(BaseModel):
    source:str=Field(min_length=1,max_length=4096)
    revision:str|None=None


@app.get('/api/tokenizers/projects')
def project_tokenizers():
    """Reuse saved input bindings without downloading or re-inspecting tokenizers."""
    result = []
    with db() as conn:
        for project_id, name, document in conn.execute(
            'SELECT id,name,document FROM projects ORDER BY updated DESC, id'
        ):
            project = json.loads(document)
            bindings = input_tokenizers(project)
            # Older projects stored their tokenizer on the project itself.
            if project.get('tokenizer'):
                bindings.setdefault('', project['tokenizer'])
            seen = set()
            for path, info in bindings.items():
                if info['id'] in seen:
                    continue
                seen.add(info['id'])
                result.append({
                    'project_id': project_id, 'project_name': name,
                    'input_path': path, 'tokenizer': info,
                })
    return result


@app.post('/api/tokenizers/inspect')
def inspect_tokenizer(request:TokenizerRequest):
    try:return tokenizers.load(request.source,request.revision)
    except Exception as exc:raise HTTPException(422,str(exc))


@app.post('/api/tokenizers/upload')
async def upload_tokenizer(files:list[UploadFile]=File(...)):
    target=DATA/'uploads'/uuid.uuid4().hex;target.mkdir(parents=True)
    try:
        total=0
        for f in files:
            content=await f.read(64*1024*1024+1);total+=len(content)
            if total>64*1024*1024:raise HTTPException(413,'Tokenizer bundle exceeds 64 MB')
            name=Path(f.filename or '').name
            if Path(name).suffix not in ALLOWED_SUFFIXES:raise HTTPException(422,'Upload tokenizer JSON, model, vocabulary, or merges files.')
            if (target/name).exists():raise HTTPException(422,'Multiple files have the same name. Upload files for one tokenizer only.')
            (target/name).write_bytes(content)
        source=target
        names=sorted(target.iterdir(),key=lambda path:(path.name!='tokenizer.json',path.name))
        if not names:raise HTTPException(422,'Choose at least one tokenizer file.')
        if len(names)==1 and names[0].suffix in ('.json','.model') and names[0].name not in ('tokenizer_config.json','config.json'):source=names[0]
        label=names[0].name if len(names)==1 else f"{names[0].name} + {len(names)-1} {'file' if len(names)==2 else 'files'}"
        try:return await run_in_threadpool(tokenizers.load, source, display_name=label)
        except Exception as exc:raise HTTPException(422,str(exc))
    finally:
        shutil.rmtree(target, ignore_errors=True)


class PreviewRequest(BaseModel):
    id:str=Field(pattern=r'^[a-f0-9]{20}$')
    text:str=Field(max_length=10000)


@app.post('/api/tokenizers/preview')
def tokenizer_preview(request:PreviewRequest):
    try:return tokenizers.preview(request.id,request.text)
    except Exception as exc:raise HTTPException(422,str(exc))


from .training.api import router as training_router
app.include_router(training_router(DATA))

from .mcp_bridge import WeaveBridge, create_mcp_server, mcp_transport_security
mcp_server = create_mcp_server(WeaveBridge(app=app))
app.mount('/mcp', mcp_server.streamable_http_app(streamable_http_path='/', stateless_http=True, transport_security=mcp_transport_security()), name='mcp')


if (ROOT/'dist').is_dir():app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='frontend')
