"""FORGE Studio: local FastAPI job bridge for the independent PS1–PS4 modules.

Run: uvicorn backend.app:app --host 127.0.0.1 --port 8000
The API is deliberately local-only, unauthenticated, and not safe for public hosting.
"""
import importlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('FORGE_PRIVATE_DIR', ROOT / '.local_data')).resolve()
ALLOWED_EXT = {'.mp4', '.mov', '.webm', '.jpg', '.jpeg', '.png'}
MAX_BYTES = 800 * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.RLock()

app = FastAPI(title='FORGE Studio local API', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'], allow_credentials=False, allow_methods=['GET','POST','PATCH','DELETE'], allow_headers=['Content-Type'])


def public_job(job: dict) -> dict:
    return {key: value for key,value in job.items() if key not in {'path','directory'}}


def get_job(job_id: str) -> dict:
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, 'Job not found')
        return JOBS[job_id]


def put(job_id: str, **kwargs):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def process_job(job_id: str):
    """Call teammate-supplied, contract-compatible functions when present.

    Missing modules are reported, never replaced by invented predictions.
    One module failure does not delete another module's successful output.
    """
    job = get_job(job_id)
    if job['media_type'] != 'video':
        put(job_id, status='partial', note='Still-image inference adapter not yet installed')
        return
    input_path, output_dir = job['path'], job['directory'] / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    issues: list[str] = []
    stages = [
        ('ps1','modules.ps1','run_ps1', lambda: (input_path, output_dir)),
        ('ps2','modules.ps2','run_ps2', lambda: (input_path, Path(outputs['ps1']), output_dir)),
        ('ps3','modules.ps3','run_ps3', lambda: (input_path, Path(outputs['ps1']), Path(outputs['ps2']), output_dir)),
        ('ps4','modules.ps4','run_ps4', lambda: (Path(outputs['ps1']), output_dir)),
    ]
    put(job_id, status='processing')
    for name, module_name, func_name, args_fn in stages:
        dependencies = {'ps1': [],'ps2':['ps1'],'ps3':['ps1','ps2'],'ps4':['ps1']}[name]
        if any(dep not in outputs for dep in dependencies):
            issues.append(f'{name}: waiting for prerequisite module')
            continue
        try:
            module = importlib.import_module(module_name)
            run = getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            issues.append(f'{name}: module unavailable ({type(exc).__name__})')
            continue
        try:
            result = Path(run(*args_fn())).resolve()
            # Never accept a teammate module that publishes output outside the private job folder.
            if not result.is_relative_to(output_dir.resolve()) or not result.is_file():
                raise ValueError('Module must return an existing private output file')
            outputs[name] = str(result)
            put(job_id, completed_modules=sorted(outputs))
        except Exception as exc:
            issues.append(f'{name}: {type(exc).__name__}: {str(exc)[:160]}')
    put(job_id, status='completed' if len(outputs)==4 else 'partial', completed_modules=sorted(outputs), issues=issues)


@app.get('/api/health')
def health():
    return {'ok':True, 'mode':'local', 'ps1_ps3_loaded': all((ROOT/'modules'/f'ps{i}.py').is_file() for i in (1,2,3))}


@app.post('/api/jobs', status_code=201)
async def submit_job(background_tasks: BackgroundTasks, file: UploadFile = File(...), client_clip_id: str = Form('')):
    name = Path(file.filename or '').name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, 'Unsupported file extension')
    if not name or name.startswith('.'):
        raise HTTPException(422, 'Invalid filename')
    job_id=uuid.uuid4().hex
    directory = DATA_DIR / job_id
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / f'input{ext}'
    n = 0
    try:
        with path.open('wb') as out:
            while chunk := await file.read(1024*1024):
                n += len(chunk)
                if n > MAX_BYTES:
                    raise HTTPException(413,'Maximum file size is 800 MB')
                out.write(chunk)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await file.close()
    job = {'job_id':job_id, 'clip_id':job_id,'client_clip_id':client_clip_id,'name':name,'size':n,
           'media_type':'image' if ext in {'.jpg','.jpeg','.png'} else 'video',
           'status':'queued','completed_modules':[],'issues':[], 'path':path,'directory':directory}
    with LOCK:
        JOBS[job_id]=job
    background_tasks.add_task(process_job, job_id)
    return public_job(job)


@app.get('/api/jobs/{job_id}')
def get_status(job_id: str):
    return public_job(get_job(job_id))


@app.get('/api/clips')
def get_clips():
    with LOCK:
        return [public_job(job) for job in JOBS.values()]


@app.get('/api/clips/{job_id}')
def get_clip(job_id: str):
    return public_job(get_job(job_id))


@app.get('/api/clips/{job_id}/media')
def get_media(job_id: str):
    job=get_job(job_id)
    return FileResponse(job['path'], filename=job['name'])


ROUTE_FILES={'hands':('ps1','hands.jsonl'),'contacts':('ps2','ps2.json'), 'actions':('ps3','actions.json'), 'pose3d':('ps4','ps4.json')}


@app.get('/api/clips/{job_id}/{kind}')
def get_annotation(job_id: str, kind: str, start_ms: int = Query(0,ge=0), end_ms: int | None = Query(None,ge=0)):
    job=get_job(job_id)
    if kind not in ROUTE_FILES:
        raise HTTPException(404, 'Unknown annotation category')
    module, default_file = ROUTE_FILES[kind]
    if module not in job['completed_modules']:
        raise HTTPException(404, f'{module} result not available')
    path=job['directory']/'results'/default_file
    if not path.is_file():
        raise HTTPException(404, f'{module} output file missing')
    if kind=='hands':
        result=[]
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                if not line.strip(): continue
                item=json.loads(line)
                t=item.get('t_ms',-1)
                if t >= start_ms and (end_ms is None or t < end_ms):
                    result.append(item)
                    if len(result)>=15000:break
        return result
    data=json.loads(path.read_text(encoding='utf-8'))
    if start_ms or end_ms is not None:
        data={**data}
        if 'frames' in data:data['frames']=[f for f in data['frames'] if f.get('t_ms',-1)>=start_ms and (end_ms is None or f.get('t_ms',0)<end_ms)]
        if 'actions' in data:data['actions']=[a for a in data['actions'] if a['end_ms']>start_ms and (end_ms is None or a['start_ms']<end_ms)]
        if 'contacts' in data:data['contacts']=[c for c in data['contacts'] if c['end_ms']>start_ms and (end_ms is None or c['start_ms']<end_ms)]
    return data


class SearchRequest(BaseModel):
    query: str = Field('',max_length=200)
    hand: str | None = None
    limit: int = Field(25,ge=1,le=100)


@app.post('/api/search')
def search(req:SearchRequest):
    term=req.query.casefold().strip()
    matches=[]
    with LOCK:
        ids=list(JOBS)
    for jid in ids:
        job=get_job(jid)
        if 'ps3' not in job['completed_modules']:continue
        path=job['directory']/'results'/'actions.json'
        if not path.exists():continue
        try: actions=json.loads(path.read_text(encoding='utf-8'))['actions']
        except (ValueError,KeyError):continue
        for action in actions:
            text=' '.join(str(action.get(k) or '') for k in ('verb','object_label','hand')).casefold()
            if (not term or term in text) and (not req.hand or action.get('hand') in {req.hand,'both'}):
                matches.append({'clip_id':jid,'clip_name':job['name'],**action})
                if len(matches)>=req.limit:return {'results':matches}
    return {'results':matches}


@app.delete('/api/jobs/{job_id}', status_code=204)
def delete_job(job_id:str):
    # Do not allow deletion while processing; prevents races with file generation.
    with LOCK:
        job=JOBS.get(job_id)
        if job is None:raise HTTPException(404,'Job not found')
        if job['status'] in {'queued','processing'}:raise HTTPException(409,'Job still processing')
        del JOBS[job_id]
    shutil.rmtree(job['directory'],ignore_errors=True)


@app.delete('/api/data')
def delete_all_finished_jobs():
    with LOCK: ids=[j for j,job in JOBS.items() if job['status'] not in {'queued','processing'}]
    for jid in ids:delete_job(jid)
    return {'deleted_jobs':len(ids),'note':'Only completed local jobs were deleted; verify all external copies separately.'}
