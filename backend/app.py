"""FORGE Studio: local FastAPI job bridge for the independent PS1–PS4 modules.

Run: uvicorn backend.app:app --host 127.0.0.1 --port 8000
The API is deliberately local-only, unauthenticated, and not safe for public hosting.
"""
import importlib
import importlib.util
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
except ImportError:
    logging.getLogger('forge.config').warning('python-dotenv is unavailable; project .env was not loaded')
DATA_DIR = Path(os.environ.get('FORGE_PRIVATE_DIR', ROOT / '.local_data')).resolve()
ALLOWED_EXT = {'.mp4', '.mov', '.webm', '.jpg', '.jpeg', '.png'}
MAX_BYTES = 800 * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.RLock()
MODULE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='forge-module')

# Recover completed local jobs; interrupted work is explicitly failed, never replayed.
for metadata in DATA_DIR.glob('*/job.json'):
    try:
        saved = json.loads(metadata.read_text())
        candidates = list(metadata.parent.glob('input.*'))
        if not candidates:
            continue
        saved.update(path=candidates[0], directory=metadata.parent)
        if saved['status'] in ('queued','processing'):
            saved.update(status='failed', issues=['Processing interrupted; upload again to retry.'])
            # An interrupted worker cannot continue after restart. Keep completed
            # modules available and stop reporting its unfinished stages as active.
            saved['module_status'] = {
                name: ('failed' if state == 'processing' else 'skipped' if state == 'queued' else state)
                for name, state in (saved.get('module_status') or {}).items()
            }
        JOBS[saved['job_id']] = saved
    except (ValueError, KeyError, OSError):
        continue

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
            job = JOBS[job_id]
            temporary = job['directory']/'job.tmp'
            temporary.write_text(json.dumps(public_job(job)))
            temporary.replace(job['directory']/'job.json')


def update_module(job_id: str, name: str, status: str, *, completed: bool | None = None, issue: str | None = None):
    """Merge one module's state under the job lock, so concurrent stages cannot overwrite each other."""
    with LOCK:
        job = JOBS[job_id]
        states = dict(job.get('module_status') or {})
        states[name] = status
        timings = dict(job.get('module_timing') or {})
        if status == 'processing':
            timings[name] = {'started_at': time.time()}
        elif name in timings and 'started_at' in timings[name]:
            timings[name] = {**timings[name], 'duration_s': round(time.time() - timings[name]['started_at'], 3)}
        modules = set(job.get('completed_modules') or [])
        if completed is True:
            modules.add(name)
        elif completed is False:
            modules.discard(name)
        issues = [i for i in job.get('issues', []) if not i.startswith(f'{name}:')]
        if issue:
            issues.append(issue)
        put(job_id, module_status=states, completed_modules=sorted(modules), issues=issues, module_timing=timings)


def process_ps3(job_id: str, contacts_future=None):
    """Run PS3 independently using an existing job's PS1 hand tracks."""
    job = get_job(job_id)
    output_dir = job['directory'] / 'results'
    hands_path = output_dir / 'hands.jsonl'
    if 'ps1' not in job.get('completed_modules', []) or not hands_path.is_file():
        raise RuntimeError('PS3 requires completed PS1 hand tracks')
    update_module(job_id, 'ps3', 'processing')
    try:
        adapter = importlib.import_module('backend.inference.ps3.adapter')
        contacts = output_dir / 'ps2.json'
        result = Path(adapter.run_ps3(job['path'], hands_path, contacts, output_dir, job_id,
                                      contacts_future=contacts_future)).resolve()
        if not result.is_relative_to(output_dir.resolve()) or not result.is_file():
            raise ValueError('PS3 must return an existing private output file')
        timeline_path = output_dir / 'ps3_timeline.json'
        timeline = json.loads(timeline_path.read_text()) if timeline_path.is_file() else {}
        failed_segments = (timeline.get('processing_metadata') or {}).get('failed_segments') or []
        state = 'partial' if failed_segments else 'completed'
        issue = None
        if failed_segments:
            issue = f"ps3: partial result; Gemini analysis failed for segment(s) {', '.join(map(str, failed_segments))}"
            logging.getLogger('forge.ps3').warning('%s for job %s', issue, job_id)
        update_module(job_id, 'ps3', state, completed=True, issue=issue)
    except Exception as exc:
        logging.getLogger('forge.ps3').exception('PS3 processing failed for job %s', job_id)
        update_module(job_id, 'ps3', 'failed', issue=f'ps3: processing failed ({type(exc).__name__}); see backend log')


def process_ps2(job_id: str):
    job = get_job(job_id)
    output_dir = job['directory'] / 'results'
    try:
        update_module(job_id, 'ps2', 'processing')
        module = importlib.import_module('backend.inference.ps2_adapter')
        path = module.run_ps2(job['path'], output_dir / 'hands.jsonl', output_dir / 'ps2.json',
                              job_id, job['directory'] / 'ps2_work')
        result = Path(path).resolve()
        if not result.is_relative_to(output_dir.resolve()) or not result.is_file():
            raise ValueError('PS2 must return an existing private output file')
        update_module(job_id, 'ps2', 'completed', completed=True)
    except (ImportError, AttributeError) as exc:
        update_module(job_id, 'ps2', 'unavailable', issue=f'ps2: module unavailable ({type(exc).__name__})')
    except Exception as exc:
        logging.getLogger('forge.ps2').exception('PS2 processing failed for job %s', job_id)
        update_module(job_id, 'ps2', 'failed', issue=f'ps2: processing failed ({type(exc).__name__}); see backend log')


def process_job(job_id: str):
    """Run PS1/PS4 first, then schedule PS2 and optional PS3 concurrently."""
    job = get_job(job_id)
    if job['media_type'] != 'video':
        put(job_id, status='partial', note='Still-image inference adapter not yet installed')
        return

    input_path, output_dir = job['path'], job['directory'] / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    issues: list[str] = []
    module_status = dict(job.get('module_status') or {})
    job_started=time.time()
    put(job_id, status='processing', processing_started_at=job_started)

    stages = [('ps1', 'backend.inference.pipeline', 'run_ps1', lambda: (input_path, output_dir))]
    for name, module_name, func_name, args_fn in stages:
        module_status[name] = 'processing'
        stage_started=time.time()
        put(job_id, module_status=module_status, module_timing={**(get_job(job_id).get('module_timing') or {}),name:{'started_at':stage_started}})
        try:
            module = importlib.import_module(module_name)
            run = getattr(module, func_name)
            result = Path(run(*args_fn())).resolve()
            if not result.is_relative_to(output_dir.resolve()) or not result.is_file():
                raise ValueError('Module must return an existing private output file')
            outputs[name] = str(result)
            module_status[name] = 'completed'
            if name == 'ps1' and (output_dir / 'ps4.json').is_file():
                outputs['ps4'] = str(output_dir / 'ps4.json')
                module_status['ps4'] = 'completed'
            put(job_id, completed_modules=sorted(outputs), module_status=module_status)
        except (ImportError, AttributeError) as exc:
            module_status[name] = 'skipped'
            issues.append(f'{name}: module unavailable ({type(exc).__name__})')
            put(job_id, module_status=module_status)

        except Exception as exc:
            module_status[name] = 'failed'
            issues.append(f'{name}: {type(exc).__name__}: {str(exc)[:160]}')
            put(job_id, module_status=module_status)
        finally:
            with LOCK:
                timings=dict(get_job(job_id).get('module_timing') or {})
                timings[name]={**timings.get(name,{}),'duration_s':round(time.time()-stage_started,3)}
                put(job_id,module_timing=timings)

    if 'ps1' not in outputs:
        update_module(job_id, 'ps2', 'skipped', issue='ps2: requires successful PS1 hand inference')
        if job.get('enable_ps3'):
            update_module(job_id, 'ps3', 'skipped', issue='ps3: requires successful PS1 hand inference')
    else:
        ps2_future = MODULE_EXECUTOR.submit(process_ps2, job_id) if job.get('enable_ps2', True) else None
        if ps2_future is None:
            update_module(job_id, 'ps2', 'skipped')
        ps3_future = MODULE_EXECUTOR.submit(process_ps3, job_id, ps2_future) if job.get('enable_ps3') else None
        if ps2_future:
            ps2_future.result()
        if ps3_future:
            ps3_future.result()
        else:
            update_module(job_id, 'ps3', 'skipped')

    latest = get_job(job_id)
    status = 'completed' if 'ps1' in latest.get('completed_modules', []) and 'ps4' in latest.get('completed_modules', []) else 'failed'
    if status == 'completed' and any(v in {'failed', 'unavailable', 'partial'} for v in latest.get('module_status', {}).values()):
        status = 'partial'
    put(job_id, status=status, processing_duration_s=round(time.time()-job_started,3))


@app.get('/api/health')
def health():
    ps2_deps = all(importlib.util.find_spec(name) is not None for name in ('torch', 'ultralytics', 'yaml'))
    return {'ok':True, 'mode':'local', 'ps1_loaded': (ROOT/'backend'/'inference'/'pipeline.py').is_file(),
            'ps2_available': ps2_deps, 'ps3_available': (ROOT/'backend'/'inference'/'ps3'/'adapter.py').is_file(),
            'ps3_configured': bool(os.environ.get('GEMINI_API_KEY'))}


@app.post('/api/jobs', status_code=201)
async def submit_job(background_tasks: BackgroundTasks, file: UploadFile = File(...), client_clip_id: str = Form(''), enable_ps3: bool = Form(False), enable_ps2: bool = Form(True)):
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
           'status':'queued','completed_modules':[],'issues':[], 'enable_ps2':enable_ps2, 'enable_ps3':enable_ps3, 'module_status':{'ps1':'queued','ps2':'queued' if enable_ps2 else 'skipped','ps3':'queued' if enable_ps3 else 'skipped','ps4':'queued'}, 'path':path,'directory':directory}
    with LOCK:
        JOBS[job_id]=job
    put(job_id)
    background_tasks.add_task(process_job, job_id)
    return public_job(job)


@app.get('/api/jobs/{job_id}')
def get_status(job_id: str):
    return public_job(get_job(job_id))


@app.post('/api/jobs/{job_id}/ps3', status_code=202)
def retry_ps3(job_id: str, background_tasks: BackgroundTasks):
    job = get_job(job_id)
    if 'ps1' not in job.get('completed_modules', []) or not (job['directory']/'results'/'hands.jsonl').is_file():
        raise HTTPException(409, 'PS3 retry requires completed PS1 hand results')
    if not os.environ.get('GEMINI_API_KEY', '').strip():
        raise HTTPException(503, 'Set GEMINI_API_KEY in the project .env file to run PS3')
    if (job.get('module_status') or {}).get('ps3') == 'processing':
        raise HTTPException(409, 'PS3 is already processing')
    module_status = dict(job.get('module_status') or {})
    module_status['ps3'] = 'queued'
    put(job_id, module_status=module_status)
    background_tasks.add_task(process_ps3, job_id)
    return public_job(get_job(job_id))


@app.get('/api/clips')
def get_clips():
    with LOCK:
        return [public_job(job) for job in JOBS.values()]


@app.get('/api/clips/{job_id}')
def get_clip(job_id: str):
    return public_job(get_job(job_id))


class ManualMaskRequest(BaseModel):
    frame_idx: int = Field(ge=0, strict=True)
    bbox_px: list[float] = Field(min_length=4, max_length=4)


@app.post('/api/clips/{job_id}/manual-mask')
def manual_mask(job_id: str, request: ManualMaskRequest):
    job = get_job(job_id)
    if job.get('media_type') != 'video':
        raise HTTPException(400, 'Manual masks currently require a video')
    with LOCK:
        if any(j.get('status') in ('queued', 'processing') or 'processing' in (j.get('module_status') or {}).values() for j in JOBS.values()):
            raise HTTPException(409, 'Wait for active processing to finish before generating a mask')
    try:
        from backend.inference.manual_mask import segment_frame
        polygon = segment_frame(job['path'], request.frame_idx, request.bbox_px)
        return {'frame_idx':request.frame_idx, 'mask_polygon_px':polygon}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logging.getLogger('forge.manual_mask').exception('Manual mask generation failed')
        raise HTTPException(503, 'Manual mask generation failed; check the backend log') from exc


@app.get('/api/clips/{job_id}/media')
def get_media(job_id: str):
    job=get_job(job_id)
    return FileResponse(job['path'], filename=job['name'])


ROUTE_FILES={'hands':('ps1','hands.jsonl'),'contacts':('ps2','ps2.json'), 'actions':('ps3','actions.json'), 'pose3d':('ps4','ps4.json')}


@app.get('/api/clips/{job_id}/overlay')
def get_overlay(job_id: str):
    job = get_job(job_id)
    path = job['directory']/'results'/'overlay.mp4'
    if 'ps1' not in job['completed_modules'] or not path.is_file():
        raise HTTPException(404, 'Overlay not available')
    return FileResponse(path, media_type='video/mp4')


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
    action: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(25,ge=1,le=100)


@app.post('/api/search')
def search(req:SearchRequest):
    term=req.query.casefold().strip()
    hand_filter=req.filters.get('hand', req.hand)
    action_filter=req.filters.get('action', req.action)
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
            if (not term or term in text) and (not hand_filter or hand_filter=='all' or action.get('hand') in {hand_filter,'both'}) and (not action_filter or action_filter=='all' or action.get('verb')==action_filter):
                matches.append({**action,'clip_id':job.get('client_clip_id') or jid,'job_id':jid,'clip_name':job['name']})
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
