/** Local FastAPI inference adapter. Never uploads data when VITE_USE_REAL_API is false. */
import {apiFetch} from './apiClient.js';
import {parseResultFile} from './annotationData.js';

const OUTPUTS=[['ps1','hands','hands.jsonl'],['ps2','contacts','ps2.json'],['ps3','actions','actions.json'],['ps4','pose3d','ps4.json']];
const pause=(ms)=>new Promise(resolve=>setTimeout(resolve,ms));

export async function createJob(file, clientClipId) {
  const data=new FormData();
  data.append('file',file,file.name);
  data.append('client_clip_id',clientClipId);
  return apiFetch('/api/jobs',{method:'POST',body:data,timeoutMs:180000});
}

export async function fetchJob(jobId){return apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`);}

export async function waitForJob(jobId,{onStatus,signal,intervalMs=1700}={}){
  // Intentionally requires a real /api/jobs response; synthetic labels are never generated here.
  for(let attempt=0;attempt<1200;attempt++){
    if(signal?.aborted)throw new Error('Processing canceled');
    const state=await fetchJob(jobId);
    onStatus?.(state);
    if(['completed','partial','failed'].includes(state.status))return state;
    await pause(intervalMs);
  }
  throw new Error('Processing is taking longer than expected; you can check the job again later.');
}

export async function loadJobResults(job){
  const modules=new Set(job.completed_modules||[]);
  const parsed=[];
  for(const [module,route,name] of OUTPUTS){
    if(!modules.has(module))continue;
    const data=await apiFetch(`/api/clips/${encodeURIComponent(job.job_id)}/${route}`,{timeoutMs:120000});
    parsed.push(parseResultFile(name,JSON.stringify(data)));
  }
  return parsed;
}
