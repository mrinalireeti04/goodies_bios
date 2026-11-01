/** Local FastAPI inference adapter. Never uploads data when VITE_USE_REAL_API is false. */
import {apiFetch} from './apiClient.js';
import {parseResultFile} from './annotationData.js';

const OUTPUTS=[['ps1','hands','hands.jsonl'],['ps2','contacts','ps2.json'],['ps3','actions','actions.json'],['ps4','pose3d','ps4.json']];
const pause=(ms)=>new Promise(resolve=>setTimeout(resolve,ms));

export async function createJob(file, clientClipId, enablePS3 = false, enablePS2 = true) {
  const data=new FormData();
  data.append('file',file,file.name);
  data.append('client_clip_id',clientClipId);
  data.append('enable_ps3',String(enablePS3));
  data.append('enable_ps2',String(enablePS2));
  return apiFetch('/api/jobs',{method:'POST',body:data,timeoutMs:180000});
}

export async function findJobForClientClip(clientClipId,clip){
  const jobs=await apiFetch('/api/clips');
  const byClientId=clientClipId?jobs.find(job=>job.client_clip_id===clientClipId):null;
  if(byClientId)return byClientId;
  if(!clip?.name||!Number.isFinite(clip.size))return null;
  const exact=jobs.filter(job=>job.name===clip.name&&job.size===clip.size);
  return exact.length===1?exact[0]:null;
}

export async function fetchJob(jobId){return apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`);}
export async function generateManualMask(jobId,frameIndex,bbox){return apiFetch(`/api/clips/${encodeURIComponent(jobId)}/manual-mask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame_idx:frameIndex,bbox_px:bbox}),timeoutMs:120000});}
export async function retryPS3(jobId){return apiFetch(`/api/jobs/${encodeURIComponent(jobId)}/ps3`,{method:'POST'});}

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
    const contents = module === 'ps1' && Array.isArray(data)
      ? data.map(frame=>JSON.stringify(frame)).join('\n')
      : JSON.stringify(data);
    parsed.push(parseResultFile(name,contents));
  }
  return parsed;
}
