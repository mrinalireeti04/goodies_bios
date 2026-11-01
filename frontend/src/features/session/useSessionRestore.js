import {useEffect} from 'react';
import {apiFetch,mediaUrl,useRealApi} from '@/services/apiClient';
import {loadJobResults,waitForJob} from '@/services/jobService';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';

// Reconnect persisted browser metadata to jobs that still exist in the local API.
// Browser object URLs cannot survive a refresh, but backend-backed media and results can.
export function useSessionRestore(){
  useEffect(()=>{
    if(!useRealApi)return;
    let cancelled=false;
    const controllers=[];
    async function restore(){
      if(!useInferenceStore.persist.hasHydrated())await useInferenceStore.persist.rehydrate();
      if(cancelled)return;
      const clips=useMediaStore.getState().clips.filter(c=>c.type!=='demo');
      if(!clips.length)return;
      let jobs;
      try{jobs=await apiFetch('/api/clips');}catch{
        if(!cancelled)clips.filter(clip=>['uploading','queued','processing','importing'].includes(clip.status)).forEach(clip=>useMediaStore.getState().updateClip(clip.id,{status:'disconnected',error:'Could not reconnect to the backend. Check the server and refresh status.'}));
        return;
      }
      for(const clip of clips){
        if(cancelled)break;
        const job=(clip.jobId&&jobs.find(item=>item.job_id===clip.jobId))
          ||jobs.find(item=>item.client_clip_id===clip.id)
          ||(clip.name&&Number.isFinite(clip.size)?jobs.filter(item=>item.name===clip.name&&item.size===clip.size).at(-1):null);
        if(!job){if(['queued','processing','importing'].includes(clip.status))useMediaStore.getState().updateClip(clip.id,{status:'disconnected',error:'Saved backend job not found.'});continue;}
        let currentJob=job;
        const loadedModules=new Set();
        let importQueue=Promise.resolve();
        let importError=null;
        const importAvailable=async(state)=>{
          const completed=(state.completed_modules||[]).filter(module=>!loadedModules.has(module));
          if(!completed.length)return;
          completed.forEach(module=>loadedModules.add(module));
          try{
            const parsed=await loadJobResults({...state,completed_modules:completed});
            if(!cancelled&&parsed.length)useInferenceStore.getState().importForClip(clip.id,parsed);
          }catch(error){
            completed.forEach(module=>loadedModules.delete(module));
            importError=`Saved results could not be restored: ${error.message}`;
            if(!cancelled)useMediaStore.getState().updateClip(clip.id,{error:importError});
          }
        };
        const updateFromJob=(state)=>{
          if(cancelled)return;
          useMediaStore.getState().updateClip(clip.id,{status:['completed','partial','failed'].includes(state.status)?'importing':state.status,issues:state.issues||[],moduleStatus:state.module_status||{},modules:state.completed_modules||[],originalUrl:mediaUrl(state.job_id),overlayUrl:state.completed_modules?.includes('ps1')?mediaUrl(state.job_id,'overlay'):null});
          importQueue=importQueue.then(()=>importAvailable(state));
        };
        if(['queued','processing'].includes(currentJob.status)){
          const controller=new AbortController();
          controllers.push(controller);
          try{currentJob=await waitForJob(currentJob.job_id,{signal:controller.signal,onStatus:updateFromJob});}catch(error){if(!cancelled)useMediaStore.getState().updateClip(clip.id,{status:'disconnected',error:`Could not check job status: ${error.message}`});continue;}
        }
        if(cancelled)break;
        await importQueue;
        if(currentJob.completed_modules?.length)await importAvailable(currentJob);
        if(cancelled)break;
        const next={jobId:currentJob.job_id,status:currentJob.status,error:importError,issues:currentJob.issues||[],modules:currentJob.completed_modules||[],moduleStatus:currentJob.module_status||{},originalUrl:mediaUrl(currentJob.job_id),overlayUrl:currentJob.completed_modules?.includes('ps1')?mediaUrl(currentJob.job_id,'overlay'):null};
        useMediaStore.getState().updateClip(clip.id,next);
      }
    }
    restore();
    return()=>{cancelled=true;controllers.forEach(controller=>controller.abort());};
  },[]);
}
