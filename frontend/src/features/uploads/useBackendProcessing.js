import {useMutation,useQueryClient} from '@tanstack/react-query';
import {createJob,waitForJob,loadJobResults,retryPS3,fetchJob,findJobForClientClip} from '@/services/jobService';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {mediaUrl} from '@/services/apiClient';

/** Optional server processing; frontend-only preview remains the default. */
export function useBackendProcessing(){
  const queryClient=useQueryClient();
  return useMutation({
    mutationFn:async ({clip,file,enablePS3,enablePS2})=>{
      const media=useMediaStore.getState();
      media.updateClip(clip.id,{status:'uploading',error:null});
      const job=await createJob(file,clip.id,enablePS3,enablePS2);
      media.updateClip(clip.id,{jobId:job.job_id,status:'queued'});
      const final=await waitForJob(job.job_id,{
        onStatus:state=>{
          const stillPresent=useMediaStore.getState().clips.some(x=>x.id===clip.id);
          if(stillPresent)useMediaStore.getState().updateClip(clip.id,{status:['completed','partial','failed'].includes(state.status)?'importing':state.status,issues:state.issues||[],modules:state.completed_modules||[],moduleStatus:state.module_status||{}});
          queryClient.setQueryData(['job',job.job_id],state);
        }
      });
      const parsed=await loadJobResults(final);
      return {clip,final,parsed};
    },
    onSuccess:({clip,final,parsed})=>{
      if(!useMediaStore.getState().clips.some(c=>c.id===clip.id))return;
      try{
        if(parsed.length)useInferenceStore.getState().importForClip(clip.id,parsed);
        useMediaStore.getState().updateClip(clip.id,{
          status:final.status,issues:final.issues||[],modules:final.completed_modules||[],moduleStatus:final.module_status||{},
          originalUrl:mediaUrl(final.job_id),
          overlayUrl:final.completed_modules?.includes('ps1')?mediaUrl(final.job_id,'overlay'):null
        });
        queryClient.invalidateQueries({queryKey:['clips']});
      }catch(err){
        useMediaStore.getState().updateClip(clip.id,{status:'partial',error:`Model outputs could not be imported: ${err.message}`});
      }
    },
    onError:(error,params)=>{
      useMediaStore.getState().updateClip(params.clip.id,{status:'failed',error:error.message});
    },
  });
}

/** Reconnect a failed browser upload to the job already stored by the local backend. */
export function useRecoverBackendClip(){
  const queryClient=useQueryClient();
  return useMutation({
    mutationFn:async ({clip})=>{
      let job=await findJobForClientClip(clip.id,clip);
      if(!job)throw new Error('No saved backend job was found for this clip.');
      if(['queued','processing'].includes(job.status))job=await waitForJob(job.job_id);
      const parsed=await loadJobResults(job);
      return {clip,job,parsed};
    },
    onSuccess:({clip,job,parsed})=>{
      if(!useMediaStore.getState().clips.some(item=>item.id===clip.id))return;
      if(parsed.length)useInferenceStore.getState().importForClip(clip.id,parsed);
      useMediaStore.getState().updateClip(clip.id,{
        jobId:job.job_id,status:job.status,error:null,issues:job.issues||[],
        modules:job.completed_modules||[],moduleStatus:job.module_status||{},
        originalUrl:mediaUrl(job.job_id),
        overlayUrl:job.completed_modules?.includes('ps1')?mediaUrl(job.job_id,'overlay'):null
      });
      queryClient.invalidateQueries({queryKey:['clips']});
    },
    onError:(error,{clip})=>useMediaStore.getState().updateClip(clip.id,{error:error.message})
  });
}

/** Retry optional PS3 on an existing clip without uploading or rerunning PS1. */
export function usePS3Retry(){
  const queryClient=useQueryClient();
  return useMutation({
    mutationFn:async ({clip})=>{
      if(!clip.jobId)throw new Error('This clip has no backend job to retry.');
      const state=await retryPS3(clip.jobId);
      useMediaStore.getState().updateClip(clip.id,{moduleStatus:state.module_status,ps3Error:null});
      for(let attempt=0;attempt<1200;attempt++){
        await new Promise(resolve=>setTimeout(resolve,1500));
        const current=await fetchJob(clip.jobId);
        useMediaStore.getState().updateClip(clip.id,{moduleStatus:current.module_status,issues:current.issues||[]});
        if(['completed','partial','failed','skipped'].includes(current.module_status?.ps3)){
          if(['failed','skipped'].includes(current.module_status.ps3))throw new Error(current.issues?.find(x=>x.startsWith('ps3:'))||`PS3 ${current.module_status.ps3}`);
          const parsed=await loadJobResults({...current,completed_modules:[...(current.completed_modules||[]),'ps3']});
          const onlyPS3=parsed.filter(item=>item.module==='ps3');
          if(!onlyPS3.length)throw new Error('PS3 completed without an actions result.');
          return {clip,current,onlyPS3};
        }
      }
      throw new Error('PS3 is taking longer than expected; check the job status again.');
    },
    onSuccess:({clip,current,onlyPS3})=>{
      useInferenceStore.getState().importForClip(clip.id,onlyPS3);
      useMediaStore.getState().updateClip(clip.id,{moduleStatus:current.module_status,modules:current.completed_modules,issues:current.issues||[],ps3Error:null});
      queryClient.invalidateQueries({queryKey:['search']});
    },
    onError:(error,{clip})=>useMediaStore.getState().updateClip(clip.id,{ps3Error:error.message}),
  });
}
