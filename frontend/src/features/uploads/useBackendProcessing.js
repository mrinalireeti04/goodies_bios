import {useMutation,useQueryClient} from '@tanstack/react-query';
import {createJob,waitForJob,loadJobResults} from '@/services/jobService';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';

/** Optional server processing; frontend-only preview remains the default. */
export function useBackendProcessing(){
  const queryClient=useQueryClient();
  return useMutation({
    mutationFn:async ({clip,file})=>{
      const media=useMediaStore.getState();
      media.updateClip(clip.id,{status:'uploading',error:null});
      const job=await createJob(file,clip.id);
      media.updateClip(clip.id,{jobId:job.job_id,status:'queued'});
      const final=await waitForJob(job.job_id,{
        onStatus:state=>{
          const stillPresent=useMediaStore.getState().clips.some(x=>x.id===clip.id);
          if(stillPresent)useMediaStore.getState().updateClip(clip.id,{status:state.status,issues:state.issues});
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
          status:final.status,issues:final.issues||[],modules:final.completed_modules||[]
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
