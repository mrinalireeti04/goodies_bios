import {useQuery} from '@tanstack/react-query';
import {apiFetch,useRealApi} from './apiClient';
import {searchInLibrary} from './clipData';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {useAnnotationStore} from '@/stores/useAnnotationStore';
export function useSearchResults(query,filters,overrides){
  const clips=useMediaStore(s=>s.clips),byClip=useInferenceStore(s=>s.byClip),added=useAnnotationStore(s=>s.addedActions);
  return useQuery({
    queryKey:['search',query,filters,overrides,clips.map(c=>c.id),byClip,added],
    queryFn:async({signal})=>{
      const local=searchInLibrary(clips,byClip,overrides,query,filters,added);
      if(!useRealApi)return local;
      const response=await apiFetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,filters}),signal});
      const covered=new Set(clips.filter(c=>byClip[c.id]||added[c.id]?.length).map(c=>c.jobId||c.id));
      return [...local,...(response.results||[]).filter(event=>!covered.has(event.clip_id))];
    },
    staleTime:useRealApi?30000:Infinity,
  });
}
