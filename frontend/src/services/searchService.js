import {useQuery} from '@tanstack/react-query';
import {apiFetch,useRealApi} from './apiClient';
import {searchInLibrary} from './clipData';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
export function useSearchResults(query,filters,overrides){
  const clips=useMediaStore(s=>s.clips),byClip=useInferenceStore(s=>s.byClip);
  return useQuery({
    queryKey:['search',query,filters,overrides,clips.map(c=>c.id),byClip],
    queryFn:async({signal})=>{
      if(!useRealApi)return searchInLibrary(clips,byClip,overrides,query,filters);
      const response=await apiFetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,filters}),signal});
      return response.results||[];
    },
    staleTime:useRealApi?30000:Infinity,
  });
}
