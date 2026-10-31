import { create } from 'zustand';
import { demoClip } from '@/mocks/demoData';
import {useInferenceStore} from '@/stores/useInferenceStore';
const allowed = /\.(mp4|mov|webm|jpg|jpeg|png)$/i;
const MAX_BYTES = 800 * 1024 * 1024;
export const useMediaStore = create((set,get)=>({
  clips:[demoClip], selectedClipId:demoClip.id, errors:[], uploadOpen:false,
  setUploadOpen:(uploadOpen)=>set({uploadOpen}),
  setSelectedClip:(selectedClipId)=>set({selectedClipId}),
  clearErrors:()=>set({errors:[]}),
  addFiles:(files)=>{
    const errors=[], added=[];
    for(const file of Array.from(files||[])){
      if(!allowed.test(file.name)){errors.push(`${file.name}: unsupported format`);continue;}
      if(file.size>MAX_BYTES){errors.push(`${file.name}: exceeds 800 MB`);continue;}
      const duplicate=get().clips.some(c=>c.name===file.name&&c.size===file.size&&c.type!=='demo');
      if(duplicate){errors.push(`${file.name}: already in the library`);continue;}
      const id=typeof crypto!=='undefined' && crypto.randomUUID ? crypto.randomUUID():`clip-${Date.now()}-${added.length}`;
      const type=/\.(jpg|jpeg|png)$/i.test(file.name)?'image':'video';
      added.push({id,name:file.name,size:file.size,type,status:'local',url:URL.createObjectURL(file),durationMs:0,width:null,height:null,synthetic:false,mimeType:file.type||'unknown'});
    }
    set(s=>({clips:[...s.clips,...added],selectedClipId:added.length?added[0].id:s.selectedClipId,errors,uploadOpen:added.length?false:s.uploadOpen}));
    return added;
  },
  updateClip:(id,values)=>set(s=>({clips:s.clips.map(c=>c.id===id?{...c,...values}:c)})),
  removeClip:(id)=>{
    const c=get().clips.find(x=>x.id===id);
    if(c && !c.synthetic && c.url) URL.revokeObjectURL(c.url);
    useInferenceStore.getState().clearClip(id);
    set(s=>{const clips=s.clips.filter(x=>x.id!==id);return {clips,selectedClipId:s.selectedClipId===id?(clips[0]?.id||null):s.selectedClipId};});
  }
}));
