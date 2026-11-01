import {create} from 'zustand';
import {persist,createJSONStorage} from 'zustand/middleware';
import {mergeResultFiles} from '@/services/annotationData';
import {inferenceStorage} from './inferenceStorage';
export const useInferenceStore=create(persist((set)=>({
  byClip:{},
  importForClip:(clipId,parsedFiles)=>set(s=>({byClip:{...s.byClip,[clipId]:mergeResultFiles(s.byClip[clipId],parsedFiles)}})),
  clearClip:(clipId)=>set(s=>{const byClip={...s.byClip};delete byClip[clipId];return {byClip};}),
}),{name:'forge-inference-session-v1',storage:createJSONStorage(()=>inferenceStorage)}));
