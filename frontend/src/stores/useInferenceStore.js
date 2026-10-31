import {create} from 'zustand';
import {mergeResultFiles} from '@/services/annotationData';
export const useInferenceStore=create((set)=>({
  byClip:{},
  importForClip:(clipId,parsedFiles)=>set(s=>({byClip:{...s.byClip,[clipId]:mergeResultFiles(s.byClip[clipId],parsedFiles)}})),
  clearClip:(clipId)=>set(s=>{const byClip={...s.byClip};delete byClip[clipId];return {byClip};}),
}));
