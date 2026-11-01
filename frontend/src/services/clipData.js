import {demoClip,syntheticFrameAt,getMergedActions,mockContacts} from '../mocks/demoData.js';
import {frameWithObjects,threeDHandsAt,mergeActionOverrides,actionMatches,applyObjectCorrections} from './annotationData.js';
export function clipFrame(clip,results,tMs,objectEdits={},manualObjects=[]){const frame=!clip?null:clip.type==='demo'?syntheticFrameAt(tMs):frameWithObjects(results,tMs);return frame?{...frame,objects:applyObjectCorrections([...(frame.objects||[]),...manualObjects.filter(o=>Math.abs(o.t_ms-(frame.t_ms??tMs))<=20)],objectEdits)}:null;}
export function clip3DHands(clip,results,tMs){return !clip?[]:clip.type==='demo'?(syntheticFrameAt(tMs).hands||[]):threeDHandsAt(results,tMs);}
export function clipActions(clip,results,overrides={},added=[]){
  if(!clip)return [];
  return mergeActionOverrides([...(clip.type==='demo'?getMergedActions(overrides):(results?.modules.ps3?.actions||[])),...added],overrides).sort((a,b)=>a.start_ms-b.start_ms);
}
export function clipContacts(clip,results){return clip?.type==='demo'?mockContacts:(results?.modules.ps2?.contacts||[]);}
export function searchInLibrary(clips,byClip,allOverrides,query,filters,allAdded={}){
  return clips.flatMap(clip=>clipActions(clip,byClip[clip.id],allOverrides[clip.id]||{},allAdded[clip.id]||[]).filter(event=>actionMatches(event,query,filters)).map(event=>({...event,clip_id:clip.id,clip_name:clip.name,synthetic:clip.type==='demo'})));
}
