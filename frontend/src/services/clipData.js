import {demoClip,syntheticFrameAt,getMergedActions,mockContacts} from '../mocks/demoData.js';
import {frameWithObjects,threeDHandsAt,mergeActionOverrides,actionMatches} from './annotationData.js';
export function clipFrame(clip,results,tMs){return !clip?null:clip.type==='demo'?syntheticFrameAt(tMs):frameWithObjects(results,tMs);}
export function clip3DHands(clip,results,tMs){return !clip?[]:clip.type==='demo'?(syntheticFrameAt(tMs).hands||[]):threeDHandsAt(results,tMs);}
export function clipActions(clip,results,overrides={}){
  if(!clip)return [];
  return clip.type==='demo'?getMergedActions(overrides):mergeActionOverrides(results?.modules.ps3?.actions,overrides);
}
export function clipContacts(clip,results){return clip?.type==='demo'?mockContacts:(results?.modules.ps2?.contacts||[]);}
export function searchInLibrary(clips,byClip,allOverrides,query,filters){
  return clips.flatMap(clip=>clipActions(clip,byClip[clip.id],allOverrides[clip.id]||{}).filter(event=>actionMatches(event,query,filters)).map(event=>({...event,clip_id:clip.id,clip_name:clip.name,synthetic:clip.type==='demo'})));
}
