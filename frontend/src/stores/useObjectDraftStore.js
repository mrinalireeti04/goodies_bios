import {create} from 'zustand';
// Drawing is temporary: only a saved correction belongs in persistent annotations.
export const useObjectDraftStore=create(set=>({
  drawingClipId:null,draft:null,
  start:clipId=>set({drawingClipId:clipId,draft:null}),
  finish:(clipId,bbox,frame)=>set({drawingClipId:null,draft:{clipId,bbox,frameIndex:frame.frame_idx,tMs:frame.t_ms}}),
  cancel:()=>set({drawingClipId:null}),
  clear:()=>set({draft:null,drawingClipId:null}),
}));
