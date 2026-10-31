import {create} from 'zustand';
// Corrections are keyed by the application clip ID, then the action ID.
// Original inference records are never mutated.
export const useAnnotationStore=create(set=>({
  selectedActionId:'A003',overrides:{},history:[],
  selectAction:(selectedActionId)=>set({selectedActionId}),
  updateAction:(clipId,id,values)=>set(s=>{
    const current=s.overrides[clipId]||{};
    return {overrides:{...s.overrides,[clipId]:{...current,[id]:{...current[id],...values}}},history:[...s.history,{clipId,id,previous:current[id]||null}]};
  }),
  undo:()=>set(s=>{
    if(!s.history.length)return s;
    const prev=s.history.at(-1),overrides={...s.overrides},clipOverrides={...(overrides[prev.clipId]||{})};
    if(prev.previous===null)delete clipOverrides[prev.id];else clipOverrides[prev.id]=prev.previous;
    overrides[prev.clipId]=clipOverrides;
    return {overrides,history:s.history.slice(0,-1)};
  }),
}));
