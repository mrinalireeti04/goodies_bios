import {create} from 'zustand';
import {persist} from 'zustand/middleware';
// Corrections are keyed by the application clip ID, then the action ID.
// Original inference records are never mutated.
export const useAnnotationStore=create(persist(set=>({
  selectedActionId:'A003',overrides:{},history:[],objectOverrides:{},addedActions:{},manualObjects:{},
  selectAction:(selectedActionId)=>set({selectedActionId}),
  updateObject:(clipId,id,values)=>set(s=>({objectOverrides:{...s.objectOverrides,[clipId]:{...s.objectOverrides[clipId],[id]:{...s.objectOverrides[clipId]?.[id],...values}}}})),
  addAction:(clipId,action)=>set(s=>({addedActions:{...s.addedActions,[clipId]:[...(s.addedActions[clipId]||[]),action]}})),
  addObject:(clipId,object)=>set(s=>({manualObjects:{...s.manualObjects,[clipId]:[...(s.manualObjects[clipId]||[]),object]}})),
  updateAction:(clipId,id,values)=>set(s=>{
    const current=s.overrides[clipId]||{};
    return {overrides:{...s.overrides,[clipId]:{...current,[id]:{...current[id],...values}}},history:[...s.history,{clipId,id,previous:current[id]||null}]};
  }),
  mergeActions:(clipId,target,source)=>set(s=>{
    const current=s.overrides[clipId]||{};
    const previousOverrides={[target.action_id]:current[target.action_id]||null,[source.action_id]:current[source.action_id]||null};
    const next={...current,[target.action_id]:{...current[target.action_id],start_ms:Math.min(target.start_ms,source.start_ms),end_ms:Math.max(target.end_ms,source.end_ms),hand_ids:[...new Set([...(target.hand_ids||[]),...(source.hand_ids||[])])],confidence:null},[source.action_id]:{...current[source.action_id],deleted:true,mergedInto:target.action_id}};
    return {overrides:{...s.overrides,[clipId]:next},history:[...s.history,{clipId,type:'merge',previousOverrides}]};
  }),
  undo:()=>set(s=>{
    if(!s.history.length)return s;
    const prev=s.history.at(-1),overrides={...s.overrides},clipOverrides={...(overrides[prev.clipId]||{})};
    if(prev.type==='merge')for(const [id,value] of Object.entries(prev.previousOverrides)){if(value===null)delete clipOverrides[id];else clipOverrides[id]=value;}
    else if(prev.previous===null)delete clipOverrides[prev.id];else clipOverrides[prev.id]=prev.previous;
    overrides[prev.clipId]=clipOverrides;
    return {overrides,history:s.history.slice(0,-1)};
  }),
}),{name:'forge-annotation-session-v1'}));
