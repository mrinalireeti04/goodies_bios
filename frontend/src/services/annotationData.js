// Shared, pure functions for model-produced annotations. No private media is bundled.
import { normalizePS1Frames } from './adapters/ps1Adapter.js';
import { normalizePS2 } from './adapters/ps2Adapter.js';
import { normalizePS3 } from './adapters/ps3Adapter.js';
import { normalizePS4 } from './adapters/ps4Adapter.js';

export const MODULES = Object.freeze(['ps1','ps2','ps3','ps4']);

export function parseResultFile(name, contents) {
  const lower = name.toLowerCase();
  let module;
  if (lower.endsWith('.jsonl') || lower.includes('hands')) module = 'ps1';
  else if (lower.includes('ps2') || lower.includes('contact')) module = 'ps2';
  else if (lower.includes('ps3') || lower.includes('action')) module = 'ps3';
  else if (lower.includes('ps4') || lower.includes('pose3d')) module = 'ps4';
  else throw new Error(`${name}: name the file hands.jsonl, ps2.json, actions.json or ps4.json`);
  let value;
  try {
    if (module==='ps1') {
      if (lower.endsWith('.jsonl')) {
        value = contents.split(/\r?\n/).filter(line=>line.trim()).map((line,index)=>{
          try {return JSON.parse(line);} catch {throw new Error(`${name}: invalid JSONL on line ${index+1}`);}
        });
      } else {
        const parsed=JSON.parse(contents);value=Array.isArray(parsed)?parsed:parsed.frames;
      }
      if (!Array.isArray(value)||!value.length) throw new Error('PS1 file needs frame entries');
      value=normalizePS1Frames(value);
    } else {
      value=JSON.parse(contents);
      if(module==='ps2')value=normalizePS2(value);
      if(module==='ps3')value=normalizePS3(value);
      if(module==='ps4')value=normalizePS4(value);
    }
  } catch(e) { throw new Error(`${name}: ${e.message}`); }
  const clipIds=new Set(module==='ps1'?value.map(f=>f.clip_id):[value.clip_id]);
  if(clipIds.size!==1||![...clipIds][0])throw new Error(`${name}: missing or inconsistent clip_id`);
  return {module,data:value,sourceClipId:[...clipIds][0]};
}

export function mergeResultFiles(existing, parsedFiles) {
  const results={...(existing||{}),modules:{...(existing?.modules||{})}};
  const sourceIds=new Set([results.sourceClipId].filter(Boolean));
  const seen=new Set();
  for(const item of parsedFiles){
    if(seen.has(item.module))throw new Error(`Duplicate ${item.module} result in this upload`);
    seen.add(item.module);sourceIds.add(item.sourceClipId);
  }
  if(sourceIds.size>1)throw new Error('PS1–PS4 files must refer to the same source clip_id. Check the files before importing.');
  for(const item of parsedFiles)results.modules[item.module]=item.data;
  // A team-wide PS1 hand ID is the source of truth. Reject obvious mismatches.
  if(results.modules.ps1){
    const ids=new Set(results.modules.ps1.flatMap(f=>f.hands.map(h=>h.hand_id)));
    const handRefs=[
      ...(results.modules.ps2?.contacts||[]).map(c=>c.hand_id),
      ...(results.modules.ps2?.frames||[]).flatMap(f=>f.hand_states.map(h=>h.hand_id)),
      ...(results.modules.ps3?.actions||[]).flatMap(a=>a.hand_ids),
      ...(results.modules.ps4?.frames||[]).flatMap(f=>f.hands.map(h=>h.hand_id)),
    ];
    const unknown=handRefs.filter(id=>id&&!ids.has(id));
    if(unknown.length)throw new Error(`Unknown hand_id in a downstream module: ${unknown[0]}. PS1 must own all tracked hand IDs.`);
  }
  results.sourceClipId=[...sourceIds][0]||null;
  return results;
}

// Use a bounded nearest observation. Never carry forward a hand pose during prolonged missing detections.
export function closestFrame(frames,tMs,maxGapMs=150){
  if(!Array.isArray(frames)||!frames.length||!Number.isFinite(tMs))return null;
  let lo=0,hi=frames.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].t_ms<tMs)lo=mid+1;else hi=mid;}
  const a=lo>0?frames[lo-1]:null,b=lo<frames.length?frames[lo]:null;
  const best=!a?b:!b?a:(Math.abs(a.t_ms-tMs)<=Math.abs(b.t_ms-tMs)?a:b);
  return best&&Math.abs(best.t_ms-tMs)<=maxGapMs?best:null;
}

export function frameWithObjects(results,tMs){
  if(!results)return null;
  const hand=closestFrame(results.modules.ps1,tMs);
  const objectFrame=closestFrame(results.modules.ps2?.frames,tMs,250);
  if(!hand&&!objectFrame)return null;
  return {
    ...(hand||{}),
    t_ms:hand?.t_ms??objectFrame.t_ms,
    hands:hand?.hands||[],
    objects:objectFrame?.objects||[],
    frame_width:hand?.frame_width??objectFrame?.frame_width,
    frame_height:hand?.frame_height??objectFrame?.frame_height,
  };
}

export function threeDHandsAt(results,tMs){
  if(!results)return [];
  const frame=closestFrame(results.modules.ps4?.frames,tMs)||closestFrame(results.modules.ps1,tMs);
  return (frame?.hands||[]).filter(h=>Array.isArray(h.joints_3d_m||h.kp3d_hand_m))
    .map(h=>({...h,kp3d_hand_m:h.joints_3d_m||h.kp3d_hand_m}));
}

export function mergeActionOverrides(actions,overrides={}){
  return (actions||[]).map(a=>({...a,...(overrides[a.action_id]||{})}));
}
export function actionMatches(event,query='',filters={}){
  const q=query.trim().toLowerCase();
  const haystack=[event.verb,event.object_label,event.hand,event.action_id].filter(Boolean).join(' ').toLowerCase();
  return (!q||haystack.includes(q))&&(!filters.hand||filters.hand==='all'||event.hand===filters.hand||event.hand==='both')&&(!filters.action||filters.action==='all'||event.verb===filters.action);
}
