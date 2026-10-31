export function normalizePS3(data){
  if(!data||typeof data.clip_id!=='string'||!data.clip_id||!Array.isArray(data.actions))throw new Error('Invalid PS3 action timeline');
  for(const a of data.actions){
    if(typeof a.action_id!=='string'||!a.action_id||typeof a.verb!=='string'||!a.verb||!Number.isInteger(a.start_ms)||!Number.isInteger(a.end_ms)||a.start_ms<0||a.end_ms<=a.start_ms||!['left','right','both','unknown'].includes(a.hand)||!Array.isArray(a.hand_ids))throw new Error(`Invalid PS3 action ${a.action_id||'(unnamed)'}`);
  }
  return {...data,actions:[...data.actions].sort((a,b)=>a.start_ms-b.start_ms)};
}
