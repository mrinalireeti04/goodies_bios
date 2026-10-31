const bbox=b=>Array.isArray(b)&&b.length===4&&b.every(Number.isFinite)&&b[2]>=b[0]&&b[3]>=b[1];
export function normalizePS2(data){
  if(!data||typeof data.clip_id!=='string'||!data.clip_id||!Array.isArray(data.frames)||!Array.isArray(data.contacts))throw new Error('Invalid PS2 data');
  for(const frame of data.frames){
    if(!Number.isInteger(frame.t_ms)||!Array.isArray(frame.objects)||!Array.isArray(frame.hand_states))throw new Error('Invalid PS2 frame');
    for(const o of frame.objects)if(typeof o.object_id!=='string'||!o.object_id||!bbox(o.bbox_px))throw new Error('Invalid PS2 object ID/bbox');
    for(const h of frame.hand_states)if(typeof h.hand_id!=='string'||!h.hand_id)throw new Error('Invalid PS2 hand reference');
  }
  for(const contact of data.contacts){
    if(!contact.event_id||!contact.hand_id||!Number.isInteger(contact.start_ms)||!Number.isInteger(contact.end_ms)||contact.start_ms<0||contact.end_ms<=contact.start_ms)throw new Error('Invalid PS2 contact interval/identity');
  }
  return {...data,frames:[...data.frames].sort((a,b)=>a.t_ms-b.t_ms)};
}
