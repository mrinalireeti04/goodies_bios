const finite=(x)=>typeof x==='number'&&Number.isFinite(x);
const box=(v)=>Array.isArray(v)&&v.length===4&&v.every(finite)&&v[2]>=v[0]&&v[3]>=v[1];
const score=(v)=>v==null||(finite(v)&&v>=0&&v<=1);
export function validatePS1Frame(frame){
  if(!frame||typeof frame.clip_id!=='string'||!frame.clip_id||!Number.isInteger(frame.t_ms)||frame.t_ms<0||!Number.isInteger(frame.frame_idx)||frame.frame_idx<0||!Array.isArray(frame.hands))throw new Error('Invalid PS1 frame metadata');
  if(!Number.isInteger(frame.frame_width)||frame.frame_width<1||!Number.isInteger(frame.frame_height)||frame.frame_height<1)throw new Error('Invalid PS1 frame dimensions');
  for(const hand of frame.hands){
    if(typeof hand.hand_id!=='string'||!hand.hand_id||!['left','right','unknown'].includes(hand.side))throw new Error('Invalid PS1 hand identity');
    if(!score(hand.handedness_score)||!box(hand.bbox_px))throw new Error('Invalid PS1 hand score or bbox');
    if(!Array.isArray(hand.kp2d_px)||hand.kp2d_px.length!==21||!hand.kp2d_px.every(p=>Array.isArray(p)&&p.length===2&&p.every(finite)))throw new Error('PS1 requires exactly 21 2D landmarks');
    if(hand.kp3d_hand_m!==null&&hand.kp3d_hand_m!==undefined&&(!Array.isArray(hand.kp3d_hand_m)||hand.kp3d_hand_m.length!==21||!hand.kp3d_hand_m.every(p=>Array.isArray(p)&&p.length===3&&p.every(finite))))throw new Error('PS1 requires 21 3D landmarks or null');
  }
  return frame;
}
export function normalizePS1Frames(records){
  const frames=records.map(validatePS1Frame).sort((a,b)=>a.t_ms-b.t_ms);
  for(let i=1;i<frames.length;i++)if(frames[i].t_ms<=frames[i-1].t_ms)throw new Error('PS1 timestamps must be unique and strictly increasing');
  return frames;
}
