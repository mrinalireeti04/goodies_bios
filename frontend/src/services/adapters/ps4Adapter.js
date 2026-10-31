export function normalizePS4(data){
  if(!data||data.coordinate_frame!=='hand_centered'||typeof data.clip_id!=='string'||!data.clip_id||!Array.isArray(data.frames))throw new Error('Invalid PS4 coordinate frame or frames');
  for(const frame of data.frames){
    if(!Number.isInteger(frame.t_ms)||!Array.isArray(frame.hands))throw new Error('Invalid PS4 timestamp/hands');
    for(const hand of frame.hands){
      if(typeof hand.hand_id!=='string'||!hand.hand_id||!Array.isArray(hand.joints_3d_m)||hand.joints_3d_m.length!==21||!hand.joints_3d_m.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)))throw new Error('PS4 hand requires a track ID and exactly 21 3D joints');
    }
  }
  return {...data,frames:[...data.frames].sort((a,b)=>a.t_ms-b.t_ms)};
}
