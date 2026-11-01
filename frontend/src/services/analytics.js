// Summaries describe observations and annotations, not model accuracy or productivity.
export function coveredMilliseconds(actions,durationMs){
  const limit=Number.isFinite(durationMs)&&durationMs>0?durationMs:Infinity;
  const ranges=actions.map(a=>[Math.max(0,a.start_ms),Math.min(limit,a.end_ms)]).filter(([a,b])=>Number.isFinite(a)&&Number.isFinite(b)&&b>a).sort((a,b)=>a[0]-b[0]);
  let total=0,start=null,end=null;
  for(const [a,b] of ranges){if(start===null){start=a;end=b;}else if(a<=end)end=Math.max(end,b);else{total+=end-start;start=a;end=b;}}
  return total+(start===null?0:end-start);
}
export function summarizeActions(actions){
  const grouped=new Map();
  for(const a of actions){const name=String(a.verb||'unknown').trim().toLowerCase().replace(/[_-]+/g,' ')||'unknown';const item=grouped.get(name)||{name,count:0,durationMs:0};item.count++;if(Number.isFinite(a.start_ms)&&Number.isFinite(a.end_ms))item.durationMs+=Math.max(0,a.end_ms-a.start_ms);grouped.set(name,item);}
  return [...grouped.values()].sort((a,b)=>b.durationMs-a.durationMs||b.count-a.count||a.name.localeCompare(b.name));
}
export function summarizeContacts(contacts){
  const grouped=new Map();
  for(const c of contacts){const name=String(c.interaction||'unknown').replace(/[_-]+/g,' ');grouped.set(name,(grouped.get(name)||0)+1);}
  return [...grouped].map(([name,count])=>({name,count})).sort((a,b)=>b.count-a.count||a.name.localeCompare(b.name));
}
export function summarizeFrames(frames){
  let withHands=0,threeD=0,interpolated=0,hands=0;
  for(const frame of frames){const found=frame.hands||[];if(found.length)withHands++;if(found.some(h=>Array.isArray(h.kp3d_hand_m)&&h.kp3d_hand_m.length===21))threeD++;for(const hand of found){hands++;if(hand.source==='interpolated')interpolated++;}}
  return {frames:frames.length,withHands,threeD,interpolated,hands};
}
