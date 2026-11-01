import {useMemo} from 'react';
import {Panel} from '@/components/shared/Panel';
import {usePlaybackTime} from '@/stores/usePlaybackTime';
import {usePlaybackStore} from '@/stores/usePlaybackStore';
import {closestFrame} from '@/services/annotationData';

const pct=(n,d)=>d?`${Math.round(n*100/d)}%`:'—';
const landmarkNames=['Wrist','Thumb CMC','Thumb MCP','Thumb IP','Thumb tip','Index MCP','Index PIP','Index DIP','Index tip','Middle MCP','Middle PIP','Middle DIP','Middle tip','Ring MCP','Ring PIP','Ring DIP','Ring tip','Pinky MCP','Pinky PIP','Pinky DIP','Pinky tip'];

export function FrameInsights({clip,inference}){
  const seek=usePlaybackStore(s=>s.seekTo),time=usePlaybackTime();
  const frames=inference?.modules?.ps1||[];
  const poseFrames=inference?.modules?.ps4?.frames||[];
  const frame=closestFrame(frames,time,Number.POSITIVE_INFINITY);
  const poseFrame=closestFrame(poseFrames,time);
  const metrics=useMemo(()=>{
    const observed=frames.length,withHands=frames.filter(f=>f.hands?.length),hands=frames.flatMap(f=>f.hands||[]);
    const interpolated=hands.filter(h=>h.source==='interpolated').length;
    const scored=hands.filter(h=>Number.isFinite(h.handedness_score));
    const actions=inference?.modules?.ps3?.actions||[];
    const actionScores=actions.filter(a=>Number.isFinite(a.confidence));
    const threeD=frames.filter(f=>{
      const pose=closestFrame(poseFrames,f.t_ms);
      return (f.hands||[]).some(h=>Array.isArray(h.kp3d_hand_m)&&h.kp3d_hand_m.length===21)||!!pose?.hands?.some(h=>Array.isArray(h.joints_3d_m)&&h.joints_3d_m.length===21);
    }).length;
    const unknown=hands.filter(h=>h.side==='unknown').length;
    return {observed,withHands:withHands.length,hands:hands.length,interpolated,scored,threeD,unknown,mean:scored.length?scored.reduce((n,h)=>n+h.handedness_score,0)/scored.length:null,actions,actionMean:actionScores.length?actionScores.reduce((n,a)=>n+a.confidence,0)/actionScores.length:null,contacts:inference?.modules?.ps2?.contacts?.length||0};
  },[frames,poseFrames,inference]);
  const status=clip?.moduleStatus||{};
  if(clip?.type==='demo')return <Panel title="Frame and video insights"><p className="text-xs text-slate-500">The bundled clip is synthetic. Upload a video to see per-frame model outputs and measured coverage.</p></Panel>;
  return <Panel title="Frame and video insights">
    {!frames.length?<div className="space-y-2 text-xs text-slate-500"><p>PS1 frame results are not available for this clip yet.</p><p className="text-[11px]">Module status: PS1 {status.ps1||'not started'} · PS4 {status.ps4||'not started'} · PS3 {status.ps3||'not started'}</p></div>:<div className="space-y-4">
      <section aria-label="Video coverage metrics" className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {[["Analyzed frames",metrics.observed],["Frames with hands",`${pct(metrics.withHands,metrics.observed)} (${metrics.withHands})`],["Frames without hands",metrics.observed-metrics.withHands],["Hand instances",metrics.hands],["3D coverage",`${pct(metrics.threeD,metrics.observed)} (${metrics.threeD})`],["Interpolated hands",metrics.interpolated],["Unknown side",`${pct(metrics.unknown,metrics.hands)} (${metrics.unknown})`],["Mean handedness confidence",metrics.mean==null?'Not provided':`${(metrics.mean*100).toFixed(1)}%`],["PS2 contact intervals",metrics.contacts],["PS3 actions",metrics.actions.length],["Mean PS3 confidence",metrics.actionMean==null?'Not provided':`${(metrics.actionMean*100).toFixed(1)}%`]].map(([label,value])=><div className="rounded-lg bg-slate-50 p-3" key={label}><p className="text-[10px] text-slate-500">{label}</p><p className="mt-1 text-sm font-bold text-slate-800">{value}</p></div>)}
      </section>
      <p className="text-[10px] leading-4 text-slate-500">These metrics reflect unedited model outputs. Coverage and model confidence are diagnostics, not accuracy scores. Accuracy requires labeled reference results. PS1={status.ps1||'unknown'} · PS2={status.ps2||'unknown'} · PS3={status.ps3||'unknown'} · PS4={status.ps4||'unknown'}</p>
      <section className="rounded-lg border border-slate-200 p-3">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-xs font-bold text-slate-800">Current frame</h3><p className="text-[11px] text-slate-500">{frame?`Frame ${frame.frame_idx+1} · ${frame.t_ms} ms · ${frame.hands.length} hand(s)`: 'No nearby PS1 frame result'}</p></div>{frame&&<div className="flex gap-1"><button className="rounded border border-slate-200 px-2 py-1 text-[11px]" onClick={()=>seek(frames[Math.max(0,frames.indexOf(frame)-1)]?.t_ms||0)} disabled={frames.indexOf(frame)<=0}>Previous</button><button className="rounded border border-slate-200 px-2 py-1 text-[11px]" onClick={()=>seek(frames[Math.min(frames.length-1,frames.indexOf(frame)+1)]?.t_ms||frame.t_ms)} disabled={frames.indexOf(frame)>=frames.length-1}>Next</button></div>}</div>
        {frame?.hands?.length?<div className="space-y-2">{frame.hands.map((hand,index)=>{const poseHand=poseFrame?.hands?.find(h=>h.hand_id===hand.hand_id);const kp3d=hand.kp3d_hand_m||poseHand?.joints_3d_m;const poseSource=hand.kp3d_hand_m?'PS1':poseHand?.joints_3d_m?'PS4':null;return <details key={`${hand.hand_id}-${index}`} className="rounded-md bg-slate-50 p-2"><summary className="cursor-pointer text-xs font-semibold text-slate-700">{hand.hand_id} · {hand.side} · {hand.source||'detected'} · confidence {hand.handedness_score==null?'n/a':`${(hand.handedness_score*100).toFixed(1)}%`} · {kp3d?.length===21?`3D from ${poseSource}`:'2D only'}</summary><div className="mt-2 grid gap-2 md:grid-cols-2"><div><p className="mb-1 text-[10px] font-bold text-slate-500">2D landmarks · source pixels</p><div className="max-h-48 overflow-auto rounded bg-white p-2 font-mono text-[10px]">{hand.kp2d_px.map((point,i)=><div key={i}>{i} {landmarkNames[i]}: [{point.map(v=>Number(v).toFixed(1)).join(', ')}]</div>)}</div></div><div><p className="mb-1 text-[10px] font-bold text-slate-500">3D landmarks · meters {kp3d?.length===21?`(${poseSource})`:'(unavailable)'}</p>{kp3d?.length===21?<div className="max-h-48 overflow-auto rounded bg-white p-2 font-mono text-[10px]">{kp3d.map((point,i)=><div key={i}>{i} {landmarkNames[i]}: [{point.map(v=>Number(v).toFixed(4)).join(', ')}]</div>)}</div>:<p className="rounded bg-white p-2 text-[10px] text-slate-500">No 21-point 3D pose for this hand in the current frame.</p>}</div></div></details>})}</div>:frame?<p className="text-xs text-slate-500">No hands were detected in this frame.</p>:null}
      </section>
    </div>}
  </Panel>;
}
