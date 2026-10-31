import {useAnnotationStore} from '@/stores/useAnnotationStore';
import {usePlaybackStore} from '@/stores/usePlaybackStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {clipActions} from '@/services/clipData';
import {formatTime} from '@/lib/utils';
const eventColors={pick:'#4b92f1',hold:'#48b988',fasten:'#e4aa4d',inspect:'#9d78e6',unknown:'#94a3b8',place:'#43a7ba',release:'#f28b82'};
export function EventTimeline({clip}){
  const time=usePlaybackStore(s=>s.currentTimeMs),seek=usePlaybackStore(s=>s.seekTo),setPlaying=usePlaybackStore(s=>s.setPlaying),overrides=useAnnotationStore(s=>s.overrides),select=useAnnotationStore(s=>s.selectAction),selected=useAnnotationStore(s=>s.selectedActionId);
  const inference=useInferenceStore(s=>s.byClip[clip?.id]);
  const actions=clipActions(clip,inference,overrides[clip?.id]||{});
  const maxEventEnd=Math.max(0,...actions.map(a=>a.end_ms));
  const duration=Math.max(1,clip?.durationMs||0,maxEventEnd);
  return <section className="panel overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3"><h2 className="studio-panel-heading">Event Timeline</h2><div className="flex flex-wrap gap-3">{Object.entries(eventColors).slice(0,5).map(([name,color])=><span key={name} className="flex items-center gap-1 text-[11px] capitalize text-slate-500"><i className="h-2 w-2 rounded-full" style={{background:color}}/>{name}</span>)}</div><span className="text-[11px] text-slate-500">{formatTime(0)} – {formatTime(duration)}</span></div>
  <div className="p-4">{actions.length?<><div className="relative h-11 overflow-hidden rounded-lg bg-slate-100" aria-label="Interactive task timeline">
    {actions.map(a=>{
      const left=Math.max(0,Math.min(100,a.start_ms/duration*100)),right=Math.max(left,Math.min(100,a.end_ms/duration*100));
      return <button type="button" key={a.action_id} onClick={()=>{select(a.action_id);seek(a.start_ms);setPlaying(false)}} style={{position:'absolute',left:`${left}%`,width:`${Math.max(.3,right-left)}%`,height:'100%',background:eventColors[a.verb]||'#76879a'}} className={`overflow-hidden border-r border-white/70 px-1 text-[11px] font-bold capitalize text-white ${selected===a.action_id?'outline outline-2 outline-offset-[-3px] outline-slate-800':''}`} title={`${a.verb} ${formatTime(a.start_ms)} – ${formatTime(a.end_ms)}`}>{a.verb}</button>;
    })}<span className="pointer-events-none absolute top-0 h-full w-[2px] bg-slate-950" style={{left:`${Math.min(100,time/duration*100)}%`}}/>
  </div><div className="mt-2 flex justify-between text-[10px] text-slate-400">{[0,.25,.5,.75,1].map(i=><span key={i}>{formatTime(i*duration)}</span>)}</div></>:<div className="flex h-15 items-center justify-center rounded-lg border border-dashed border-slate-200 text-xs text-slate-400">No action timeline yet. Import actions.json (PS3) for this clip.</div>}</div></section>;
}
