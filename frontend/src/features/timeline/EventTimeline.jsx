import {useState} from 'react';
import {useAnnotationStore} from '@/stores/useAnnotationStore';
import {usePlaybackStore} from '@/stores/usePlaybackStore';
import {usePlaybackTime} from '@/stores/usePlaybackTime';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {clipActions} from '@/services/clipData';
import {formatTime} from '@/lib/utils';

const eventColors={pick:'#4b92f1','pick up':'#4b92f1',pick_up:'#4b92f1',hold:'#48b988',fasten:'#e4aa4d',inspect:'#9d78e6',unknown:'#94a3b8',place:'#43a7ba',release:'#f28b82',move:'#5988b6',attach:'#b98d53',test:'#9d78e6','route wire':'#6486a2',route_wire:'#6486a2','take out':'#ad8194',take_out:'#ad8194'};
const labelFor=event=>String(event.verb||'unknown').replace(/[_-]+/g,' ');

export function EventTimeline({clip}){
  const [preview,setPreview]=useState(null);
  const time=usePlaybackTime(),seek=usePlaybackStore(s=>s.seekTo),setPlaying=usePlaybackStore(s=>s.setPlaying);
  const overrides=useAnnotationStore(s=>s.overrides),select=useAnnotationStore(s=>s.selectAction),selected=useAnnotationStore(s=>s.selectedActionId);
  const inference=useInferenceStore(s=>s.byClip[clip?.id]);
  const added=useAnnotationStore(s=>s.addedActions[clip?.id]);
  const actions=clipActions(clip,inference,overrides[clip?.id]||{},added);
  const duration=Math.max(1,clip?.durationMs||0,...actions.map(a=>a.end_ms||0));
  const open=event=>{select(event.action_id);seek(event.start_ms);setPlaying(false);};
  const active=actions.find(a=>a.action_id===selected);
  return <section className="panel overflow-hidden">
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-slate-100 px-4 py-3">
      <div><h2 className="studio-panel-heading">Event Timeline</h2><p className="mt-0.5 text-[11px] text-slate-500">{actions.length?`${actions.length} events · Select an event to jump to its start`:'Actions will appear here after analysis'}</p></div>
      <span className="text-xs tabular-nums text-slate-500">{formatTime(0)} – {formatTime(duration)}</span>
    </div>
    <div className="space-y-4 p-4">{actions.length?<>
      <div className="relative">
        {preview&&<div role="tooltip" className="pointer-events-none absolute z-30 whitespace-nowrap rounded-md bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-white shadow-lg" style={{bottom:'calc(100% + 8px)',left:`${preview.position}%`,transform:'translateX(-50%)'}}><span className="capitalize">{preview.label}</span><span className="ml-2 text-slate-300">{preview.start} – {preview.end}</span></div>}
        <div className="relative h-10 overflow-hidden rounded-md bg-slate-100" aria-label="Interactive task timeline">
          {actions.map((event,index)=>{
            const left=Math.max(0,Math.min(100,event.start_ms/duration*100)),right=Math.max(left,Math.min(100,event.end_ms/duration*100)),width=right-left;
            const label=labelFor(event),description=`${label}, ${formatTime(event.start_ms)} to ${formatTime(event.end_ms)}`;
            const showPreview=()=>setPreview({label,start:formatTime(event.start_ms),end:formatTime(event.end_ms),position:Math.max(12,Math.min(88,(left+right)/2))});
            return <button type="button" key={`${event.action_id}-${index}`} onClick={()=>open(event)} onMouseEnter={showPreview} onMouseLeave={()=>setPreview(null)} onFocus={showPreview} onBlur={()=>setPreview(null)} aria-label={description} aria-pressed={selected===event.action_id} style={{position:'absolute',top:0,height:'100%',left:`${left}%`,width:`${Math.max(.3,width)}%`,backgroundColor:eventColors[event.verb]||'#76879a',fontSize:'11px',lineHeight:'1.2'}} className={`overflow-hidden border-r border-white/60 text-left font-semibold text-white transition hover:brightness-110 focus-visible:z-20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white ${selected===event.action_id?'z-10 ring-2 ring-inset ring-white':''}`}>
              <span className="block truncate px-1 capitalize">{label}</span>
            </button>;
          })}
          <span aria-hidden="true" className="pointer-events-none absolute z-20 w-0.5 bg-white" style={{top:0,height:'100%',left:`${Math.max(0,Math.min(99.8,time/duration*100))}%`}}/>
        </div>
        <div className="mt-2 flex justify-between text-[11px] tabular-nums text-slate-400">{[0,.25,.5,.75,1].map(i=><span key={i}>{formatTime(i*duration)}</span>)}</div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-slate-500">{active?<>Selected: <span className="font-semibold capitalize text-slate-700">{labelFor(active)}</span> · {formatTime(active.start_ms)} – {formatTime(active.end_ms)}</>:'Choose an event to inspect it.'}</p>
        <span className="text-[11px] text-slate-400">Hover or focus on a segment for details</span>
      </div>
    </>:<div className="flex min-h-16 items-center justify-center rounded-lg border border-dashed border-slate-200 p-3 text-center text-xs text-slate-400">No action timeline yet. Import actions.json (PS3) for this clip.</div>}</div>
  </section>;
}
