import React,{useState} from 'react';
import {Pencil,RotateCcw,Check} from 'lucide-react';
import {Panel} from '@/components/shared/Panel';
import {Badge} from '@/components/ui/badge';
import {Button} from '@/components/ui/button';
import {Dialog,DialogContent,DialogTitle,DialogDescription} from '@/components/ui/dialog';
import {clipActions} from '@/services/clipData';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {useAnnotationStore} from '@/stores/useAnnotationStore';
import {usePlaybackStore} from '@/stores/usePlaybackStore';
import {formatTime} from '@/lib/utils';
export function ObservationInspector({clip}){
  const selectedId=useAnnotationStore(s=>s.selectedActionId),overrides=useAnnotationStore(s=>s.overrides),update=useAnnotationStore(s=>s.updateAction),undo=useAnnotationStore(s=>s.undo),history=useAnnotationStore(s=>s.history),time=usePlaybackStore(s=>s.currentTimeMs);
  const inference=useInferenceStore(s=>s.byClip[clip?.id]);const edits=overrides[clip?.id]||{};
  const actions=clipActions(clip,inference,edits);
  const active=actions.find(a=>a.start_ms<=time&&a.end_ms>time),event=active||actions.find(a=>a.action_id===selectedId);
  const [open,setOpen]=useState(false),[draft,setDraft]=useState(null),[validation,setValidation]=useState('');
  const edit=()=>{if(!event)return;setDraft({...event});setValidation('');setOpen(true);};
  const save=()=>{
    if(!draft||!clip)return;
    const start=Number(draft.start_ms),end=Number(draft.end_ms);
    if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<=start||(clip.durationMs>0&&end>clip.durationMs)){
      setValidation('Start/end must be integer milliseconds, ordered, and within the clip duration.');return;
    }
    update(clip.id,draft.action_id,{verb:draft.verb.toLowerCase().trim()||'unknown',object_label:draft.object_label?.trim()||null,start_ms:start,end_ms:end,hand:draft.hand});setOpen(false);
  };
  return <><Panel title="Selected Observation" action={<div className="flex gap-1"><button title="Undo last correction" disabled={!history.length} onClick={undo} className="rounded p-1.5 text-slate-400 enabled:hover:bg-slate-100 disabled:opacity-30"><RotateCcw size={15}/></button><button title="Edit annotation" disabled={!event} onClick={edit} className="rounded p-1.5 text-blue-600 enabled:hover:bg-blue-50 disabled:opacity-30"><Pencil size={15}/></button></div>}>
    {event?<div className="space-y-3"><div className="flex items-center justify-between"><Badge variant={edits[event.action_id]?'green':clip.type==='demo'?'amber':'gray'}>{edits[event.action_id]?'Manually corrected':clip.type==='demo'?'Synthetic example':'Imported prediction'}</Badge><span className="text-[11px] text-slate-400">{event.action_id}</span></div>
      <div className="grid grid-cols-[115px_1fr] gap-y-3 text-xs"><span className="text-slate-500">Detected object</span><b className="text-slate-800">{event.object_label||'Unknown'}</b><span className="text-slate-500">Hand side</span><b className="capitalize text-slate-800">{event.hand}</b><span className="text-slate-500">Interaction / action</span><b className="capitalize text-slate-800">{event.verb}</b><span className="text-slate-500">Timestamp</span><b className="text-slate-800">{formatTime(event.start_ms,true)} – {formatTime(event.end_ms,true)}</b><span className="text-slate-500">Confidence</span><b className="text-slate-800">{event.confidence==null?'Not available':`${(event.confidence*100).toFixed(0)}%${clip.type==='demo'?' (mock)':''}`}</b></div>
      <Button variant="outline" className="w-full" size="sm" onClick={edit}><Pencil size={14}/> Edit annotation</Button></div>:<p className="py-8 text-center text-xs text-slate-400">No action prediction available for this clip.</p>}
  </Panel><Dialog open={open} onOpenChange={setOpen}><DialogContent><DialogTitle className="text-lg font-extrabold">Correct annotation</DialogTitle><DialogDescription className="mt-1 text-sm text-slate-500">Corrections are stored in this browser session; the source prediction is preserved.</DialogDescription>{draft&&<div className="mt-6 grid gap-4"><label className="text-xs font-bold text-slate-600">Action label<input className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm" value={draft.verb} onChange={e=>setDraft(s=>({...s,verb:e.target.value}))}/></label><label className="text-xs font-bold text-slate-600">Object label<input className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm" value={draft.object_label||''} onChange={e=>setDraft(s=>({...s,object_label:e.target.value}))}/></label><label className="text-xs font-bold text-slate-600">Hand side<select className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm" value={draft.hand} onChange={e=>setDraft(s=>({...s,hand:e.target.value}))}>{['left','right','both','unknown'].map(v=><option key={v} value={v}>{v}</option>)}</select></label><div className="grid grid-cols-2 gap-3">{[['start_ms','Start (ms)'],['end_ms','End (ms)']].map(([key,label])=><label className="text-xs font-bold text-slate-600" key={key}>{label}<input type="number" className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm" value={draft[key]} onChange={e=>setDraft(s=>({...s,[key]:Number(e.target.value)}))}/></label>)}</div>{validation&&<p className="text-xs text-rose-700">{validation}</p>}<div className="flex justify-end gap-2"><Button variant="outline" onClick={()=>setOpen(false)}>Cancel</Button><Button onClick={save}><Check size={15}/> Save correction</Button></div></div>}</DialogContent></Dialog></>;
}
