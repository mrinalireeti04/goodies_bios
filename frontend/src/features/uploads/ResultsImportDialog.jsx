import React,{useRef,useState} from 'react';
import {Upload,FileJson,CheckCircle2,AlertTriangle} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {Dialog,DialogContent,DialogDescription,DialogTitle} from '@/components/ui/dialog';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {parseResultFile,mergeResultFiles} from '@/services/annotationData';
export function ResultsImportDialog({clip,open,onOpenChange}){
  const [errors,setErrors]=useState([]),[success,setSuccess]=useState(''),[busy,setBusy]=useState(false);
  const ref=useRef(null),importForClip=useInferenceStore(s=>s.importForClip),existing=useInferenceStore(s=>clip&&s.byClip[clip.id]);
  async function ingest(files){
    if(!clip||clip.type==='demo')return;
    setBusy(true);setErrors([]);setSuccess('');
    try{
      const parsed=await Promise.all(Array.from(files).map(async f=>parseResultFile(f.name,await f.text())));
      mergeResultFiles(existing,parsed); // Validate atomically before updating the store.
      importForClip(clip.id,parsed);
      setSuccess(`${parsed.map(p=>p.module.toUpperCase()).join(', ')} imported for ${clip.name}.`);
    }catch(e){setErrors([e.message]);}finally{setBusy(false);if(ref.current)ref.current.value='';}
  }
  return <Dialog open={open} onOpenChange={onOpenChange}><DialogContent>
    <DialogTitle>Import PS1–PS4 model results</DialogTitle>
    <DialogDescription className="mt-2 text-sm text-slate-500">Attach your teammates' outputs to <b>{clip?.name||'the selected file'}</b>. Nothing is sent to a server.</DialogDescription>
    <div className="mt-5 rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm text-blue-900"><p className="font-semibold">Accepted files:</p><p className="mt-2 leading-6">hands.jsonl (PS1), ps2.json (contacts), actions.json (PS3), ps4.json (3D). You may import all four together or add modules as they finish.</p><p className="mt-2 text-xs">Every imported file must have the same source <code>clip_id</code>. It is associated with the selected media in this browser session.</p></div>
    <input ref={ref} type="file" multiple accept=".json,.jsonl" className="mt-5 block w-full text-sm" onChange={e=>ingest(e.target.files)} aria-label="Select PS output files" disabled={!clip||clip.type==='demo'||busy}/>
    {errors.map(err=><p key={err} className="mt-3 flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-700"><AlertTriangle size={17} className="shrink-0"/>{err}</p>)}
    {success&&<p className="mt-3 flex items-center gap-2 rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800"><CheckCircle2 size={17}/>{success}</p>}
    {!!existing&&<p className="mt-3 text-xs text-slate-500">Currently loaded: {Object.keys(existing.modules).map(m=>m.toUpperCase()).join(' · ')}</p>}
    <div className="mt-5 flex justify-end"><Button variant="outline" onClick={()=>onOpenChange(false)}>Close</Button></div>
  </DialogContent></Dialog>;
}
