import * as Dialog from '@radix-ui/react-dialog';
import {LoaderCircle,CheckCircle2,Clock3,AlertCircle} from 'lucide-react';
import {useMediaStore} from '@/stores/useMediaStore';

const ACTIVE=new Set(['uploading','queued','processing','importing']);
const STAGES=[['ps1','Hand tracking'],['ps4','3D hand pose'],['ps2','Objects and contact'],['ps3','Task timeline']];
const STATUS_LABELS={queued:'Waiting',processing:'Processing',completed:'Complete',partial:'Partial result',failed:'Failed',skipped:'Skipped',unavailable:'Unavailable'};

export function ProcessingDialog(){
  const clips=useMediaStore(s=>s.clips);
  const active=clips.filter(clip=>!clip.synthetic&&ACTIVE.has(clip.status));
  if(!active.length)return null;
  return <Dialog.Root open>
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-slate-950/65 backdrop-blur-[3px]"/>
      <Dialog.Content onEscapeKeyDown={event=>event.preventDefault()} onPointerDownOutside={event=>event.preventDefault()} className="fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[calc(100vw-30px)] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl" style={{color:'#1e293b'}}>
        <div className="flex items-start gap-3"><span className="rounded-xl bg-blue-50 p-3 text-blue-600"><LoaderCircle size={23} className="animate-spin"/></span><div><Dialog.Title className="text-lg font-bold">Processing your media</Dialog.Title><Dialog.Description className="mt-1 text-sm text-slate-500">This closes automatically when processing and result loading finish. You can leave this tab open.</Dialog.Description></div></div>
        <div className="mt-5 space-y-4" aria-live="polite" aria-atomic="false">{active.map(clip=><section key={clip.id} className="rounded-xl border border-slate-200 p-4">
          <div className="flex flex-wrap items-start justify-between gap-2"><h3 className="max-w-[75%] break-all text-sm font-semibold">{clip.name}</h3><span className="text-xs font-medium text-blue-700">{clip.status==='uploading'?'Uploading':clip.status==='importing'?'Loading results':clip.status==='queued'?'Waiting':'Processing'}</span></div>
          {clip.type==='video'&&<div className="mt-3 grid gap-2 sm:grid-cols-2">{STAGES.map(([key,label])=>{const state=clip.moduleStatus?.[key]||'queued';const Icon=state==='completed'?CheckCircle2:state==='processing'?LoaderCircle:['failed','partial','unavailable'].includes(state)?AlertCircle:Clock3;return <div key={key} className="flex items-center gap-2 text-xs text-slate-600"><Icon size={15} className={state==='processing'?'animate-spin text-blue-600':state==='completed'?'text-emerald-600':['failed','partial','unavailable'].includes(state)?'text-amber-600':'text-slate-400'}/><span>{label}</span><span className="ml-auto text-slate-500">{STATUS_LABELS[state]||state}</span></div>;})}</div>}
          {clip.status==='importing'&&<p className="mt-3 text-xs text-slate-500">Preparing completed results for the viewer…</p>}
          {!!clip.issues?.length&&<p className="mt-3 text-xs text-amber-700">{clip.issues.join(' · ')}</p>}
        </section>)}</div>
        <p className="mt-4 text-xs text-slate-500">Long videos can take several minutes. If a stage fails, its status will be available in the media library.</p>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}
