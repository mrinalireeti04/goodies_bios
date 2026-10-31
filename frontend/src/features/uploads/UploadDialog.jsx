import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, FileVideo, FileImage, ShieldCheck, AlertCircle } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useMediaStore } from '@/stores/useMediaStore';
import { usePlaybackStore } from '@/stores/usePlaybackStore';
import {useBackendProcessing} from './useBackendProcessing';
import {useRealApi} from '@/services/apiClient';
export function UploadDialog(){
  const open=useMediaStore(s=>s.uploadOpen),setOpen=useMediaStore(s=>s.setUploadOpen);
  const addFiles=useMediaStore(s=>s.addFiles),errors=useMediaStore(s=>s.errors),clearErrors=useMediaStore(s=>s.clearErrors);
  const inputRef=useRef(null),[dragging,setDragging]=useState(false),navigate=useNavigate(),resetPlayback=usePlaybackStore(s=>s.reset);
  const backend=useBackendProcessing();
  function ingest(files){const originals=Array.from(files||[]);const added=addFiles(originals);if(added.length){
      if(useRealApi){for(const clip of added){const file=originals.find(f=>f.name===clip.name&&f.size===clip.size);if(file)backend.mutate({clip,file});}}
      resetPlayback();setOpen(false);navigate('/studio');
    }}
  return <Dialog open={open} onOpenChange={v=>{setOpen(v);if(v)clearErrors();}}><DialogContent>
    <div className="mb-4 flex items-center gap-3"><div className="rounded-xl bg-blue-50 p-3 text-blue-600"><UploadCloud size={23}/></div><div><DialogTitle className="text-xl font-extrabold tracking-tight">Import factory media</DialogTitle><DialogDescription className="mt-1 text-sm text-slate-500">Add several images or videos to your private workspace.</DialogDescription></div></div>
    <input ref={inputRef} type="file" accept="image/png,image/jpeg,video/mp4,video/quicktime,video/webm,.mov" multiple className="hidden" onChange={e=>{ingest(e.target.files);e.target.value='';}}/>
    <div onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);ingest(e.dataTransfer.files)}} className={`flex min-h-52 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition-colors ${dragging?'border-blue-500 bg-blue-50':'border-slate-200 bg-slate-50 hover:border-blue-300 hover:bg-blue-50/50'}`} onClick={()=>inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inputRef.current?.click();}}}>
      <UploadCloud size={34} className="mb-3 text-blue-500"/><p className="font-bold text-slate-800">Drop files here or click to browse</p><p className="mt-2 text-sm text-slate-500">JPG, PNG, MP4, MOV, WebM · Max 800 MB per file</p>
      <div className="mt-4 flex gap-2 text-xs text-slate-500"><span className="flex items-center gap-1 rounded-md border bg-white px-2 py-1"><FileVideo size={13}/> Video</span><span className="flex items-center gap-1 rounded-md border bg-white px-2 py-1"><FileImage size={13}/> Image</span></div>
    </div>
    {!!errors.length&&<div className="mt-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{errors.map((e,i)=><p className="flex items-center gap-2" key={i}><AlertCircle size={14}/>{e}</p>)}</div>}
    <div className="mt-4 flex items-start gap-2 rounded-lg bg-emerald-50 px-3 py-3 text-xs text-emerald-800"><ShieldCheck size={17} className="shrink-0"/> {useRealApi?'Backend mode is enabled. Uploaded files will be processed by the LOCAL FastAPI service and stored in its private working directory. Do not expose the API publicly.':'Files stay in your browser during this frontend preview. Model processing is not connected yet; real footage is never packaged with the app.'}</div>
    <div className="mt-5 flex justify-end gap-2"><Button variant="outline" onClick={()=>setOpen(false)}>Cancel</Button><Button onClick={()=>inputRef.current?.click()}>Choose files</Button></div>
  </DialogContent></Dialog>;
}
