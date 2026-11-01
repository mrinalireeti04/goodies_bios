import React,{useEffect,useRef} from 'react';
import { Maximize2, Pause, Play, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AnnotationCanvas } from './AnnotationCanvas';
import {clipFrame} from '@/services/clipData';
import {closestFrameIndex} from '@/services/annotationData';
import {usePlaybackTime} from '@/stores/usePlaybackTime';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {useObjectDraftStore} from '@/stores/useObjectDraftStore';
import { usePlaybackStore } from '@/stores/usePlaybackStore';
import { useMediaStore } from '@/stores/useMediaStore';
import { useAnnotationStore } from '@/stores/useAnnotationStore';
import { formatTime } from '@/lib/utils';
import { mediaUrl } from '@/services/apiClient';
export function VideoWorkspace({clip}){
  const mediaRef=useRef(null),overlayRef=useRef(null),wrapperRef=useRef(null);
  const time=usePlaybackTime(),isPlaying=usePlaybackStore(s=>s.isPlaying),setPlaying=usePlaybackStore(s=>s.setPlaying),setTime=usePlaybackStore(s=>s.setTime),rate=usePlaybackStore(s=>s.playbackRate),setRate=usePlaybackStore(s=>s.setRate),seek=usePlaybackStore(s=>s.seekTo),seekRequest=usePlaybackStore(s=>s.seekRequest),overlays=usePlaybackStore(s=>s.overlays),toggleOverlay=usePlaybackStore(s=>s.toggleOverlay);
  const updateClip=useMediaStore(s=>s.updateClip),inference=useInferenceStore(s=>s.byClip[clip?.id]);
  const frames=inference?.modules?.ps1||[];
  const activeFrameIndex=closestFrameIndex(frames,time);
  const activeFrame=activeFrameIndex>=0?frames[activeFrameIndex]:null;
  const stepFrame=(direction)=>{if(!frames.length)return;const index=Math.max(0,Math.min(frames.length-1,(activeFrameIndex<0?0:activeFrameIndex)+direction));seek(frames[index].t_ms);};
  const isDemo=clip?.type==='demo',isVideo=clip?.type==='video', duration=clip?.durationMs||0;
  const moduleStatus=clip?.moduleStatus||{};
  const overlayMessage=clip?.error|| (clip?.status==='failed'?clip.issues?.join(' · '):null) ||
    (moduleStatus.ps1==='completed' ? (moduleStatus.ps2==='processing' ? 'PS1 complete · processing PS2 object/contact analysis…' : moduleStatus.ps3==='processing' ? 'PS1 complete · processing PS3 task timeline…' : 'Overlay is being prepared…') : 'Processing hand detection…');
  useEffect(()=>{if(!isDemo||!isPlaying)return;let raf,last=performance.now(),lastSent=0;const tick=(now)=>{const dt=now-last;last=now;const next=Math.min(84000,usePlaybackStore.getState().currentTimeMs+dt*usePlaybackStore.getState().playbackRate);if(now-lastSent>50){setTime(next);lastSent=now;}if(next>=84000){setTime(84000);setPlaying(false);return;}raf=requestAnimationFrame(tick);};raf=requestAnimationFrame(tick);return ()=>cancelAnimationFrame(raf);},[isDemo,isPlaying,setTime,setPlaying]);
  useEffect(()=>{if(!isVideo||!Number.isFinite(seekRequest.t))return;const desired=seekRequest.t/1000;for(const video of [mediaRef.current,overlayRef.current])if(video&&Math.abs(video.currentTime-desired)>.1)video.currentTime=desired;},[seekRequest.nonce,isVideo,clip?.overlayUrl]);
  useEffect(()=>{if(isVideo && mediaRef.current)mediaRef.current.playbackRate=rate;},[rate,isVideo]);
  // Frame-accurate timing on browsers supporting requestVideoFrameCallback.
  useEffect(()=>{
    if(!isVideo||!mediaRef.current)return;
    const video=mediaRef.current;
    if(typeof video.requestVideoFrameCallback!=='function')return;
    let active=true,handle=0,lastReported=0,lastSync=0;
    const onFrame=(now,metadata)=>{
      if(!active)return;
      if(now-lastReported>=30){setTime(metadata.mediaTime*1000);lastReported=now;}
      if(now-lastSync>=120){const overlay=overlayRef.current;if(overlay&&Math.abs(overlay.currentTime-metadata.mediaTime)>.08)overlay.currentTime=metadata.mediaTime;lastSync=now;}
      handle=video.requestVideoFrameCallback(onFrame);
    };
    handle=video.requestVideoFrameCallback(onFrame);
    return ()=>{active=false;if(typeof video.cancelVideoFrameCallback==='function')video.cancelVideoFrameCallback(handle);};
  },[isVideo,clip?.id,setTime]);

  useEffect(()=>{if(!isVideo)return;const original=mediaRef.current,overlay=overlayRef.current;if(original){if(isPlaying){const p=original.play();p?.catch(()=>setPlaying(false));}else original.pause();}if(overlay){overlay.playbackRate=rate;if(isPlaying)overlay.play().catch(()=>{});else overlay.pause();}},[isPlaying,rate,isVideo,clip?.id,clip?.overlayUrl,setPlaying]);
  const drawingClipId=useObjectDraftStore(s=>s.drawingClipId),draftBox=useObjectDraftStore(s=>s.draft),startDraw=useObjectDraftStore(s=>s.start),finishDraw=useObjectDraftStore(s=>s.finish),cancelDraw=useObjectDraftStore(s=>s.cancel);
  const objectEdits=useAnnotationStore(s=>s.objectOverrides[clip?.id]),manualObjects=useAnnotationStore(s=>s.manualObjects[clip?.id]);
  const frame=clipFrame(clip,inference,time,objectEdits,manualObjects);
  const hasImported=!!inference,sourceUrl=clip?.originalUrl||clip?.url||(clip?.jobId?mediaUrl(clip.jobId):null),overlayUrl=clip?.overlayUrl||(clip?.jobId&&moduleStatus.ps1==='completed'?mediaUrl(clip.jobId,'overlay'):null);
  return <div className="panel overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3"><div className="flex items-center gap-2"><h2 className="studio-panel-heading">Video Viewer & Annotations</h2><Badge variant={isDemo?'amber':'gray'}>{isDemo?'Synthetic demo':'Local preview'}</Badge></div><div className="flex items-center gap-2"><span className="max-w-45 truncate text-[11px] text-slate-400">{clip?.name||'No selection'}</span>{clip?.moduleStatus?.ps3&&<span role="status" className="rounded border border-slate-200 px-2 py-1 text-[10px] text-slate-500">Task timeline: {clip.moduleStatus.ps3}</span>}</div></div>
    <div ref={wrapperRef} className="bg-[#152334]">
    <div className="relative aspect-video w-full overflow-hidden">
      {!clip?<div className="flex h-full items-center justify-center text-sm text-slate-300">Choose an image or video to begin</div>:isDemo?<img src={clip.url} className="h-full w-full object-contain" alt="Synthetic diagram of an assembly workstation"/>:isVideo?(sourceUrl?<video key={clip.id} ref={mediaRef} src={sourceUrl} playsInline preload="metadata" className="h-full w-full object-contain" onTimeUpdate={e=>{if(typeof e.currentTarget.requestVideoFrameCallback!=='function')setTime(e.currentTarget.currentTime*1000);}} onLoadedMetadata={e=>updateClip(clip.id,{durationMs:e.currentTarget.duration*1000,width:e.currentTarget.videoWidth,height:e.currentTarget.videoHeight})} onEnded={()=>setPlaying(false)} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} />:<div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-sm text-slate-300"><span>This browser-only file is no longer attached after refresh.</span><span className="text-xs text-slate-400">Select it again from the upload dialog to restore the preview.</span></div>):(sourceUrl?<img key={clip.id} src={sourceUrl} alt={clip.name} className="h-full w-full object-contain" onLoad={e=>updateClip(clip.id,{width:e.currentTarget.naturalWidth,height:e.currentTarget.naturalHeight})}/>:<div className="flex h-full items-center justify-center p-6 text-center text-sm text-slate-300">Select this image again to restore its browser preview.</div>)}
      {frame&&<AnnotationCanvas frame={frame} sourceWidth={frame.frame_width||clip.width||1280} sourceHeight={frame.frame_height||clip.height||720} drawing={drawingClipId===clip.id} draftBox={draftBox?.clipId===clip.id&&draftBox.frameIndex===activeFrame?.frame_idx?draftBox.bbox:null} onDraw={box=>finishDraw(clip.id,box,activeFrame)}/>}
      {isDemo&&<div className="absolute left-4 top-4 rounded-md border border-amber-300/30 bg-slate-950/75 px-2 py-1 text-[11px] font-semibold text-amber-100">SIMULATED ANNOTATIONS</div>}
      {isVideo&&<span className="absolute left-3 top-3 rounded bg-slate-950/75 px-2 py-1 text-xs text-white">{frame?'Combined overlay':'Original'}</span>}
      {clip&&<button aria-label="Toggle fullscreen" title="Fullscreen" onClick={()=>wrapperRef.current?.requestFullscreen?.()} className="absolute right-3 top-3 rounded-md bg-slate-900/70 p-2 text-white hover:bg-slate-800"><Maximize2 size={16}/></button>}
    </div>
    </div>
    {clip&&(isDemo||isVideo)&&<div className="flex flex-wrap items-center gap-3 bg-[#1d2939] px-4 py-3 text-white"><button onClick={()=>setPlaying(!isPlaying)} aria-label={isPlaying?'Pause':'Play'} className="text-white">{isPlaying?<Pause fill="currentColor" size={19}/>:<Play fill="currentColor" size={19}/>}</button><span className="w-[94px] shrink-0 text-[11px] tabular-nums text-slate-200">{formatTime(time)} / {formatTime(duration)}</span><input aria-label="Seek video" type="range" min="0" max={Math.max(1,duration)} step="1" value={Math.min(time,Math.max(1,duration))} onChange={e=>seek(Number(e.target.value))} className="h-1 min-w-0 flex-1 cursor-pointer"/>{isVideo&&frames.length>0&&<div className="flex items-center gap-1 text-[10px] text-slate-300"><button type="button" aria-label="Previous analyzed frame" title="Previous analyzed frame" disabled={activeFrameIndex===0} onClick={()=>stepFrame(-1)} className="rounded p-1 hover:bg-slate-700 disabled:opacity-40"><ChevronLeft size={15}/></button><span className="whitespace-nowrap tabular-nums">Frame {activeFrame?activeFrameIndex+1:0} / {frames.length}</span><button type="button" aria-label="Next analyzed frame" title="Next analyzed frame" disabled={activeFrameIndex===frames.length-1} onClick={()=>stepFrame(1)} className="rounded p-1 hover:bg-slate-700 disabled:opacity-40"><ChevronRight size={15}/></button></div>}<select aria-label="Playback speed" className="rounded bg-[#28374a] px-1 py-1 text-xs text-slate-100" value={rate} onChange={e=>setRate(Number(e.target.value))}>{[.5,1,1.5,2].map(v=><option key={v} value={v}>{v}x</option>)}</select></div>}
    <div className="flex flex-wrap items-center gap-3 border-t border-slate-100 px-4 py-3 text-xs text-slate-600"><span className="mr-1 font-bold text-slate-700">Overlays</span>{[['hands','Hands'],['objects','Objects'],['masks','Masks'],['labels','Labels'],['connections','Skeleton']].map(([key,label])=><button key={key} onClick={()=>toggleOverlay(key)} className={`rounded-md border px-2 py-1.5 font-medium ${(overlays[key]??(key==='masks'))?'border-blue-200 bg-blue-50 text-blue-700':'border-slate-200 text-slate-400'}`}>{label}</button>)}{isVideo&&activeFrame&&<button type="button" onClick={()=>{if(drawingClipId===clip.id)cancelDraw();else{setPlaying(false);startDraw(clip.id);wrapperRef.current?.scrollIntoView({behavior:'smooth',block:'center'});}}} className="rounded-md border border-blue-200 px-2 py-1.5 font-medium text-blue-700">{drawingClipId===clip.id?'Cancel drawing':'Draw object box'}</button>}{drawingClipId===clip?.id&&<span role="status" className="text-[11px] text-blue-700">Drag across the object, then add a label in Review annotations</span>}<span className="ml-auto text-[11px] text-slate-400">{isDemo?'Source 1280 × 720 · Mock':hasImported?'Imported PS annotations':'No inference available'}</span></div>
  </div>;
}
