import React,{useEffect,useRef} from 'react';
import { Maximize2, Pause, Play, Volume2, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AnnotationCanvas } from './AnnotationCanvas';
import {clipFrame} from '@/services/clipData';
import {useInferenceStore} from '@/stores/useInferenceStore';
import { usePlaybackStore } from '@/stores/usePlaybackStore';
import { useMediaStore } from '@/stores/useMediaStore';
import { formatTime } from '@/lib/utils';
export function VideoWorkspace({clip}){
  const mediaRef=useRef(null),overlayRef=useRef(null),wrapperRef=useRef(null);
  const time=usePlaybackStore(s=>s.currentTimeMs),isPlaying=usePlaybackStore(s=>s.isPlaying),setPlaying=usePlaybackStore(s=>s.setPlaying),setTime=usePlaybackStore(s=>s.setTime),rate=usePlaybackStore(s=>s.playbackRate),setRate=usePlaybackStore(s=>s.setRate),seek=usePlaybackStore(s=>s.seekTo),seekRequest=usePlaybackStore(s=>s.seekRequest),overlays=usePlaybackStore(s=>s.overlays),toggleOverlay=usePlaybackStore(s=>s.toggleOverlay);
  const updateClip=useMediaStore(s=>s.updateClip),inference=useInferenceStore(s=>s.byClip[clip?.id]);
  const isDemo=clip?.type==='demo',isVideo=clip?.type==='video', duration=clip?.durationMs||0;
  useEffect(()=>{if(!isDemo||!isPlaying)return;let raf,last=performance.now(),lastSent=0;const tick=(now)=>{const dt=now-last;last=now;const next=Math.min(84000,usePlaybackStore.getState().currentTimeMs+dt*usePlaybackStore.getState().playbackRate);if(now-lastSent>50){setTime(next);lastSent=now;}if(next>=84000){setTime(84000);setPlaying(false);return;}raf=requestAnimationFrame(tick);};raf=requestAnimationFrame(tick);return ()=>cancelAnimationFrame(raf);},[isDemo,isPlaying,setTime,setPlaying]);
  useEffect(()=>{if(isVideo && mediaRef.current && Number.isFinite(seekRequest.t)){const desired=seekRequest.t/1000;if(Math.abs(mediaRef.current.currentTime-desired)>.1)mediaRef.current.currentTime=desired;}},[seekRequest.nonce,isVideo]);
  useEffect(()=>{if(isVideo && mediaRef.current)mediaRef.current.playbackRate=rate;},[rate,isVideo]);
  // Frame-accurate timing on browsers supporting requestVideoFrameCallback.
  useEffect(()=>{
    if(!isVideo||!mediaRef.current)return;
    const video=mediaRef.current;
    if(typeof video.requestVideoFrameCallback!=='function')return;
    let active=true,handle=0,lastReported=0;
    const onFrame=(now,metadata)=>{
      if(!active)return;
      if(now-lastReported>=30){setTime(metadata.mediaTime*1000);lastReported=now;}
      handle=video.requestVideoFrameCallback(onFrame);
    };
    handle=video.requestVideoFrameCallback(onFrame);
    return ()=>{active=false;if(typeof video.cancelVideoFrameCallback==='function')video.cancelVideoFrameCallback(handle);};
  },[isVideo,clip?.id,setTime]);

  useEffect(()=>{if(isVideo&&mediaRef.current){if(isPlaying){const p=mediaRef.current.play();p?.catch(()=>setPlaying(false));}else mediaRef.current.pause();}},[isPlaying,isVideo,clip?.id,setPlaying]);
  const frame=clipFrame(clip,inference,time);
  useEffect(()=>{
    const overlay=overlayRef.current;
    const original=mediaRef.current;
    if(!overlay||!original)return;
    overlay.playbackRate=rate;
    if(Math.abs(overlay.currentTime-original.currentTime)>.06)overlay.currentTime=original.currentTime;
    if(isPlaying)overlay.play().catch(()=>{});else overlay.pause();
  },[time,rate,isPlaying,clip?.overlayUrl,seekRequest.nonce]);
  const hasImported=!!inference;
  return <div className="panel overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3"><div className="flex items-center gap-2"><h2 className="studio-panel-heading">Video Viewer & Annotations</h2><Badge variant={isDemo?'amber':'gray'}>{isDemo?'Synthetic demo':'Local preview'}</Badge></div><span className="max-w-45 truncate text-[11px] text-slate-400">{clip?.name||'No selection'}</span></div>
    <div ref={wrapperRef} className={isVideo?'grid gap-px bg-[#152334] md:grid-cols-2':'bg-[#152334]'}>
    <div className="relative aspect-video w-full overflow-hidden">
      {!clip?<div className="flex h-full items-center justify-center text-sm text-slate-300">Choose an image or video to begin</div>:isDemo?<img src={clip.url} className="h-full w-full object-contain" alt="Synthetic diagram of an assembly workstation"/>:isVideo?<video key={clip.id} ref={mediaRef} src={clip.originalUrl||clip.url} playsInline preload="metadata" className="h-full w-full object-contain" onTimeUpdate={e=>setTime(e.currentTarget.currentTime*1000)} onLoadedMetadata={e=>updateClip(clip.id,{durationMs:e.currentTarget.duration*1000,width:e.currentTarget.videoWidth,height:e.currentTarget.videoHeight})} onEnded={()=>setPlaying(false)} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} />:<img key={clip.id} src={clip.url} alt={clip.name} className="h-full w-full object-contain" onLoad={e=>updateClip(clip.id,{width:e.currentTarget.naturalWidth,height:e.currentTarget.naturalHeight})}/>}
      {frame&&!isVideo&&<AnnotationCanvas frame={frame} sourceWidth={frame.frame_width||clip.width||1280} sourceHeight={frame.frame_height||clip.height||720}/>}
      {isDemo&&<div className="absolute left-4 top-4 rounded-md border border-amber-300/30 bg-slate-950/75 px-2 py-1 text-[11px] font-semibold text-amber-100">SIMULATED ANNOTATIONS</div>}
      {isVideo&&<span className="absolute left-3 top-3 rounded bg-slate-950/75 px-2 py-1 text-xs text-white">Original</span>}
      {clip&&<button aria-label="Toggle fullscreen" title="Fullscreen" onClick={()=>wrapperRef.current?.requestFullscreen?.()} className="absolute right-3 top-3 rounded-md bg-slate-900/70 p-2 text-white hover:bg-slate-800"><Maximize2 size={16}/></button>}
    </div>
    {isVideo&&<div className="relative aspect-video overflow-hidden"><span className="absolute left-3 top-3 z-10 rounded bg-slate-950/75 px-2 py-1 text-xs text-white">Hand detection overlay</span>{clip.overlayUrl?<video ref={overlayRef} src={clip.overlayUrl} muted playsInline preload="auto" className="h-full w-full object-contain"/>:<div className="flex h-full items-center justify-center p-6 text-center text-sm text-slate-300">{clip.error|| (clip.status==='failed'?clip.issues?.join(' · '):'Processing hand detection…') || 'Awaiting results'}</div>}</div>}
    </div>
    {clip&&(isDemo||isVideo)&&<div className="flex items-center gap-3 bg-[#1d2939] px-4 py-3 text-white"><button onClick={()=>setPlaying(!isPlaying)} aria-label={isPlaying?'Pause':'Play'} className="text-white">{isPlaying?<Pause fill="currentColor" size={19}/>:<Play fill="currentColor" size={19}/>}</button><span className="w-[94px] shrink-0 text-[11px] tabular-nums text-slate-200">{formatTime(time)} / {formatTime(duration)}</span><input aria-label="Seek video" type="range" min="0" max={Math.max(1,duration)} step="100" value={Math.min(time,Math.max(1,duration))} onChange={e=>seek(Number(e.target.value))} className="h-1 min-w-0 flex-1 cursor-pointer"/><select aria-label="Playback speed" className="rounded bg-[#28374a] px-1 py-1 text-xs text-slate-100" value={rate} onChange={e=>setRate(Number(e.target.value))}>{[.5,1,1.5,2].map(v=><option key={v} value={v}>{v}x</option>)}</select></div>}
    <div className="flex flex-wrap items-center gap-3 border-t border-slate-100 px-4 py-3 text-xs text-slate-600"><span className="mr-1 font-bold text-slate-700">Overlays</span>{[['hands','Hands'],['objects','Objects'],['labels','Labels'],['connections','Skeleton']].map(([key,label])=><button key={key} onClick={()=>toggleOverlay(key)} className={`rounded-md border px-2 py-1.5 font-medium ${overlays[key]?'border-blue-200 bg-blue-50 text-blue-700':'border-slate-200 text-slate-400'}`}>{label}</button>)}<span className="ml-auto text-[11px] text-slate-400">{isDemo?'Source 1280 × 720 · Mock':hasImported?'Imported PS annotations':'No inference available'}</span></div>
  </div>;
}
