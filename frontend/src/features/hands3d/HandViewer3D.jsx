import React,{useMemo,useRef,useEffect} from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Hand, RotateCcw, Move3d } from 'lucide-react';
import * as THREE from 'three';
import { Panel } from '@/components/shared/Panel';
import { usePlaybackStore } from '@/stores/usePlaybackStore';
import { HAND_CONNECTIONS } from '@/mocks/demoData';
import {clip3DHands} from '@/services/clipData';
import {useInferenceStore} from '@/stores/useInferenceStore';
function Bones({joints,color}){
  const geometry=useMemo(()=>{
    const vertices=[];
    for(const [i,j] of HAND_CONNECTIONS){vertices.push(...joints[i].map(x=>x*9));vertices.push(...joints[j].map(x=>x*9));}
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));return geo;
  },[joints]);
  useEffect(()=>()=>geometry.dispose(),[geometry]);
  return <group rotation={[-Math.PI/7,0,0]}><lineSegments geometry={geometry}><lineBasicMaterial color={color}/></lineSegments>{joints.map((p,i)=><mesh key={i} position={p.map(v=>v*9)}><sphereGeometry args={[.028,10,10]}/><meshStandardMaterial color={i===0?'#e2f6ff':color} emissive={color} emissiveIntensity={.13}/></mesh>)}</group>;
}
export function HandViewer3D({clip}){
  const time=usePlaybackStore(s=>s.currentTimeMs),handId=usePlaybackStore(s=>s.selectedHandId),setHand=usePlaybackStore(s=>s.setHand),inference=useInferenceStore(s=>s.byClip[clip?.id]);
  const trackList=useMemo(()=>{
    if(clip?.type==='demo')return [{hand_id:'H0',side:'right'},{hand_id:'H1',side:'left'}];
    const frames=inference?.modules.ps4?.frames||inference?.modules.ps1||[];
    const tracks=new Map();
    for(const frame of frames){for(const hand of frame.hands||[]){if(!tracks.has(hand.hand_id))tracks.set(hand.hand_id,{hand_id:hand.hand_id,side:hand.side});}}
    return [...tracks.values()];
  },[clip?.id,clip?.type,inference]);
  const hands=clip3DHands(clip,inference,time);
  const actualId=trackList.some(t=>t.hand_id===handId)?handId:trackList[0]?.hand_id;
  const hand=hands.find(h=>h.hand_id===actualId);
  const controlsRef=useRef(null);
  return <Panel title="3D Hand Viewer" className="h-full" action={<select className="rounded-md border border-slate-200 bg-white px-2 py-1.5 text-[11px] font-medium text-slate-700" value={actualId||''} onChange={e=>setHand(e.target.value)} aria-label="Select hand" disabled={!trackList.length}>{trackList.length?trackList.map(h=><option value={h.hand_id} key={h.hand_id}>{h.side} · {h.hand_id}</option>):<option value="">No tracks</option>}</select>} bodyClassName="p-3">
    <div className="relative aspect-[1/1] min-h-[220px] max-h-[370px] overflow-hidden rounded-lg border border-slate-100 bg-gradient-to-b from-[#f7f9fc] to-[#e7edf4]">
      {hand?.kp3d_hand_m?<Canvas camera={{position:[1.1,1,1.8],fov:42}}><ambientLight intensity={1.7}/><directionalLight position={[1,3,4]} intensity={1.8}/><gridHelper args={[3,12,'#adbfd4','#d4dfeb']} position={[0,-.55,0]}/><axesHelper args={[.33]}/><Bones joints={hand.kp3d_hand_m} color={hand.side==='right'?'#10a878':'#ed9b43'}/><OrbitControls ref={controlsRef} enableDamping target={[0,.18,0]} minDistance={.8} maxDistance={4}/></Canvas>:<div className="flex h-full flex-col items-center justify-center gap-2 text-slate-500"><Hand size={30}/><p className="text-sm font-semibold">3D pose unavailable</p><p className="px-8 text-center text-xs">{clip?.type==='demo'?'No hand observed during this interval.':inference?'No valid 3D pose at this time.':'Awaiting PS1 / PS4 data.'}</p></div>}
      {hand?.kp3d_hand_m&&<button onClick={()=>controlsRef.current?.reset()} className="absolute bottom-3 right-3 rounded-md border border-slate-200 bg-white/90 p-2 text-slate-500 hover:text-blue-600" title="Reset 3D view"><RotateCcw size={16}/></button>}
      <span className="absolute left-3 top-3 text-[10px] font-bold text-slate-400">HAND-CENTERED · METERS</span>
    </div><div className="mt-3 flex items-center justify-between text-xs text-slate-500"><span className="flex items-center gap-1"><Move3d size={14}/> Drag to rotate · Scroll to zoom</span><span className="font-semibold text-slate-700">{hand?'21 joints':'0 joints'}</span></div><p className="mt-3 text-[11px] leading-5 text-slate-400">3D landmarks are relative to the hand—not global factory coordinates.</p>
  </Panel>;
}
