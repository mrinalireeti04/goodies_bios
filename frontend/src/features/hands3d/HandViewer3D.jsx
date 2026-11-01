import React,{useMemo,useRef,useEffect,useState,Suspense} from 'react';
import {RiggedHand} from './RiggedHand';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Hand, RotateCcw, Move3d } from 'lucide-react';
import * as THREE from 'three';
import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
import { Panel } from '@/components/shared/Panel';
import { usePlaybackTime } from '@/stores/usePlaybackTime';
import {clip3DHands,clipFrame} from '@/services/clipData';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {HAND_CONNECTIONS} from '@/mocks/demoData';
const PALM=[0,5,9,13,17];
const FINGERTIPS=[4,8,12,16,20];
const FINGER_SEGMENTS=[
  [0,1,.105],[1,2,.125],[2,3,.105],[3,4,.08],
  [5,6,.12],[6,7,.095],[7,8,.072],
  [9,10,.13],[10,11,.102],[11,12,.078],
  [13,14,.115],[14,15,.09],[15,16,.068],
  [17,18,.098],[18,19,.078],[19,20,.058],
];
const SKIN_MATERIAL=new THREE.MeshPhysicalMaterial({color:'#c58b6d',roughness:.56,metalness:0,clearcoat:.08,clearcoatRoughness:.62});
const NAIL_GEOMETRY=new THREE.SphereGeometry(1,12,8);
const NAIL_MATERIAL=new THREE.MeshStandardMaterial({color:'#ead0bd',roughness:.58,metalness:0});
function asVector(point){return new THREE.Vector3(point[0]*9,-point[1]*9,point[2]*9);}

const HandPoints=React.memo(function HandPoints({joints,side}){
  const points=useMemo(()=>{
    const vectors=joints.map(asVector),center=new THREE.Vector3();
    vectors.forEach(point=>center.add(point));
    center.divideScalar(vectors.length);
    return vectors.map(point=>point.sub(center));
  },[joints]);
  const segments=useMemo(()=>new Float32Array(HAND_CONNECTIONS.flatMap(([a,b])=>[...points[a].toArray(),...points[b].toArray()])),[points]);
  const color=side==='right'?'#059669':'#d97706';
  return <group><lineSegments renderOrder={2}><bufferGeometry><bufferAttribute attach="attributes-position" args={[segments,3]}/></bufferGeometry><lineBasicMaterial color={color} depthTest={false}/></lineSegments>{points.map((point,index)=><mesh key={index} position={point} renderOrder={3}><sphereGeometry args={[0.025,12,8]}/><meshBasicMaterial color={color} depthTest={false}/></mesh>)}</group>;
});

const Fingernail=React.memo(function Fingernail({point,previous,normal,palmWidth}){
  const axis=new THREE.Vector3().subVectors(point,previous).normalize();
  const across=new THREE.Vector3().crossVectors(axis,normal).normalize();
  const nailNormal=new THREE.Vector3().crossVectors(across,axis).normalize();
  const orientation=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(across,axis,nailNormal));
  const position=point.clone().addScaledVector(axis,-palmWidth*.035).addScaledVector(nailNormal,palmWidth*.045);
  return <mesh position={position} quaternion={orientation} scale={[palmWidth*.075,palmWidth*.11,palmWidth*.014]} geometry={NAIL_GEOMETRY} material={NAIL_MATERIAL}/>;
});

const HandMesh=React.memo(function HandMesh({joints}){
  const points=useMemo(()=>{
    const result=joints.map(asVector),center=new THREE.Vector3();
    for(const point of result)center.add(point);center.divideScalar(result.length);
    return result.map(point=>point.sub(center));
  },[joints]);
  const palmWidth=points[5].distanceTo(points[17]);
  const palmNormal=new THREE.Vector3().subVectors(points[5],points[0]).cross(new THREE.Vector3().subVectors(points[17],points[0])).normalize();
  if(palmNormal.z<0)palmNormal.negate();
  const surface=useMemo(()=>{
    const local=points;
    const extent=Math.max(...['x','y','z'].map(axis=>Math.max(...local.map(point=>point[axis]))-Math.min(...local.map(point=>point[axis]))));
    const scale=Math.max(.12,extent*.8),resolution=18,subtract=64,isolation=80;
    const field=new MarchingCubes(resolution,SKIN_MATERIAL,false,false,4000);
    field.isolation=isolation;
    const addBall=(point,radius,polarity=1,weight=.62)=>{
      const strength=Math.pow(Math.max(.004,radius/scale),2)*(subtract+isolation)*weight;
      field.addBall(.5+point.x/(2*scale),.5+point.y/(2*scale),.5+point.z/(2*scale),strength*polarity,subtract);
    };
    const addTube=(start,end,radius,polarity=1)=>{
      const distance=start.distanceTo(end),steps=Math.max(2,Math.ceil(distance/Math.max(radius*1.3,.025)));
      for(let step=0;step<=steps;step++)addBall(start.clone().lerp(end,step/steps),radius,polarity,.48);
    };

    const palmCenter=new THREE.Vector3();for(const index of PALM)palmCenter.add(local[index]);palmCenter.divideScalar(PALM.length);
    addBall(palmCenter,palmWidth*.34,1,.58);
    for(const index of PALM)addBall(local[index],palmWidth*(index===0?.15:.16),1,.52);
    const forearm=local[0].clone().sub(local[9].clone().sub(local[0]).normalize().multiplyScalar(palmWidth*.42));
    addTube(forearm,local[0],palmWidth*.14);
    for(const [a,b,width] of FINGER_SEGMENTS)addTube(local[a],local[b],palmWidth*width);
    for(const index of [5,9,13,17])addBall(local[index],palmWidth*.14,1,.58);
    const fingerRoots=[5,9,13,17];
    for(let finger=0;finger<fingerRoots.length-1;finger++){
      for(let joint=1;joint<4;joint++){
        const between=local[fingerRoots[finger]+joint].clone().add(local[fingerRoots[finger+1]+joint]).multiplyScalar(.5);
        addBall(between,palmWidth*(joint===1?.115:.095),-1,1.25);
      }
    }
    addBall(local[1].clone().add(local[5]).multiplyScalar(.5),palmWidth*.105,-1,1.15);
    field.update();field.scale.setScalar(scale);field.geometry.computeBoundingSphere();
    return field;
  },[points,palmWidth]);
  useEffect(()=>()=>{surface.geometry.dispose();},[surface]);
  return <group rotation={[-Math.PI/7,0,0]}><primitive object={surface} dispose={null}/>{FINGERTIPS.map(tip=><Fingernail key={tip} point={points[tip]} previous={points[tip-1]} normal={palmNormal} palmWidth={palmWidth}/>)}</group>;
});

function handPlacement(hand,anchor,frameWidth,frameHeight,displayScale){
  const landmarks=anchor?.kp2d_px;
  const landmarkCenter=landmarks?.length===21?[0,1].map(axis=>landmarks.reduce((sum,point)=>sum+point[axis],0)/landmarks.length):null;
  const box=anchor?.bbox_px;
  const center=landmarkCenter||[frameWidth/2,frameHeight/2];
  const position=[(center[0]/frameWidth-.5)*3.2,(.5-center[1]/frameHeight)*3.2*frameHeight/frameWidth,0];
  const joints=hand.joints_3d_m||hand.kp3d_hand_m;
  let poseSpan=0;
  for(let i=0;i<joints.length;i++)for(let j=i+1;j<joints.length;j++)poseSpan=Math.max(poseSpan,Math.hypot(joints[i][0]-joints[j][0],joints[i][1]-joints[j][1],joints[i][2]-joints[j][2])*9);
  const targetWidth=box?(box[2]-box[0])/frameWidth*3.2:.62;
  const scale=poseSpan>0?Math.max(.08,Math.min(3.5,targetWidth/poseSpan*displayScale)):.5;
  return {position,scale};
}
class RigBoundary extends React.Component {
  constructor(props){super(props);this.state={failed:false};}
  static getDerivedStateFromError(){return {failed:true};}
  componentDidCatch(error){console.error('Rigged hand viewer failed:',error);}
  render(){return this.state.failed?this.props.fallback:this.props.children;}
}
export function HandViewer3D({clip}){
  const time=usePlaybackTime(),inference=useInferenceStore(s=>s.byClip[clip?.id]);const [displayScale,setDisplayScale]=useState(2.2);
  const hands=clip3DHands(clip,inference,time),frame=clipFrame(clip,inference,time);
  const frameWidth=frame?.frame_width||clip?.width||1280,frameHeight=frame?.frame_height||clip?.height||720;
  const posedHands=hands.flatMap(hand=>{
    const joints=hand.joints_3d_m||hand.kp3d_hand_m;
    if(!Array.isArray(joints)||joints.length!==21)return [];
    const anchor=frame?.hands?.find(item=>item.hand_id===hand.hand_id);
    return [{...hand,joints, ...handPlacement(hand,anchor,frameWidth,frameHeight,displayScale)}];
  });
  const controlsRef=useRef(null);const [view,setView]=useState('rig');
  return <Panel title="3D Hand Viewer" className="h-full" action={<span className="text-[11px] font-semibold text-slate-500">{posedHands.length?`${posedHands.length} hand${posedHands.length===1?'':'s'} · video-relative`:'No 3D tracks'}</span>} bodyClassName="p-3">
    <div className="relative aspect-[1/1] min-h-[220px] max-h-[370px] overflow-hidden rounded-lg border border-slate-100 bg-gradient-to-b from-[#f7f9fc] to-[#e7edf4]">
      {posedHands.length?<Canvas frameloop="demand" dpr={1} gl={{antialias:false,powerPreference:'low-power'}} camera={{position:[0,0,5],fov:42}}><ambientLight intensity={1.35}/><directionalLight position={[2,3,5]} intensity={2}/><gridHelper args={[3.2,16,'#adbfd4','#d4dfeb']} rotation={[Math.PI/2,0,0]} position={[0,0,-.48]}/>{posedHands.map(hand=><group key={hand.hand_id} position={hand.position} scale={hand.scale}>{view==='rig'&&['left','right'].includes(hand.side)?<RigBoundary key={`${hand.hand_id}-${view}`} fallback={<HandPoints joints={hand.joints} side={hand.side}/>}><Suspense fallback={<HandPoints joints={hand.joints} side={hand.side}/>}><RiggedHand joints={hand.joints} side={hand.side}/></Suspense></RigBoundary>:<HandPoints joints={hand.joints} side={hand.side}/>}</group>)}<OrbitControls ref={controlsRef} enableDamping target={[0,0,0]} minDistance={2.6} maxDistance={8}/></Canvas>:<div className="flex h-full flex-col items-center justify-center gap-2 text-slate-500"><Hand size={30}/><p className="text-sm font-semibold">3D pose unavailable</p><p className="px-8 text-center text-xs">{clip?.type==='demo'?'No hand observed during this interval.':inference?'No valid 3D pose at this time.':'Awaiting PS1 / PS4 data.'}</p></div>}
      {posedHands.length>0&&<button onClick={()=>controlsRef.current?.reset()} className="absolute bottom-3 right-3 rounded-md border border-slate-200 bg-white/90 p-2 text-slate-500 hover:text-blue-600" title="Reset 3D view"><RotateCcw size={16}/></button>}
      <span className="absolute left-3 top-3 text-[10px] font-bold text-slate-400">{view==='rig'?'RIGGED HANDS':'3D HAND LANDMARKS'} · RELATIVE TO VIDEO</span>
    </div><div className="mt-3 flex items-center justify-between text-xs text-slate-500"><span className="flex items-center gap-1"><Move3d size={14}/> Drag to rotate · Scroll to zoom</span><span className="flex gap-2 font-semibold">{posedHands.map(hand=><span key={hand.hand_id} className={hand.side==='right'?'text-emerald-700':'text-amber-700'}>{hand.side} · {hand.hand_id}</span>)}</span></div><label className="mt-3 flex items-center gap-2 text-[11px] text-slate-500"><span className="shrink-0">Display size</span><input aria-label="Hand display size" className="min-w-0 flex-1" type="range" min="1" max="3.5" step="0.1" value={displayScale} onChange={e=>setDisplayScale(Number(e.target.value))}/><span className="tabular-nums">{displayScale.toFixed(1)}×</span></label><div className="mt-3 flex gap-2" aria-label="Hand display mode">{[['rig','Rigged hands'],['points','Landmarks']].map(([mode,label])=><button type="button" key={mode} onClick={()=>setView(mode)} aria-pressed={view===mode} className={`rounded-md border px-2.5 py-1.5 text-[11px] font-semibold ${view===mode?'border-blue-300 bg-blue-50 text-blue-700':'border-slate-200 text-slate-500'}`}>{label}</button>)}</div><p className="mt-3 text-[11px] leading-5 text-slate-400">{view==='rig'?'The rig follows the estimated joints and is an illustrative visualization.':'Both hands follow the current video frame using PS1 image landmarks and PS4 3D joints.'}</p>
  </Panel>;
}
