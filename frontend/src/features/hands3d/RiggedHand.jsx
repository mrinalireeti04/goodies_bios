import React,{useLayoutEffect,useMemo} from 'react';
import {useLoader} from '@react-three/fiber';
import * as THREE from 'three';
import {FBXLoader} from 'three/addons/loaders/FBXLoader.js';
import {basis,makeRig,moveBones,vector} from './rigMath.js';

const MODEL='/assets/rigged-hand/hand.fbx';
export const RiggedHand=React.memo(function RiggedHand({joints,side}){
  const asset=useLoader(FBXLoader,MODEL);
  const rig=useMemo(()=>makeRig(asset,side),[asset,side]);
  const target=useMemo(()=>{
    const points=joints.map(vector),center=new THREE.Vector3();points.forEach(p=>center.add(p));center.divideScalar(points.length);
    const rotation=basis(points[0],points[5],points[9],points[17]);
    const scale=points[5].distanceTo(points[17])/rig.palmWidth;
    if(!rotation||!Number.isFinite(scale)||scale<=0)return null;
    return {wrist:points[0].clone().sub(center),rotation:rotation.multiply(rig.restBasis.clone().invert()),scale};
  },[joints,rig]);
  useLayoutEffect(()=>{if(target)moveBones(rig,joints);},[rig,joints,target]);
  if(!target)return null;
  const offset=rig.wristPosition.clone().multiplyScalar(-target.scale);
  return <group position={target.wrist} quaternion={target.rotation}><group position={offset} scale={target.scale}><primitive object={rig.scene}/></group></group>;
});
