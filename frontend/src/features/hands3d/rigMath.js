import * as THREE from 'three';
import {clone} from 'three/addons/utils/SkeletonUtils.js';

const fingers=[['thumb',[1,2,3,4]],['finger_index',[5,6,7,8]],['finger_middle',[9,10,11,12]],['finger_ring',[13,14,15,16]],['finger_pinky',[17,18,19,20]]];
export const vector=p=>new THREE.Vector3(p[0]*9,-p[1]*9,p[2]*9);
const world=node=>node.getWorldPosition(new THREE.Vector3());
export function basis(wrist,index,middle,pinky){
  const across=index.clone().sub(pinky),forward=middle.clone().sub(wrist);
  if(across.lengthSq()<1e-10||forward.lengthSq()<1e-10)return null;
  across.normalize();forward.addScaledVector(across,-forward.dot(across));
  if(forward.lengthSq()<1e-10)return null;
  forward.normalize();const normal=new THREE.Vector3().crossVectors(across,forward).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(across,forward,normal));
}
export function makeRig(asset,side){
  const scene=clone(asset),suffix=side==='left'?'L':'R',meshName=side==='left'?'Cube005':'Cube000';
  let mesh;
  scene.traverse(node=>{if(node.isMesh){node.visible=node.name===meshName;node.frustumCulled=false;if(node.name===meshName){mesh=node;const fixMaterial=material=>{const copy=material.clone();copy.opacity=1;copy.transparent=false;copy.alphaMap=null;copy.side=THREE.DoubleSide;return copy;};node.material=Array.isArray(node.material)?node.material.map(fixMaterial):fixMaterial(node.material);}}if(node.isLight||node.isCamera)node.visible=false;});
  if(!mesh?.skeleton)throw new Error('The hand model has no usable skeleton');
  // FBX contains duplicate leaf bones for one mesh; animate the shared parent rig.
  const find=name=>scene.getObjectByName(name);
  const wrist=find(`hand${suffix}`),index=find(`finger_index01${suffix}`),middle=find(`finger_middle01${suffix}`),pinky=find(`finger_pinky01${suffix}`);
  if(!wrist||!index||!middle||!pinky)throw new Error('The hand model is missing palm bones');
  scene.updateMatrixWorld(true);
  const wristPosition=world(wrist),restBasis=basis(wristPosition,world(index),world(middle),world(pinky));
  const palmWidth=world(index).distanceTo(world(pinky));
  const bones=[];
  for(const [name,indices] of fingers)for(let i=0;i<3;i++){
    const bone=find(`${name}0${i+1}${suffix}`);
    const nextName=i<2?`${name}0${i+2}${suffix}`:`${name}03${suffix}_end`;
    const next=i<2?find(nextName):bone?.children.find(child=>child.name===nextName);
    if(!bone||!next)throw new Error(`Missing bone ${nextName}`);
    bones.push({bone,next,from:indices[i],to:indices[i+1],rest:bone.quaternion.clone()});
  }
  if(!restBasis||palmWidth<1e-6)throw new Error('The hand model has an invalid rest pose');
  return {scene,bones,wristPosition,restBasis,palmWidth};
}
export function moveBones(rig,joints){
  for(const {bone,rest} of rig.bones)bone.quaternion.copy(rest);
  rig.scene.updateWorldMatrix(true,true);
  for(const {bone,next,from,to} of rig.bones){
    const actual=world(next).sub(world(bone));
    const desired=vector(joints[to]).sub(vector(joints[from]));
    if(actual.lengthSq()<1e-10||desired.lengthSq()<1e-10)continue;
    const delta=new THREE.Quaternion().setFromUnitVectors(actual.normalize(),desired.normalize());
    const parent=new THREE.Quaternion();bone.parent.getWorldQuaternion(parent);
    bone.quaternion.premultiply(parent.clone().invert().multiply(delta).multiply(parent));
    rig.scene.updateWorldMatrix(true,true);
  }
}
