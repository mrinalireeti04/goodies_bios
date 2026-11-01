import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from 'three';
import {FBXLoader} from 'three/addons/loaders/FBXLoader.js';
import {makeRig,moveBones,basis,vector} from '../src/features/hands3d/rigMath.js';
import {syntheticFrameAt} from '../src/mocks/demoData.js';
THREE.TextureLoader.prototype.load=()=>new THREE.Texture();
const data=fs.readFileSync(new URL('../public/assets/rigged-hand/hand.fbx',import.meta.url));
const asset=new FBXLoader().parse(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),'');
const hands=syntheticFrameAt(1000).hands;
for(const hand of hands){
  const rig=makeRig(asset,hand.side);
  assert.equal(rig.bones.length,15);
  const mesh=rig.scene.getObjectByName(hand.side==='left'?'Cube005':'Cube000');
  assert.equal(mesh.visible,true);assert.equal(mesh.frustumCulled,false);assert.equal(mesh.material.opacity,1);
  const joints=hand.kp3d_hand_m,points=joints.map(vector),q=basis(points[0],points[5],points[9],points[17]);
  assert.ok(q&&Number.isFinite(q.w));
  moveBones(rig,joints);
  assert.ok(rig.bones.every(({bone})=>bone.quaternion.toArray().every(Number.isFinite)));
  assert.ok(rig.bones.some(({bone,rest})=>Math.abs(bone.quaternion.dot(rest))<0.999),'Pose moves at least one bone');
}
console.log('PASS: both FBX hand rigs load, remain visible, and respond to 21 landmarks.');
