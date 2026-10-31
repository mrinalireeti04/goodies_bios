import { create } from 'zustand';
export const usePlaybackStore=create(set=>({
  currentTimeMs:0,isPlaying:false,playbackRate:1,selectedHandId:'H0',seekRequest:{t:0,nonce:0},
  overlays:{hands:true,objects:true,labels:true,connections:true},
  setTime:(currentTimeMs)=>set({currentTimeMs}),
  setPlaying:(isPlaying)=>set({isPlaying}),
  setRate:(playbackRate)=>set({playbackRate}),
  setHand:(selectedHandId)=>set({selectedHandId}),
  seekTo:(t)=>set(s=>({currentTimeMs:t,seekRequest:{t,nonce:s.seekRequest.nonce+1}})),
  toggleOverlay:(key)=>set(s=>({overlays:{...s.overlays,[key]:!s.overlays[key]}})),
  reset:()=>set({currentTimeMs:0,isPlaying:false,seekRequest:{t:0,nonce:0}})
}));
