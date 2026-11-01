import {useEffect,useState} from 'react';
import {usePlaybackStore} from './usePlaybackStore';

const DEFAULT_INTERVAL_MS=66;

// Keep native video playback smooth while limiting React-driven panels to ~15 fps.
export function usePlaybackTime(intervalMs=DEFAULT_INTERVAL_MS){
  const [time,setTime]=useState(()=>usePlaybackStore.getState().currentTimeMs);
  useEffect(()=>{
    let lastUpdate=performance.now();
    let timer=null;
    let seekNonce=usePlaybackStore.getState().seekRequest.nonce;
    const publish=()=>{
      const latest=usePlaybackStore.getState();
      lastUpdate=performance.now();
      setTime(current=>current===latest.currentTimeMs?current:latest.currentTimeMs);
    };
    const unsubscribe=usePlaybackStore.subscribe(state=>{
      const seekChanged=state.seekRequest.nonce!==seekNonce;
      seekNonce=state.seekRequest.nonce;
      if(!state.isPlaying||seekChanged){
        if(timer!==null)clearTimeout(timer);
        timer=null;
        publish();
        return;
      }
      const wait=intervalMs-(performance.now()-lastUpdate);
      if(wait<=0){
        if(timer!==null)clearTimeout(timer);
        timer=null;
        publish();
      }else if(timer===null){
        timer=setTimeout(()=>{timer=null;publish();},wait);
      }
    });
    return ()=>{unsubscribe();if(timer!==null)clearTimeout(timer);};
  },[intervalMs]);
  return time;
}
