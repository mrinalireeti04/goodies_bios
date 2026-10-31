import React,{useEffect,useRef} from 'react';
import {HAND_CONNECTIONS} from '@/mocks/demoData';
import {usePlaybackStore} from '@/stores/usePlaybackStore';
export function AnnotationCanvas({frame,sourceWidth=1280,sourceHeight=720}){
  const canvasRef=useRef(null),overlays=usePlaybackStore(s=>s.overlays);
  useEffect(()=>{
    const canvas=canvasRef.current;if(!canvas)return;
    const draw=()=>{const bounds=canvas.getBoundingClientRect();if(!bounds.width||!bounds.height)return;const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(bounds.width*dpr);canvas.height=Math.round(bounds.height*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,bounds.width,bounds.height);if(!frame)return;
      const scale=Math.min(bounds.width/sourceWidth,bounds.height/sourceHeight),offsetX=(bounds.width-sourceWidth*scale)/2,offsetY=(bounds.height-sourceHeight*scale)/2;
      function pos(p){return [offsetX+p[0]*scale,offsetY+p[1]*scale]}
      const label=(text,x,y,color)=>{ctx.font='600 12px Inter, sans-serif';const width=ctx.measureText(text).width+16;ctx.fillStyle=color;ctx.fillRect(x,Math.max(0,y-22),width,20);ctx.fillStyle='#fff';ctx.fillText(text,x+8,Math.max(13,y-8));};
      if(overlays.objects){for(const obj of frame.objects||[]){const[x1,y1]=pos([obj.bbox_px[0],obj.bbox_px[1]]),[x2,y2]=pos([obj.bbox_px[2],obj.bbox_px[3]]);ctx.strokeStyle='#f66d92';ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);if(overlays.labels)label(`${obj.label} ${(obj.confidence*100).toFixed(0)}%`,x1,y1,'#c8486e');}}
      if(overlays.hands){for(const hand of frame.hands||[]){const color=hand.side==='right'?'#42e8ac':'#f7b95d';const pts=hand.kp2d_px.map(pos);if(overlays.connections){ctx.strokeStyle=color;ctx.lineWidth=2.5;for(const [a,b] of HAND_CONNECTIONS){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke();}}
        for(const [x,y] of pts){ctx.beginPath();ctx.arc(x,y,3.2,0,2*Math.PI);ctx.fillStyle='#e8fff7';ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.stroke();}
        const [x1,y1]=pos([hand.bbox_px[0],hand.bbox_px[1]]),[x2,y2]=pos([hand.bbox_px[2],hand.bbox_px[3]]);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.setLineDash([5,4]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);if(overlays.labels)label(`${hand.side} · ${hand.hand_id}`,x1,y1,color==='#42e8ac'?'#138d73':'#c17d2b');}}
    };
    const ro=new ResizeObserver(draw);ro.observe(canvas);draw();return ()=>ro.disconnect();
  },[frame,sourceWidth,sourceHeight,overlays]);
  return <canvas ref={canvasRef} aria-label="Hand landmark and object annotation overlay" className="pointer-events-none absolute inset-0 h-full w-full"/>;
}
