// All examples here are synthetic. Never treat these records as model inference.
export const DEMO_ID = 'synthetic-demo';
export const demoClip = { id: DEMO_ID, name: 'Synthetic Assembly Walkthrough', type: 'demo', status: 'demo', durationMs: 84000, width: 1280, height: 720, size: 0, url: '/assets/mock-workstation.svg', synthetic: true, mimeType: 'image/svg+xml' };
export const initialActions = [
  { action_id:'A001', clip_id:DEMO_ID,start_ms:0,end_ms:12000,verb:'pick',object_label:'metal component',hand:'left',hand_ids:['H1'],confidence:.92,source:'synthetic' },
  { action_id:'A002', clip_id:DEMO_ID,start_ms:12000,end_ms:27000,verb:'hold',object_label:'metal component',hand:'both',hand_ids:['H0','H1'],confidence:.88,source:'synthetic' },
  { action_id:'A003', clip_id:DEMO_ID,start_ms:27000,end_ms:57000,verb:'fasten',object_label:'screwdriver',hand:'right',hand_ids:['H0'],confidence:.90,source:'synthetic' },
  { action_id:'A004', clip_id:DEMO_ID,start_ms:57000,end_ms:67000,verb:'unknown',object_label:null,hand:'unknown',hand_ids:[],confidence:null,source:'synthetic' },
  { action_id:'A005', clip_id:DEMO_ID,start_ms:67000,end_ms:84000,verb:'inspect',object_label:'assembly',hand:'both',hand_ids:['H0','H1'],confidence:.84,source:'synthetic' },
];
export const mockContacts = [
  { event_id:'C001',clip_id:DEMO_ID,start_ms:2000,end_ms:12000,hand_id:'H1',object_id:'O1',object_label:'metal component',state:'portable',interaction:'hold',confidence:.82 },
  { event_id:'C002',clip_id:DEMO_ID,start_ms:27000,end_ms:57000,hand_id:'H0',object_id:'O2',object_label:'screwdriver',state:'portable',interaction:'hold',confidence:.79 },
];
const points = [ [0,0],[-.35,.12],[-.55,.32],[-.72,.43],[-.86,.52],[-.2,.38],[-.22,.68],[-.25,.94],[-.26,1.12],[.03,.44],[.05,.81],[.05,1.09],[.05,1.27],[.25,.39],[.33,.74],[.38,.97],[.42,1.11],[.45,.25],[.59,.52],[.66,.72],[.74,.86] ];
export const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
export function syntheticFrameAt(tMs) {
  const t = Math.min(84000,Math.max(0,tMs));
  if (t >= 57000 && t < 67000) return {schema_version:'1.0',clip_id:DEMO_ID,frame_idx:Math.floor(t/40),t_ms:t,frame_width:1280,frame_height:720,hands:[],objects:[]};
  const shift = Math.sin(t/1800)*15, curve = Math.sin(t/2300)*6;
  const hands = ['right','left'].map((side,idx)=>{
    const sign = idx===0?-1:1;
    const cx = idx===0?765:535, cy=405;
    const kp2d_px=points.map(([x,y],j)=>[cx+sign*x*71+shift,cy-y*94+curve+(j%2)*1.5]);
    const kp3d_hand_m=points.map(([x,y],j)=>[sign*x*.055,y*.062,Math.sin(j*.7+t/2600)*.006]);
    const xs=kp2d_px.map(p=>p[0]),ys=kp2d_px.map(p=>p[1]);
    return {hand_id:idx===0?'H0':'H1',side,handedness_score:idx===0?.96:.93,bbox_px:[Math.min(...xs)-12,Math.min(...ys)-12,Math.max(...xs)+12,Math.max(...ys)+12],kp2d_px,kp3d_hand_m,source:'detected'};
  });
  const obj = t>=27000 && t<57000 ? [{object_id:'O2',label:'screwdriver',type:'portable',bbox_px:[644,157,748,343],confidence:.88}]:[{object_id:'O1',label:'metal component',type:'portable',bbox_px:[555,301,716,408],confidence:.89}];
  return {schema_version:'1.0',clip_id:DEMO_ID,frame_idx:Math.floor(t/40),t_ms:t,frame_width:1280,frame_height:720,hands,objects:obj};
}
export function getMergedActions(overrides={}) {return initialActions.map(a=>({...a,...overrides[a.action_id]}));}
export function queryEvents(query='',filters={},overrides={}){
  const q=query.trim().toLowerCase();
  return getMergedActions(overrides).filter(a=>{
    const haystack=[a.verb,a.object_label,a.hand,a.action_id].filter(Boolean).join(' ').toLowerCase();
    return (!q || haystack.includes(q)) && (!filters.hand || filters.hand==='all' || a.hand===filters.hand || a.hand==='both') && (!filters.action || filters.action==='all' || a.verb===filters.action);
  });
}
