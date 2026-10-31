import {BarChart,Bar,XAxis,YAxis,CartesianGrid,Tooltip,ResponsiveContainer,Cell} from 'recharts';
import {Panel} from '@/components/shared/Panel';
import {clipActions,clipContacts} from '@/services/clipData';
import {useAnnotationStore} from '@/stores/useAnnotationStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {useMediaStore} from '@/stores/useMediaStore';
import {Badge} from '@/components/ui/badge';
export default function AnalyticsPage(){
  const selectedClipId=useMediaStore(s=>s.selectedClipId),clips=useMediaStore(s=>s.clips),clip=clips.find(c=>c.id===selectedClipId),overrides=useAnnotationStore(s=>s.overrides),results=useInferenceStore(s=>s.byClip[selectedClipId]);
  const actions=clipActions(clip,results,overrides[selectedClipId]||{}),contacts=clipContacts(clip,results);
  const data=actions.map(e=>({name:e.verb,seconds:Number(((e.end_ms-e.start_ms)/1000).toFixed(2)),id:e.action_id}));
  const total=actions.reduce((sum,a)=>sum+(a.end_ms-a.start_ms),0)/1000;
  return <div className="mx-auto max-w-6xl space-y-5"><div className="flex flex-wrap items-center gap-3"><h1 className="text-3xl font-extrabold tracking-tight">Workflow Analytics</h1><Badge variant={clip?.type==='demo'?'amber':'gray'}>{clip?.type==='demo'?'Synthetic data':'Selected clip'}</Badge></div><p className="text-sm text-slate-500">{clip?.name||'No selected media'} · Evidence-based action durations and inferred contact intervals. These do not establish worker productivity.</p>
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3"><div className="panel p-5"><p className="text-xs text-slate-500">Annotated actions</p><p className="mt-2 text-3xl font-bold">{actions.length}</p></div><div className="panel p-5"><p className="text-xs text-slate-500">Sum of action durations</p><p className="mt-2 text-3xl font-bold">{total.toFixed(1)}s</p></div><div className="panel p-5"><p className="text-xs text-slate-500">Contact events</p><p className="mt-2 text-3xl font-bold">{contacts.length}</p></div></div>
    <Panel title="Action durations">{data.length?<div className="h-[370px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{top:18,right:22,left:0,bottom:12}}><CartesianGrid strokeDasharray="3 3" stroke="#e7ebf1"/><XAxis dataKey="name" tick={{fontSize:12}}/><YAxis tick={{fontSize:12}} unit="s"/><Tooltip formatter={v=>[`${v} sec`,'Duration']}/><Bar dataKey="seconds" radius={[6,6,0,0]} fill="#3b82f6">{data.map((_,i)=><Cell key={i} fill={['#5b9cf6','#58b98f','#e6af55','#9ba6b6','#a084dd'][i%5]}/>)}</Bar></BarChart></ResponsiveContainer></div>:<p className="p-10 text-center text-sm text-slate-500">Import actions.json (PS3) for this clip to display real event analytics.</p>}
      <p className="mt-3 text-xs text-slate-500">{clip?.type==='demo'?'Illustrative values only.':'Only imported labels are counted. Overlapping intervals may double-count duration.'}</p></Panel></div>;
}
