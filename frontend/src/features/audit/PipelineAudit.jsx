import {useQuery} from '@tanstack/react-query';
import {Clock3,ChevronDown} from 'lucide-react';
import {Panel} from '@/components/shared/Panel';
import {fetchJob} from '@/services/jobService';
import {useInferenceStore} from '@/stores/useInferenceStore';

const duration=value=>{
  if(!Number.isFinite(value)||value<0)return null;
  if(value<60)return `${value.toFixed(1)} s`;
  const minutes=Math.floor(value/60),seconds=Math.round(value%60);
  return seconds===60?`${minutes+1} min`:`${minutes} min ${seconds} s`;
};
const label=key=>key.replaceAll('_',' ');

export function PipelineAudit({clip,compact=false}){
  const results=useInferenceStore(s=>s.byClip[clip?.id]);
  const {data:job,isError,isLoading,refetch}=useQuery({
    queryKey:['job',clip?.jobId],queryFn:()=>fetchJob(clip.jobId),enabled:!!clip?.jobId,
    refetchInterval:query=>clip?.moduleStatus?.ps3==='processing'||['queued','processing'].includes(query.state.data?.status)||Object.values(query.state.data?.module_status||{}).includes('processing')?5000:false,
  });
  const timings=job?.module_timing||{},ps2=results?.modules?.ps2?.meta||{},ps3=results?.modules?.ps3?.processing_metadata||{};
  const stage=(key)=>timings[key]?.duration_s??(job?.module_status?.[key]==='processing'&&Number.isFinite(timings[key]?.started_at)?Math.max(0,Date.now()/1000-timings[key].started_at):null);
  const total=job?.processing_duration_s??(job?.status==='processing'&&Number.isFinite(job?.processing_started_at)?Math.max(0,Date.now()/1000-job.processing_started_at):null);
  const rows=[['PS1 + PS4','Hand tracking and 3D pose',stage('ps1'),job?.module_status?.ps1],['PS2','Objects and contact',stage('ps2'),job?.module_status?.ps2],['PS3','Task timeline',stage('ps3')??ps3.processing_time_s,job?.module_status?.ps3]];
  const details=[['PS2 · Gemini ledger',ps2.ledger_run?.seconds],['PS2 · SAM2 masks',ps2.sam2?.seconds],...Object.entries(ps2.stage_timings_s||{}).map(([key,value])=>[`PS2 · ${label(key)}`,value]),...Object.entries(ps3.stage_timings_s||{}).map(([key,value])=>[`PS3 · ${label(key)}`,value])].filter(([,value])=>Number.isFinite(value));
  const missing=rows.some(([, ,value])=>duration(value)==null)||duration(total)==null;
  return <Panel title="Pipeline audit" action={!compact&&<span className="flex items-center gap-1.5 text-[10px] text-slate-500"><Clock3 size={13}/> Processing time</span>}>
    {!clip?.jobId?<p className="text-xs text-slate-500">Timing is available for clips processed by the local backend.</p>:isLoading?<p className="text-xs text-slate-500">Loading pipeline timings…</p>:isError?<div className="flex flex-wrap items-center justify-between gap-2 text-xs"><span className="text-rose-700">Could not load the backend job timings.</span><button className="font-semibold text-blue-600" onClick={()=>refetch()}>Try again</button></div>:<div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-slate-50 px-4 py-3"><div><p className="text-xs font-semibold text-slate-800">Total processing</p><p className="mt-1 text-[11px] text-slate-500">Excludes upload{job?.status==='processing'?' · in progress':''}</p></div><span className="text-lg font-bold tabular-nums text-slate-800">{duration(total)||'Not recorded'}</span></div>
      <div className={compact?"grid gap-2":"grid gap-2 sm:grid-cols-3"}>{rows.map(([name,description,value,status])=><div key={name} className="rounded-lg border border-slate-200 px-3 py-3"><div className="flex items-center justify-between gap-2"><span className="text-xs font-bold text-slate-800">{name}</span><span className="text-xs font-semibold tabular-nums text-slate-800">{duration(value)||'—'}</span></div><p className="mt-1 text-[11px] text-slate-500">{description}</p><p className="mt-1 text-[10px] capitalize text-slate-500">{status||'Status unavailable'}{status==='processing'?' · elapsed':''}</p></div>)}</div>
      {missing&&<p className="text-[11px] leading-5 text-slate-500">Some timings were not recorded for this run. Detailed stage timings are available on new runs; missing values cannot be reconstructed.</p>}
      {details.length>0&&<details className="group rounded-lg border border-slate-200"><summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-semibold text-slate-800">Sub-stage timings <ChevronDown size={15} className="text-slate-400 transition-transform group-open:rotate-180"/></summary><div className="space-y-2 border-t border-slate-100 px-3 py-3">{details.map(([name,value])=><div key={name} className="flex justify-between gap-3 text-xs"><span className="capitalize text-slate-600">{name}</span><span className="shrink-0 tabular-nums text-slate-700">{duration(value)}</span></div>)}</div></details>}
      <p className="text-[10px] leading-4 text-slate-500">PS2 and PS3 can overlap, so their durations should not be added together. PS3 may include time spent waiting for object context.</p>
    </div>}
  </Panel>;
}
