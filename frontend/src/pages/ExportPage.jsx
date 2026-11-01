import {Download,FileJson,FileSpreadsheet,ShieldCheck} from 'lucide-react';
import {Panel} from '@/components/shared/Panel';
import {Button} from '@/components/ui/button';
import {clipActions} from '@/services/clipData';
import {reviewedPS2} from '@/services/annotationData';
import {useAnnotationStore} from '@/stores/useAnnotationStore';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {downloadText} from '@/lib/utils';
import {useQuery} from '@tanstack/react-query';
import {fetchJob} from '@/services/jobService';

export default function ExportPage(){
  const selectedClipId=useMediaStore(s=>s.selectedClipId),clips=useMediaStore(s=>s.clips),clip=clips.find(c=>c.id===selectedClipId),results=useInferenceStore(s=>s.byClip[selectedClipId]);
  const allOverrides=useAnnotationStore(s=>s.overrides),allObjectEdits=useAnnotationStore(s=>s.objectOverrides),allManual=useAnnotationStore(s=>s.manualObjects),allAdded=useAnnotationStore(s=>s.addedActions);
  const overrides=allOverrides[selectedClipId]||{},objectEdits=allObjectEdits[selectedClipId]||{},manual=allManual[selectedClipId]||[],added=allAdded[selectedClipId]||[],actions=clipActions(clip,results,overrides,added),name=clip?.name?.replace(/\.[^.]*$/,'')||'annotations';
  const reviewed=reviewedPS2(results?.modules?.ps2,objectEdits,manual);
  const {data:job}=useQuery({queryKey:['job',clip?.jobId],queryFn:()=>fetchJob(clip.jobId),enabled:!!clip?.jobId});
  function exportJson(){downloadText(`${name}-annotations.json`,JSON.stringify({schema_version:'1.0',source:clip?.type==='demo'?'synthetic':'imported',clip_name:clip?.name,source_clip_id:results?.sourceClipId??clip?.id,actions,contacts:reviewed?.contacts||[],corrections:{actions:overrides,objects:objectEdits,manual_objects:manual,manual_actions:added},reviewed_ps2:reviewed,audit:job?{processing_duration_s:job.processing_duration_s??null,module_timing:job.module_timing||{},issues:job.issues||[]}:null,original_modules:results?.modules||{}},null,2));}
  function exportPs2(){downloadText(`${name}-ps2-reviewed.json`,JSON.stringify(reviewed,null,2));}
  function exportPs3(){downloadText(`${name}-ps3-reviewed.json`,JSON.stringify({schema_version:'1.0',clip_id:results?.sourceClipId,actions:actions.map(a=>({...a,clip_id:results?.sourceClipId})),review_note:'Edited events; model-generated grouping and summaries are omitted because they were not recomputed.'},null,2));}
  function exportPs1(){downloadText(`${name}-ps1-hands.jsonl`,results.modules.ps1.map(frame=>JSON.stringify(frame)).join('\n')+'\n','application/x-ndjson');}
  function exportPs4(){downloadText(`${name}-ps4-pose.json`,JSON.stringify(results.modules.ps4,null,2));}
  function exportCsv(){const esc=v=>`"${String(v??'').replaceAll('"','""')}"`;const rows=[['action_id','start_ms','end_ms','verb','object_label','hand','source'],...actions.map(a=>[a.action_id,a.start_ms,a.end_ms,a.verb,a.object_label,a.hand,a.source])];downloadText(`${name}-actions.csv`,rows.map(r=>r.map(esc).join(',')).join('\n'),'text/csv');}
  return <div className="mx-auto max-w-4xl space-y-6"><div><h1 className="text-3xl font-extrabold tracking-tight">Export Annotations</h1><p className="mt-2 text-sm text-slate-500">Selected clip: <b>{clip?.name||'None'}</b>. Export model results and manual corrections while preserving originals.</p></div>
    <div className="grid gap-4 md:grid-cols-2"><Panel title="Complete JSON dataset"><div className="flex flex-col items-start gap-3"><FileJson className="text-blue-600" size={30}/><p className="text-sm text-slate-500">Original PS1–PS4 results, corrections and reviewed objects and events.</p><Button disabled={!actions.length&&!results} onClick={exportJson}><Download size={16}/> Download JSON</Button></div></Panel><Panel title="Reviewed results"><div className="flex flex-wrap items-start gap-3"><Button disabled={!reviewed} onClick={exportPs2} variant="outline"><Download size={16}/> PS2 objects JSON</Button><Button disabled={!results?.sourceClipId||!actions.length} onClick={exportPs3} variant="outline"><Download size={16}/> PS3 timeline JSON</Button><Button disabled={!results?.modules?.ps1?.length} onClick={exportPs1} variant="outline"><Download size={16}/> PS1 hands JSONL</Button><Button disabled={!results?.modules?.ps4} onClick={exportPs4} variant="outline"><Download size={16}/> PS4 pose JSON</Button><Button disabled={!actions.length} onClick={exportCsv} variant="outline"><FileSpreadsheet size={16}/> Actions CSV</Button><p className="text-xs text-slate-500">PS2 safety and tool summaries are omitted from reviewed output because corrections do not recompute them.</p></div></Panel></div>
    <div className="flex items-start gap-3 rounded-lg border border-emerald-100 bg-emerald-50 p-4 text-sm text-emerald-800"><ShieldCheck size={21} className="shrink-0"/>Real factory annotations are private data too. Exports remain on your machine and must not be committed to public GitHub.</div></div>;
}
