import {Download,FileJson,FileSpreadsheet,ShieldCheck} from 'lucide-react';
import {Panel} from '@/components/shared/Panel';
import {Button} from '@/components/ui/button';
import {clipActions,clipContacts} from '@/services/clipData';
import {useAnnotationStore} from '@/stores/useAnnotationStore';
import {useMediaStore} from '@/stores/useMediaStore';
import {useInferenceStore} from '@/stores/useInferenceStore';
import {downloadText} from '@/lib/utils';
export default function ExportPage(){
  const selectedClipId=useMediaStore(s=>s.selectedClipId),clips=useMediaStore(s=>s.clips),clip=clips.find(c=>c.id===selectedClipId),allOverrides=useAnnotationStore(s=>s.overrides),results=useInferenceStore(s=>s.byClip[selectedClipId]);
  const overrides=allOverrides[selectedClipId]||{},actions=clipActions(clip,results,overrides),contacts=clipContacts(clip,results),name=clip?.name?.replace(/\.[^.]*$/,'')||'annotations';
  function exportJson(){downloadText(`${name}-annotations.json`,JSON.stringify({schema_version:'1.0',source:clip?.type==='demo'?'synthetic':'imported',clip_name:clip?.name,source_clip_id:results?.sourceClipId??clip?.id,actions,contacts,corrections:overrides,hands:results?.modules.ps1||[],pose3d:results?.modules.ps4?.frames||[]},null,2));}
  function exportCsv(){const esc=v=>`"${String(v??'').replaceAll('"','""')}"`;const rows=[['action_id','start_ms','end_ms','verb','object_label','hand','source'],...actions.map(a=>[a.action_id,a.start_ms,a.end_ms,a.verb,a.object_label,a.hand,a.source])];downloadText(`${name}-actions.csv`,rows.map(r=>r.map(esc).join(',')).join('\n'),'text/csv');}
  return <div className="mx-auto max-w-4xl space-y-6"><div><h1 className="text-3xl font-extrabold tracking-tight">Export Annotations</h1><p className="mt-2 text-sm text-slate-500">Selected clip: <b>{clip?.name||'None'}</b>. Export model results with manual corrections while preserving originals.</p></div>
    <div className="grid gap-4 md:grid-cols-2"><Panel title="JSON dataset"><div className="flex flex-col items-start gap-3"><FileJson className="text-blue-600" size={30}/><p className="text-sm text-slate-500">Actions, contacts, corrected fields and available hand/3D frames.</p><Button disabled={!actions.length&&!results} onClick={exportJson}><Download size={16}/> Download JSON</Button></div></Panel><Panel title="CSV actions"><div className="flex flex-col items-start gap-3"><FileSpreadsheet className="text-emerald-600" size={30}/><p className="text-sm text-slate-500">Corrected action timeline for spreadsheet analysis.</p><Button disabled={!actions.length} variant="outline" onClick={exportCsv}><Download size={16}/> Download CSV</Button></div></Panel></div>
    <div className="flex items-start gap-3 rounded-lg border border-emerald-100 bg-emerald-50 p-4 text-sm text-emerald-800"><ShieldCheck size={21} className="shrink-0"/>Real factory annotations are private data too. Exports remain on your machine and must not be committed to public GitHub. Delete copies at the end of the event.</div></div>;
}
