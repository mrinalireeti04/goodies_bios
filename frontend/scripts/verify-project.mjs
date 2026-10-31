import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {demoClip,syntheticFrameAt,getMergedActions} from '../src/mocks/demoData.js';
import {validatePS1Frame} from '../src/services/adapters/ps1Adapter.js';
import {parseResultFile,mergeResultFiles,closestFrame,frameWithObjects,threeDHandsAt} from '../src/services/annotationData.js';
import {clipActions,clip3DHands,searchInLibrary} from '../src/services/clipData.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(import.meta.url);
let ts;
try {ts=require('typescript');} catch {try {ts=require('/opt/nvm/versions/node/v22.16.0/lib/node_modules/typescript/lib/typescript.js');}catch{ts=null;}}
function walk(folder){return fs.readdirSync(folder,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?walk(path.join(folder,entry.name)):[path.join(folder,entry.name)]);}
const files=walk(path.join(root,'src')).filter(f=>/\.jsx?$/.test(f));
const missing=[];
for(const f of files){
  const source=fs.readFileSync(f,'utf8');
  for(const match of source.matchAll(/(?:from\s*|import\s*\()\s*['"](@\/[^'"]+)['"]/g)){
    const rel=match[1].slice(2),base=path.join(root,'src',rel);
    if(![base,base+'.js',base+'.jsx',path.join(base,'index.js'),path.join(base,'index.jsx')].some(p=>fs.existsSync(p)))missing.push([path.relative(root,f),match[1]]);
  }
  if(ts){
    const d=ts.transpileModule(source,{fileName:f,reportDiagnostics:true,compilerOptions:{jsx:ts.JsxEmit.ReactJSX,allowJs:true,target:ts.ScriptTarget.ES2022}}).diagnostics||[];
    const errors=d.filter(e=>e.category===ts.DiagnosticCategory.Error);
    assert.equal(errors.length,0,`${path.relative(root,f)} syntax errors: ${errors.map(e=>ts.flattenDiagnosticMessageText(e.messageText,'\n')).join('; ')}`);
  }
}
assert.equal(missing.length,0,`Missing local imports: ${JSON.stringify(missing)}`);
for(const t of [0,10000,30000,68000,80000]){
  const frame=syntheticFrameAt(t);validatePS1Frame(frame);assert.equal(frame.clip_id,demoClip.id);
}
assert.equal(syntheticFrameAt(60000).hands.length,0);
assert.equal(syntheticFrameAt(4000).hands.length,2);
assert.equal(getMergedActions().length,5);
const one=syntheticFrameAt(4000),two=syntheticFrameAt(8000);
one.clip_id='factory_sample';two.clip_id='factory_sample';
const records=parseResultFile('hands.jsonl',[one,two].map(x=>JSON.stringify(x)).join('\n'));
const actions=parseResultFile('actions.json',JSON.stringify({schema_version:'1.0',clip_id:'factory_sample',actions:[{action_id:'A1',start_ms:1000,end_ms:9000,verb:'fasten',object_label:'screwdriver',hand:'right',hand_ids:['H0'],confidence:null}]}));
const object=parseResultFile('ps2.json',JSON.stringify({schema_version:'1.0',clip_id:'factory_sample',frames:[{frame_idx:one.frame_idx,t_ms:4000,objects:one.objects,hand_states:[]}],contacts:[]}));
const pose=parseResultFile('ps4.json',JSON.stringify({schema_version:'1.0',clip_id:'factory_sample',coordinate_frame:'hand_centered',frames:[{frame_idx:one.frame_idx,t_ms:4000,hands:one.hands.map(h=>({hand_id:h.hand_id,side:h.side,joints_3d_m:h.kp3d_hand_m}))}]}));
const imported=mergeResultFiles(null,[records,actions,object,pose]);
assert.deepEqual(Object.keys(imported.modules).sort(),['ps1','ps2','ps3','ps4']);
assert.equal(closestFrame(imported.modules.ps1,4001).frame_idx,one.frame_idx);
assert.equal(closestFrame(imported.modules.ps1,7000),null,'No stale hand observations');
assert.equal(frameWithObjects(imported,4000).objects.length,one.objects.length);
assert.equal(threeDHandsAt(imported,4000).length,2);
const clip={id:'local-clip',type:'video',name:'factory_sample.mp4'};
assert.equal(clipActions(clip,imported).length,1);
assert.equal(clip3DHands(clip,imported,4000).length,2);
assert.equal(searchInLibrary([clip],{[clip.id]:imported},{},'screwdriver',{hand:'right'}).length,1);
assert.equal(searchInLibrary([clip],{[clip.id]:imported},{},'screwdriver',{hand:'left'}).length,0);
assert.throws(()=>mergeResultFiles(imported,[{...actions,sourceClipId:'other_clip'}]),/same source clip_id/);
console.log(`PASS: ${files.length} JS/JSX source files validated${ts?' (TypeScript syntax parser)':''}, local imports resolved; PS1–PS4 import/search/timestamp and no-stale-pose tests passed.`);
console.log('NOTE: npm install and full Vite build require npm registry access; browser rendering is not verified.');
