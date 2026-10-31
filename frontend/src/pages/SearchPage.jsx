import {Search} from 'lucide-react';
import {Panel} from '@/components/shared/Panel';
import {SearchPanel} from '@/features/search/SearchPanel';
export default function SearchPage(){return <div className="mx-auto max-w-4xl space-y-6"><div><h1 className="text-3xl font-extrabold tracking-tight">Search Video Intelligence</h1><p className="mt-2 text-sm text-slate-500">Find evidence-linked hand, object and task events across indexed clips.</p></div><Panel title="Search indexed events"><SearchPanel/></Panel></div>}
