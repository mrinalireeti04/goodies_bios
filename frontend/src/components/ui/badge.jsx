import { cn } from '@/lib/utils';
const variants = {default:'bg-blue-50 text-blue-700 border-blue-200',green:'bg-emerald-50 text-emerald-700 border-emerald-200',gray:'bg-slate-100 text-slate-600 border-slate-200',dark:'bg-white/10 text-slate-100 border-white/20',amber:'bg-amber-50 text-amber-700 border-amber-200'};
export function Badge({children,variant='default',className}){return <span className={cn('inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-semibold',variants[variant],className)}>{children}</span>}
