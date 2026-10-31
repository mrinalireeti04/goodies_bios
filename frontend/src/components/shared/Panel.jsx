import { cn } from '@/lib/utils';
export function Panel({title,children,action,className,bodyClassName}){return <section className={cn('panel overflow-hidden',className)}><div className="flex min-h-12 items-center justify-between gap-3 border-b border-slate-100 px-4"><h2 className="studio-panel-heading">{title}</h2>{action}</div><div className={cn('p-4',bodyClassName)}>{children}</div></section>}
