import React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva } from 'class-variance-authority';
import { cn } from '@/lib/utils';
const styles = cva('inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed', { variants:{ variant:{ default:'bg-blue-600 text-white hover:bg-blue-700',outline:'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50',ghost:'text-slate-600 hover:bg-slate-100',secondary:'bg-slate-100 text-slate-800 hover:bg-slate-200',dark:'bg-white text-slate-950 hover:bg-slate-100'},size:{default:'h-10 px-4 py-2',sm:'h-8 px-3 text-xs',lg:'h-12 px-6',icon:'h-9 w-9'} },defaultVariants:{variant:'default',size:'default'} });
export const Button = React.forwardRef(function Button({className,variant,size,asChild=false,...props},ref){const Comp=asChild?Slot:'button';return <Comp ref={ref} className={cn(styles({variant,size}),className)} {...props}/>;});
