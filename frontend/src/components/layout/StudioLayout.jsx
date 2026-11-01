import { useEffect, useState } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { ArrowLeft, ChartNoAxesCombined, Download, Moon, ScanEye, Sun, UploadCloud } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useMediaStore } from '@/stores/useMediaStore';
const nav=[['/studio','Vision Explorer',ScanEye],['/analytics','Analytics',ChartNoAxesCombined],['/export','Export',Download]];
export function StudioLayout(){
  const setOpen=useMediaStore(s=>s.setUploadOpen),navigate=useNavigate();
  const [dark,setDark]=useState(()=>window.localStorage.getItem('forge-studio-theme')!=='light');

  useEffect(()=>{window.localStorage.setItem('forge-studio-theme',dark?'dark':'light')},[dark]);

  const shell=dark?'studio-shell studio-dark min-h-screen bg-[#0c1423] text-slate-100':'studio-shell studio-light min-h-screen bg-[#f5f7fa] text-slate-900';
  const header=dark?'sticky top-0 z-20 border-b border-white/10 bg-[#101827]/95 backdrop-blur':'sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur';
  return <div className={shell}>
    <header className={header}>
      <div className="mx-auto flex h-[65px] max-w-[1800px] items-center justify-between gap-2 px-4 md:px-7">
        <div className="flex items-center gap-8">
          <button className={`flex items-center gap-2.5 text-lg font-extrabold tracking-tight ${dark?'text-white':'text-slate-900'}`} onClick={()=>navigate('/')}>
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white"><ScanEye size={22}/></span>FORGE <span className={dark?'-ml-1 font-medium text-slate-300':'-ml-1 font-medium text-slate-500'}>Studio</span>
          </button>
          <nav className="hidden items-center gap-1 lg:flex">{nav.map(([path,label,Icon])=><NavLink key={path} to={path} className={({isActive})=>`flex h-[65px] items-center gap-2 border-b-2 px-3 text-[13px] font-semibold transition ${isActive?(dark?'border-blue-400 text-blue-300':'border-blue-600 text-blue-600'):(dark?'border-transparent text-slate-400 hover:text-white':'border-transparent text-slate-500 hover:text-slate-900')}`}><Icon size={16}/>{label}</NavLink>)}</nav>
        </div>
        <div className="flex items-center gap-2">
          <span className={dark?'hidden text-[11px] font-semibold text-slate-500 sm:block':'hidden text-[11px] font-semibold text-slate-400 sm:block'}>LOCAL DEMO</span>
          <Button size="sm" variant="outline" onClick={()=>setDark(value=>!value)} aria-label={`Switch to ${dark?'light':'dark'} theme`} title={`Switch to ${dark?'light':'dark'} theme`}>{dark?<Sun size={14}/>:<Moon size={14}/>}<span className="hidden sm:inline">{dark?'Light':'Dark'}</span></Button>
          <Button size="sm" variant="outline" onClick={()=>navigate('/')}><ArrowLeft size={14}/> Home</Button>
          <Button size="sm" onClick={()=>setOpen(true)}><UploadCloud size={15}/> <span className="hidden sm:inline">Upload files</span></Button>
        </div>
      </div>
      <nav className={dark?'flex overflow-x-auto border-t border-white/10 px-3 lg:hidden':'flex overflow-x-auto border-t border-slate-100 px-3 lg:hidden'}>{nav.map(([path,label])=><NavLink key={path} to={path} className={({isActive})=>`shrink-0 px-3 py-2 text-xs font-semibold ${isActive?(dark?'text-blue-300':'text-blue-700'):(dark?'text-slate-400':'text-slate-500')}`}>{label}</NavLink>)}</nav>
    </header>
    <main className="mx-auto max-w-[1800px] px-4 py-5 md:px-7"><Outlet/></main>
  </div>
}
