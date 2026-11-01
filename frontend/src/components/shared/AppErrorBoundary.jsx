import React from 'react';

export class AppErrorBoundary extends React.Component{
  state={error:null};
  static getDerivedStateFromError(error){return {error};}
  componentDidCatch(error,info){console.error('FORGE UI error',error,info);}
  render(){
    if(!this.state.error)return this.props.children;
    return <main className="flex min-h-screen items-center justify-center bg-slate-50 p-6"><section className="max-w-lg rounded-2xl border border-rose-200 bg-white p-6 shadow-sm"><p className="text-xs font-bold uppercase tracking-[0.18em] text-rose-600">Workspace recovered with an error</p><h1 className="mt-2 text-xl font-extrabold text-slate-900">This view could not render</h1><p className="mt-2 text-sm leading-6 text-slate-600">The saved media and backend results are still stored. Return to the studio or reload after checking the event data.</p><button type="button" className="mt-5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-bold text-white" onClick={()=>this.setState({error:null})}>Try again</button></section></main>;
  }
}
