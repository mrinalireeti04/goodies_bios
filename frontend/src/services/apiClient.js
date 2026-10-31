const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
export async function apiFetch(path, options={}){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),options.timeoutMs||30000);
  const signal=options.signal && typeof AbortSignal.any==='function'
    ? AbortSignal.any([options.signal,controller.signal])
    : options.signal||controller.signal;
  try{
    const response=await fetch(`${BASE}${path}`,{...options,signal});
    if(!response.ok)throw new Error(`Request failed (${response.status})`);
    return response.status===204?null:response.json();
  }finally{clearTimeout(timer);}
}
export const useRealApi = import.meta.env.VITE_USE_REAL_API==='true';
