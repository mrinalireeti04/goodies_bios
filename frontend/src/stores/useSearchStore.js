import {create} from 'zustand';
export const useSearchStore=create(set=>({query:'',filters:{hand:'all',action:'all'},setQuery:(query)=>set({query}),setFilter:(key,value)=>set(s=>({filters:{...s.filters,[key]:value}})),clear:()=>set({query:'',filters:{hand:'all',action:'all'}})}));
