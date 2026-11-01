import {create} from 'zustand';
import {persist} from 'zustand/middleware';
export const useSearchStore=create(persist(set=>({query:'',filters:{hand:'all',action:'all'},setQuery:(query)=>set({query}),setFilter:(key,value)=>set(s=>({filters:{...s.filters,[key]:value}})),clear:()=>set({query:'',filters:{hand:'all',action:'all'}})}),{name:'forge-search-session-v1'}));
