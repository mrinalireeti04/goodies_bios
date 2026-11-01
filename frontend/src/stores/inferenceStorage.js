// Inference results can be tens of megabytes; localStorage is too small.
const DB_NAME='forge-inference-cache';
const STORE='sessions';
function openDatabase(){
  return new Promise((resolve,reject)=>{
    const request=indexedDB.open(DB_NAME,1);
    request.onupgradeneeded=()=>request.result.createObjectStore(STORE);
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(request.error);
  });
}
async function transact(mode,operation){
  const db=await openDatabase();
  try{return await new Promise((resolve,reject)=>{
    const transaction=db.transaction(STORE,mode),request=operation(transaction.objectStore(STORE));
    let result;
    request.onsuccess=()=>{result=request.result;};
    transaction.oncomplete=()=>resolve(result);
    transaction.onerror=()=>reject(transaction.error);
    transaction.onabort=()=>reject(transaction.error||new Error('Inference storage transaction aborted'));
  });}finally{db.close();}
}
export const inferenceStorage={
  async getItem(name){
    const value=await transact('readonly',store=>store.get(name));
    // Read the old format for users whose smaller sessions were saved successfully.
    return value??localStorage.getItem(name);
  },
  async setItem(name,value){await transact('readwrite',store=>store.put(value,name));},
  async removeItem(name){await transact('readwrite',store=>store.delete(name));},
};
