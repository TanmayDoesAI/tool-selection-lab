"""Shortlist retrieval over the catalog: BM25 keywords (no model) and embeddings (any OpenAI-style /embeddings API)."""
import hashlib,json,math,threading
from pathlib import Path
import httpx
from .agents import words,PROVIDERS,provider_settings
from .http import post_json

def tool_text(t):
    """What a tool is indexed by: name, description and parameter names/descriptions."""
    params=t['parameters'].get('properties',{})
    return ' '.join([t['name'].replace('_',' '),t['description'],*(f'{k} {v.get("description","") if isinstance(v,dict) else ""}' for k,v in params.items())])

class KeywordIndex:
    """BM25 (k1=1.5, b=0.75) over tool name + description + parameters."""
    kind='keyword'
    def __init__(self,tools,k1=1.5,b=0.75):
        self.names=[t['name'] for t in tools];docs=[words(tool_text(t)) for t in tools]
        self.k1,self.b=k1,b;self.lengths=[len(d) for d in docs];self.avg=sum(self.lengths)/max(1,len(docs)) or 1
        self.tf=[{w:d.count(w) for w in set(d)} for d in docs];df={}
        for d in self.tf:
            for w in d:df[w]=df.get(w,0)+1
        self.idf={w:math.log(1+(len(docs)-v+0.5)/(v+0.5)) for w,v in df.items()}
    def scores(self,query):
        q=set(words(query))
        return [sum(self.idf[w]*tf[w]*(self.k1+1)/(tf[w]+self.k1*(1-self.b+self.b*n/self.avg)) for w in q if w in tf) for tf,n in zip(self.tf,self.lengths)]
    def retrieve(self,query,k):
        s=self.scores(query);order=sorted(range(len(s)),key=lambda i:(-s[i],i))[:k]
        return [self.names[i] for i in order],[round(s[i],4) for i in order]

def normalise(v):
    n=math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

class Embedder:
    """Embeddings from OpenAI, NVIDIA NIM or any OpenAI-compatible /embeddings endpoint, cached on disk by text."""
    def __init__(self,provider='openai',model=None,base_url=None,cache_dir='.toolpick-cache',timeout=60,retries=6,transport=None,sleep=None,dimensions=None):
        p=PROVIDERS.get(provider,{});model=model or p.get('embed_model')
        if not model:raise ValueError(f'{provider}: set --embed-model')
        self.s=provider_settings(provider,model,base_url)
        self.provider=provider;self.model=self.s['model'];self.url=self.s['base_url']+'/embeddings'
        self.headers={'Authorization':f'Bearer {self.s["api_key"]}'} if self.s['api_key'] else {}
        self.http=httpx.Client(timeout=timeout,transport=transport);self.retry={'retries':retries,**({'sleep':sleep} if sleep else {})}
        self.dimensions=dimensions
        self.lock=threading.Lock();self.path=Path(cache_dir)/f'embeddings-{hashlib.sha256(f"{self.url}|{self.model}|{dimensions or ""}".encode()).hexdigest()[:12]}.jsonl'
        self.cache={}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:row=json.loads(line);self.cache[row['key']]=row['vec']
                except (ValueError,KeyError):pass
    def embed(self,texts,kind):
        """kind is 'passage' (tools) or 'query' (requests); NIM retrieval models embed the two differently."""
        keys=[hashlib.sha256(f'{kind}|{t}'.encode()).hexdigest() for t in texts]
        with self.lock:todo=[(k,t) for k,t in zip(keys,texts) if k not in self.cache]
        for i in range(0,len(todo),32):
            batch=todo[i:i+32];body={'model':self.model,'input':[t for _,t in batch]}
            if self.provider=='nim':body.update(input_type=kind,truncate='END')
            if self.dimensions:body['dimensions']=self.dimensions
            data=post_json(self.http,self.url,self.headers,body,**self.retry)
            vecs=[normalise(d['embedding']) for d in sorted(data['data'],key=lambda d:d['index'])]
            with self.lock:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with open(self.path,'a') as f:
                    for (k,_),v in zip(batch,vecs):self.cache[k]=v;f.write(json.dumps({'key':k,'vec':[round(x,6) for x in v]})+'\n')
        with self.lock:return [self.cache[k] for k in keys]

class HashEmbedder:
    """Offline stand-in for an embedding model (bag of hashed words). Used by --dry-run and the tests."""
    provider='fake';model='hash-256'
    def embed(self,texts,kind):
        out=[]
        for t in texts:
            v=[0.0]*256
            for w in words(t):v[int(hashlib.md5(w.encode()).hexdigest(),16)%256]+=1
            out.append(normalise(v))
        return out

class EmbeddingIndex:
    """Cosine top-k between the request embedding and each tool's embedding."""
    kind='semantic'
    def __init__(self,tools,embedder):
        self.names=[t['name'] for t in tools];self.embedder=embedder
        self.matrix=embedder.embed([tool_text(t) for t in tools],'passage')
    def retrieve(self,query,k):
        q=self.embedder.embed([query],'query')[0]
        try:
            import numpy as np
            s=(np.asarray(self.matrix)@np.asarray(q)).tolist()
        except ImportError:s=[sum(a*b for a,b in zip(row,q)) for row in self.matrix]
        order=sorted(range(len(s)),key=lambda i:(-s[i],i))[:k]
        return [self.names[i] for i in order],[round(s[i],4) for i in order]
