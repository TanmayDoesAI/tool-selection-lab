import hashlib,json,time
from pathlib import Path
import os
import httpx
import numpy as np
from .tools import TOOLS,CATEGORIES

class EmbeddingIndex:
    """Pretrained static semantic embeddings; never silently substitutes lexical search."""
    def __init__(self,cache='.cache',model='en_core_web_md'):
        import spacy
        start=time.perf_counter()
        self.nlp=spacy.load(model,exclude=['parser','ner','tagger','lemmatizer','attribute_ruler'])
        if not self.nlp.vocab.vectors_length: raise ValueError('Embedding model contains no pretrained vectors')
        self.load_seconds=time.perf_counter()-start
        self.names=list(TOOLS)
        documents=[json.dumps(TOOLS[n].schema['function'],sort_keys=True).replace('_',' ') for n in self.names]
        identity={'documents':documents,'model':model,'version':self.nlp.meta['version'],'spacy':spacy.__version__,'vectors':self.nlp.vocab.vectors_length}
        self.fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=Path(cache)/f'{self.fingerprint}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter(); self.cache_hit=path.exists()
        if self.cache_hit: self.matrix=np.load(path)['matrix']
        else:
            self.matrix=np.stack([self.encode(text) for text in documents])
            np.savez_compressed(path,matrix=self.matrix)
        self.prepare_seconds=time.perf_counter()-start
        self.metadata={k:getattr(self,k) for k in ('fingerprint','load_seconds','prepare_seconds','cache_hit')}
        self.metadata.update(model=model,version=self.nlp.meta['version'],dimensions=self.nlp.vocab.vectors_length,encoder='mean pretrained spaCy word vectors; L2 normalization')
    def encode(self,text):
        doc=self.nlp.make_doc(text)
        vectors=[t.vector for t in doc if t.has_vector and not t.is_punct and not t.is_space]
        vec=np.mean(vectors,axis=0) if vectors else np.zeros(self.nlp.vocab.vectors_length)
        return vec/max(float(np.linalg.norm(vec)),1e-12)
    def retrieve(self,request,k):
        if not 1<=k<=len(self.names): raise ValueError('top-k must be 1..100')
        scores=self.matrix@self.encode(request)
        order=np.argsort(-scores,kind='stable')[:k]
        return [self.names[i] for i in order], [{'tool':self.names[i],'score':float(scores[i])} for i in order]

NIM_URL='https://integrate.api.nvidia.com/v1/embeddings'

class NimEmbeddingIndex:
    """NVIDIA NIM sentence embeddings (asymmetric passage/query); cached tool vectors, separate from the Groq budget."""
    def __init__(self,cache='.cache',model='nvidia/nemotron-3-embed-1b',timeout=30,retries=2,transport=None,sleep=time.sleep):
        self.model=model;self.retries=retries;self.sleep=sleep;self.requests=0;self.query_seconds=[]
        import threading
        self.cache_lock=threading.Lock();self.cache_hits=0
        self.query_cache_path=Path(cache)/f'nim-queries-{hashlib.sha256(model.encode()).hexdigest()[:12]}.jsonl'
        self.query_cache={}
        if self.query_cache_path.exists():
            for line in self.query_cache_path.read_text().splitlines():
                try:row=json.loads(line);self.query_cache[row['key']]=row['vec']
                except (ValueError,KeyError):pass  # a torn last line from a crash is skipped
        self.http=httpx.Client(timeout=timeout,transport=transport,trust_env=True)
        self.names=list(TOOLS)
        documents=[json.dumps(TOOLS[n].schema['function'],sort_keys=True).replace('_',' ') for n in self.names]
        self.fingerprint=hashlib.sha256(json.dumps({'documents':documents,'model':model,'provider':'nvidia-nim','input_type':'passage'},sort_keys=True).encode()).hexdigest()
        path=Path(cache)/f'{self.fingerprint}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter(); self.cache_hit=path.exists()
        if self.cache_hit: self.matrix=np.load(path)['matrix']
        else:
            self.matrix=np.concatenate([self.embed(documents[i:i+25],'passage') for i in range(0,len(documents),25)])
            np.savez_compressed(path,matrix=self.matrix)
        self.prepare_seconds=time.perf_counter()-start
        self.metadata={'fingerprint':self.fingerprint,'prepare_seconds':self.prepare_seconds,'cache_hit':self.cache_hit,'model':model,'provider':'NVIDIA NIM','dimensions':int(self.matrix.shape[1]),'encoder':'NIM embeddings API; passage/query input types; L2 normalization','index_requests':self.requests}
    def embed(self,texts,input_type):
        from .api import APIError
        key=os.getenv('NIM_API_KEY')
        if not key:raise APIError('NIM_API_KEY is not configured')
        for retry in range(self.retries+1):
            self.requests+=1
            try:response=self.http.post(NIM_URL,headers={'Authorization':f'Bearer {key}'},json={'model':self.model,'input':texts,'input_type':input_type,'encoding_format':'float','truncate':'END'})
            except httpx.HTTPError as exc:error=f'NIM transport error: {type(exc).__name__}'
            else:
                if response.is_success:
                    vectors=np.array([d['embedding'] for d in sorted(response.json()['data'],key=lambda d:d['index'])],dtype=np.float32)
                    return vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
                error=f'NIM HTTP {response.status_code}'
                if response.status_code not in (408,409,429) and response.status_code<500:break
            if retry<self.retries:self.sleep(min(2**retry,10))
        raise APIError(error)
    def encode(self,text):
        """Query embeddings are cached on disk by (model, text): the same request is embedded once and
        reused across models, arms (top-k, hybrid, k-sweep) and runs, so retrieval is identical for all."""
        start=time.perf_counter();key=hashlib.sha256(f'{self.model}|query|{text}'.encode()).hexdigest()
        with self.cache_lock:hit=self.query_cache.get(key)
        if hit is not None:self.cache_hits+=1;vec=np.asarray(hit,dtype=np.float32)
        else:
            vec=self.embed([text],'query')[0]
            with self.cache_lock:
                self.query_cache[key]=vec.tolist()
                with open(self.query_cache_path,'a') as f:f.write(json.dumps({'key':key,'vec':[round(float(x),7) for x in vec]})+'\n')
        self.query_seconds.append(time.perf_counter()-start);return vec
    retrieve=EmbeddingIndex.retrieve

class KeywordIndex:
    """BM25 over tool name, description and parameter names; a lexical baseline with no model."""
    def __init__(self,k1=1.5,b=0.75):
        import re
        self.tokenize=lambda text:re.findall(r'[a-z0-9]+',text.lower().replace('_',' '))
        self.names=list(TOOLS)
        docs=[self.tokenize(' '.join([n,TOOLS[n].schema['function']['description'],*TOOLS[n].parameters['properties']])) for n in self.names]
        self.lengths=np.array([len(d) for d in docs],dtype=float);self.k1=k1;self.b=b;self.avg=self.lengths.mean()
        self.tf=[{t:d.count(t) for t in set(d)} for d in docs]
        df={}
        for d in self.tf:
            for t in d:df[t]=df.get(t,0)+1
        self.idf={t:float(np.log(1+(len(docs)-v+0.5)/(v+0.5))) for t,v in df.items()}
        self.metadata={'encoder':'BM25 (k1=1.5, b=0.75) over name+description+parameter names','vocabulary':len(self.idf)}
    def retrieve(self,request,k):
        if not 1<=k<=len(self.names): raise ValueError('top-k must be 1..100')
        terms=set(self.tokenize(request))
        scores=np.array([sum(self.idf[t]*tf[t]*(self.k1+1)/(tf[t]+self.k1*(1-self.b+self.b*length/self.avg)) for t in terms if t in tf) for tf,length in zip(self.tf,self.lengths)])
        order=np.argsort(-scores,kind='stable')[:k]
        return [self.names[i] for i in order],[{'tool':self.names[i],'score':float(scores[i])} for i in order]

RETRIEVERS=('semantic','semantic-nim','keyword')
# Agent-driven discovery: the model starts with only search_tools and adds what it finds, step by step.
AGENT_SEARCH={'agent-search':'semantic-nim','agent-search-kw':'keyword','hybrid':'semantic-nim'}
HYBRID_K=10  # hybrid starts with the semantic-nim top-10 shortlist, then can search for more
SEARCH_TOOL={'type':'function','function':{'name':'search_tools','description':'Search the full Northstar tool catalogue (100 tools: orders, inventory, shipping, customers, billing, support, marketing, catalog, returns, analytics). The tools you can currently call are only the ones you have found so far. Describe one capability you need in your own words, e.g. "find the invoice for an order" or "refund a payment". The closest matches become callable from your next step. Search again, with different words, for each further capability you need.','parameters':{'type':'object','properties':{'query':{'type':'string','minLength':1,'description':'The capability you need'}},'required':['query'],'additionalProperties':False}}}

def index_for(base):
    """Which retrieval index a strategy needs, if any."""
    return AGENT_SEARCH.get(base,base if base in RETRIEVERS else None)

def parse_strategy(strategy):
    """'semantic-nim@5' -> ('semantic-nim', 5); a bare name uses the configured top-k."""
    base,_,k=strategy.partition('@')
    if k and (base not in RETRIEVERS+tuple(AGENT_SEARCH) or not k.isdigit() or not 1<=int(k)<=100):raise ValueError(f'Invalid strategy: {strategy}')
    if base not in ('all','category','category-laya')+RETRIEVERS+tuple(AGENT_SEARCH):raise ValueError(f'Unknown strategy: {strategy}')
    return base,int(k) if k else None

LAYA_ROUTES=Path(__file__).resolve().parents[2]/'data'/'laya_routes.json'

def laya_categories(request):
    """Top-2 categories from the Laya decision model, precomputed on the Pi CPU and keyed by the request text only."""
    routes=json.loads(LAYA_ROUTES.read_text())
    key=hashlib.sha256(request.encode()).hexdigest()
    if key not in routes['routes']:raise ValueError('No cached Laya route for this request; run the Laya router first')
    return routes['routes'][key],routes['meta']

def discover(strategy,request,client,config,index=None):
    start=time.perf_counter(); categories=[]; ranking=[]
    base,k=parse_strategy(strategy)
    if strategy=='all': names=list(TOOLS)
    elif base in RETRIEVERS:names,ranking=(index[base] if isinstance(index,dict) else index).retrieve(request,k or config['top_k'])
    elif base=='category-laya':
        categories,meta=laya_categories(request)
        names=[n for n,t in TOOLS.items() if t.category in categories]
        start-=meta['median_seconds_per_request']  # charge the measured CPU routing latency (routes are cached)
    elif base=='hybrid':names,ranking=(index['semantic-nim'] if isinstance(index,dict) else index).retrieve(request,HYBRID_K)
    elif base in AGENT_SEARCH:names=[]  # nothing up front; execute() grows the shortlist from search_tools calls
    elif strategy=='category':
        result=client.chat([{'role':'system','content':f'Select 1 to {config["category_limit"]} retail tool categories needed for the request. Use only these descriptions. Respond with JSON {{"categories": [names]}}.\n'+json.dumps(CATEGORIES)},{'role':'user','content':request}],phase='routing',json_mode=True)
        categories=json.loads(result['choices'][0]['message']['content'])['categories']
        if not isinstance(categories,list) or not 1<=len(categories)<=config['category_limit'] or len(set(categories))!=len(categories) or any(c not in CATEGORIES for c in categories): raise ValueError('Invalid router selection')
        names=[n for n,t in TOOLS.items() if t.category in categories]
    else:raise ValueError(f'Unknown strategy: {strategy}')
    return names,{'categories':categories,'ranking':ranking,'shortlist':list(names),'size':len(names),'seconds':time.perf_counter()-start}

def agent_search(strategy,query,names,config,index):
    """Run one search_tools call: add the top-k matches to the callable set and describe them to the model."""
    base,k=parse_strategy(strategy);start=time.perf_counter()
    found,ranking=(index[AGENT_SEARCH[base]] if isinstance(index,dict) else index).retrieve(query,k or config.get('search_k',5))
    added=[n for n in found if n not in names];names.extend(added)
    output={'tools':[{'name':n,'description':TOOLS[n].schema['function']['description']} for n in found],'note':f'These are the {len(found)} closest of {len(TOOLS)} tools and are now callable. If none of them does what you need, search again with different words.'}
    return output,{'query':query,'results':found,'added':added,'ranking':ranking,'seconds':time.perf_counter()-start}
