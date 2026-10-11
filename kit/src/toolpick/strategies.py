"""Discovery strategies: which tools the agent sees for a request, and (for search strategies) how it finds more.

  all             every tool in the catalog, every turn
  category        the model first picks 1..N categories (one extra call); it then sees those categories' tools
  keyword@k       BM25 top-k over name + description + parameters, fixed from the request
  semantic@k      embedding top-k, fixed from the request (needs an embedding provider)
  agent-search    the agent starts with only search_tools and adds what it finds (@k = results per search)
  hybrid          top-k shortlist up front (semantic if available, else keyword) plus search_tools
Optional (only when named in --strategies):
  agent-search-kw agent-search that always searches with BM25
  hybrid-kw       hybrid with a BM25 shortlist and BM25 search
  spacy@k         top-k by mean spaCy word vectors        (pip install 'toolpick[spacy]')
  category-laya   a local Laya model picks the categories (pip install 'toolpick[laya]')
"""
import json,time

BASES=('all','category','keyword','semantic','agent-search','hybrid')
OPTIONAL=('agent-search-kw','hybrid-kw','spacy','category-laya')
SEARCHING=('agent-search','hybrid','agent-search-kw','hybrid-kw')
DEFAULT=('all','category','keyword@10','semantic@10','agent-search','hybrid')

def parse(strategy):
    base,_,k=strategy.partition('@')
    if base not in BASES+OPTIONAL:raise ValueError(f'unknown strategy {strategy!r}; choose from {", ".join(BASES+OPTIONAL)} (@k for keyword, semantic, spacy and the search strategies)')
    if k and (base in ('all','category','category-laya') or not k.isdigit() or int(k)<1):raise ValueError(f'invalid strategy {strategy!r}')
    return base,int(k) if k else None

def search_tool(n,categories):
    area=f' ({", ".join(sorted(categories))})' if categories else ''
    return {'name':'search_tools','description':f'Search the full catalog of {n} tools{area}. You can only call tools you have found so far. Describe one capability you need in your own words, e.g. "look up an order" or "refund a payment". The closest matches become callable from your next step. Search again, with different words, for each further capability you need.',
            'parameters':{'type':'object','properties':{'query':{'type':'string','minLength':1,'description':'The capability you need'}},'required':['query'],'additionalProperties':False}}

class Context:
    """Everything strategies need, shared by all jobs in a run (indexes are built once)."""
    def __init__(self,catalog,agent,k=10,search_k=5,category_limit=2,keyword=None,semantic=None,spacy=None,laya=None):
        self.catalog=catalog;self.tools={t['name']:t for t in catalog['tools']};self.agent=agent
        self.k=k;self.search_k=search_k;self.category_limit=category_limit;self.keyword=keyword;self.semantic=semantic
        self.spacy=spacy;self.laya=laya   # optional indexes/routers, built only when their strategy is requested
        cats={}
        for t in catalog['tools']:
            if t.get('category'):cats.setdefault(t['category'],[]).append(t['name'])
        self.categories=cats
    def index(self,base):
        """Shortlist index for a strategy; search strategies prefer embeddings and fall back to keywords."""
        if base in ('keyword','agent-search-kw','hybrid-kw'):return self.keyword
        if base=='spacy':return self.spacy
        if base=='semantic':
            if not self.semantic:raise ValueError('semantic strategies need an embedding provider (--embed-provider)')
            return self.semantic
        return self.semantic or self.keyword
    def available(self,strategy):
        """Why a strategy cannot run on this setup, or None."""
        base,_=parse(strategy)
        if base in ('category','category-laya') and not self.categories:return 'catalog has no categories'
        if base=='spacy' and not self.spacy:return "spaCy is not set up (pip install 'toolpick[spacy]')"
        if base=='category-laya' and not self.laya:return "Laya is not set up (pip install 'toolpick[laya]')"
        if base=='semantic' and not self.semantic:return 'no embedding provider configured'
        return None

def route(ctx,request):
    """Category strategy: one model call that picks categories via a choose_categories tool."""
    lines='\n'.join(f'- {c}: {ctx.catalog.get("categories",{}).get(c) or "tools: "+", ".join(ns)}' for c,ns in ctx.categories.items())
    tool={'name':'choose_categories','description':f'Choose the tool categories needed for the request.\n{lines}',
          'parameters':{'type':'object','properties':{'categories':{'type':'array','items':{'enum':list(ctx.categories)},'minItems':1,'maxItems':ctx.category_limit}},'required':['categories'],'additionalProperties':False}}
    messages=[{'role':'system','content':f'Pick 1 to {ctx.category_limit} tool categories that the request needs, by calling choose_categories. Do not answer the request itself.'},{'role':'user','content':request}]
    turn=ctx.agent.step(messages,[tool]);chosen=None
    for c in turn.tool_calls:
        if c.name=='choose_categories' and isinstance(c.arguments,dict):chosen=c.arguments.get('categories')
    if chosen is None and turn.text:  # a model that answered in text: accept JSON or category names it mentions
        try:chosen=json.loads(turn.text).get('categories')
        except (ValueError,AttributeError):chosen=[c for c in ctx.categories if c in turn.text]
    if isinstance(chosen,str):chosen=[chosen]
    if not isinstance(chosen,list):chosen=[]
    chosen=[c for c in dict.fromkeys(chosen) if c in ctx.categories][:ctx.category_limit]
    if not chosen:raise ValueError('router chose no valid category')
    return chosen,turn

def discover(strategy,request,ctx):
    """Initial tool names for a request plus a record of how they were chosen (and any routing usage)."""
    base,k=parse(strategy);start=time.perf_counter();info={'strategy':strategy}
    if base=='all':names=list(ctx.tools)
    elif base=='category':
        chosen,turn=route(ctx,request);names=[n for c in chosen for n in ctx.categories[c]]
        info.update(categories=chosen,routing_input_tokens=turn.input_tokens,routing_output_tokens=turn.output_tokens)
    elif base=='category-laya':
        chosen,probs=ctx.laya.route(request,ctx.category_limit);names=[n for c in chosen for n in ctx.categories[c]]
        info.update(categories=chosen,router=ctx.laya.kind,scores={c:round(v,4) for c,v in probs.items()})
    elif base in ('keyword','semantic','spacy'):
        idx=ctx.index(base);names,scores=idx.retrieve(request,min(k or ctx.k,len(ctx.tools)));info['scores']=scores
    elif base in ('hybrid','hybrid-kw'):
        idx=ctx.index(base);names,_=idx.retrieve(request,min(k or ctx.k,len(ctx.tools)));info['index']=idx.kind
    else:names=[];info['index']=ctx.index(base).kind
    info.update(initial=list(names),seconds=round(time.perf_counter()-start,4))
    return names,info

def searching(strategy):return parse(strategy)[0] in SEARCHING

def search(strategy,query,names,ctx):
    """Handle one search_tools call: make the top matches callable and describe them to the model."""
    base,k=parse(strategy);per=ctx.search_k if base in ('hybrid','hybrid-kw') else (k or ctx.search_k)
    found,_=ctx.index(base).retrieve(query,min(per,len(ctx.tools)))
    added=[n for n in found if n not in names];names.extend(added)
    return {'tools':[{'name':n,'description':ctx.tools[n]['description']} for n in found],
            'note':f'These are the {len(found)} closest of {len(ctx.tools)} tools and are now callable. If none fits, search again with different words.'},added
