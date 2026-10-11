import json,os,random,threading,time,uuid,hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from jsonschema import ValidationError
from .api import GroqClient,LimitReached,APIError,atomic_json,sanitize
from .tools import TOOLS
from .tasks import load,fixture,verify_freeze
from .strategies import EmbeddingIndex,NimEmbeddingIndex,KeywordIndex,discover,parse_strategy,index_for,agent_search,AGENT_SEARCH,SEARCH_TOOL
from .grading import grade

INSTRUCTIONS='''You are the operations assistant for Northstar Retail. Staff message you about orders, inventory, shipping, customers, billing, support tickets, marketing, the product catalog, returns and reporting. You work through the tools you have been given; they read and change Northstar's live systems.

- Look things up with your tools instead of guessing. Never make up IDs, numbers or outcomes, and only say something is done once a tool has confirmed it.
- If a request is ambiguous (a name matches more than one customer, or an ID, amount or other detail you need is missing and you can't look it up), ask a short clarifying question before changing anything.
- If none of your tools can do what is asked, say so plainly. Don't attempt a workaround.
- Do everything the request asks for, and don't change anything it didn't ask for.
- When you're finished, reply to the staff member in a few sentences with the outcome, including the specific figures or details they asked for.'''

def execute(task,strategy,client,cfg,index):
    start=time.perf_counter();state=fixture(task);calls=[];final=None;error=None;names=[];discovery={};execution_start=None;messages=[]
    try:
        names,discovery=discover(strategy,task['request'],client,cfg,index)
        searching=parse_strategy(strategy)[0] in AGENT_SEARCH
        if searching:discovery['searches']=[]
        messages=[{'role':'system','content':INSTRUCTIONS},{'role':'user','content':task['request']}]
        execution_start=time.perf_counter()
        for step in range(cfg['max_steps']):
            result=client.chat(messages,[TOOLS[n].schema for n in names]+([SEARCH_TOOL] if searching else []))
            msg=result['choices'][0]['message']; msg={k:msg[k] for k in ('role','content','tool_calls') if k in msg};messages.append(msg)
            if not msg.get('tool_calls'):final=msg.get('content');break
            for call in msg['tool_calls']:
                # Some servers leak chat-template tokens into tool names (gpt-oss: "get_order<|channel|>commentary").
                raw=call['function']['name'];name=raw.split('<|')[0].strip()
                if searching and name=='search_tools':
                    try:
                        query=json.loads(call['function']['arguments'])['query']
                        if not isinstance(query,str) or not query.strip():raise ValueError('query must be a non-empty string')
                        output,record=agent_search(strategy,query,names,cfg,index);record['step']=step;discovery['searches'].append(record)
                    except (ValueError,TypeError,KeyError) as exc:output={'error':str(exc).split('\n')[0]};discovery['searches'].append({'step':step,'error':output['error']})
                    messages.append({'role':'tool','tool_call_id':call['id'],'name':name,'content':json.dumps(output)});continue
                entry={'name':name,'arguments':None,'ok':False,'schema_valid':False,'step':step,'mutates':TOOLS[name].mutates if name in TOOLS else False}
                if raw!=name:entry['raw_name']=raw
                try:
                    args=json.loads(call['function']['arguments']);entry['arguments']=args
                    if name not in names:raise ValueError('Tool not available in shortlist')
                    tool=TOOLS[name]; tool.parameters # validation precedes mutation
                    from jsonschema import Draft202012Validator
                    Draft202012Validator(tool.parameters).validate(args);entry['schema_valid']=True
                    output=tool.execute(state,args);entry.update(ok=True,output=output)
                except (ValueError,TypeError,KeyError,ValidationError) as exc:
                    output={'error':str(exc).split('\n')[0]};entry['output']=output
                calls.append(entry)
                messages.append({'role':'tool','tool_call_id':call['id'],'name':name,'content':json.dumps(output)})
        else:error='step_limit'
    except LimitReached:error='request_limit'
    except APIError as exc:error='api_error';discovery['error']=str(exc)
    except (ValueError,KeyError,TypeError) as exc:error='routing_error';discovery['error']=str(exc)
    elapsed=time.perf_counter()-start
    if discovery.get('searches') is not None:discovery.update(shortlist=list(names),size=len(names),seconds=discovery.get('seconds',0)+sum(x.get('seconds',0) for x in discovery['searches']))
    return {'task_id':task['id'],'type':task['type'],'split':task['split'],'request':task['request'],'strategy':strategy,'transcript':messages if execution_start else None,'discovery':discovery,'shortlist':names,'calls':calls,'final':final,'state':state,'error':error,'grading':grade(task,state,calls,final,names,error),'timings':{'end_to_end_seconds':elapsed,'discovery_seconds':discovery.get('seconds',0),'execution_seconds':time.perf_counter()-execution_start if execution_start else 0}}

def protocol_fingerprint():
    base=Path(__file__).parent
    names=('world.py','tools.py','strategies.py','api.py','runner.py','tasks.py','grading.py')
    return {n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}

def create_run(path,cfg,tasks):
    rng=random.Random(cfg['seed']);plan=[]
    for repetition in range(cfg['repetitions']):
        ordered=tasks.copy();rng.shuffle(ordered)
        for t in ordered:
            strategies=cfg['strategies'].copy();rng.shuffle(strategies)
            plan.extend({'task_id':t['id'],'strategy':s,'repetition':repetition} for s in strategies)
    manifest={'run_id':uuid.uuid4().hex,'created_at':datetime.now(timezone.utc).isoformat(),'kind':'live','protocol_fingerprint':protocol_fingerprint(),'config':cfg,'freeze':verify_freeze(),'tasks':tasks,'plan':plan,'api_requests':0,'attempts':[],'results':[],'preflight_done':False,'status':'created','storage':'jsonl'}
    atomic_json(path/'checkpoint.json',manifest);return manifest

class RunStore:
    """Append-only run storage: checkpoint.json (config/plan/status, rewritten rarely),
    attempts.jsonl (every attempt snapshot, fsynced before I/O) and results.jsonl."""
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.Lock()
    def append(self,name,record):
        line=json.dumps(sanitize(record),ensure_ascii=False)+'\n'
        with self.lock,open(self.path/name,'a') as f:
            f.write(line);f.flush();os.fsync(f.fileno())
    def save_meta(self,manifest):
        with self.lock:atomic_json(self.path/'checkpoint.json',{k:v for k,v in manifest.items() if k not in ('attempts','results')}|{'storage':'jsonl'})

def load_run(path):
    """Rebuild a manifest (with attempts and results) from either storage layout."""
    path=Path(path);manifest=json.loads((path/'checkpoint.json').read_text())
    if manifest.get('storage')!='jsonl':return manifest
    attempts={}
    if (path/'attempts.jsonl').exists():
        for line in (path/'attempts.jsonl').read_text().splitlines():
            if line.strip():event=json.loads(line);attempts[event['request_number']]=event  # last snapshot wins
    results=[json.loads(l) for l in (path/'results.jsonl').read_text().splitlines() if l.strip()] if (path/'results.jsonl').exists() else []
    manifest.update(attempts=[attempts[k] for k in sorted(attempts)],results=results,api_requests=max([manifest.get('api_requests',0),*attempts]))
    return manifest

def run(path,manifest,transport=None,index=None,concurrency=None):
    path=Path(path);store=RunStore(path)
    save=lambda:store.save_meta(manifest)
    client=GroqClient(manifest,save,transport=transport,log=lambda e:store.append('attempts.jsonl',e))
    cfg=manifest['config'];workers=concurrency or cfg.get('concurrency',1)
    if manifest.get('protocol_fingerprint') and manifest['protocol_fingerprint']!=protocol_fingerprint():
        manifest.update(status='blocked',blocker='Execution/grading code changed since checkpoint; start a new run rather than mix protocols');save();return manifest
    manifest['status']='running';save()
    bases={index_for(parse_strategy(s)[0]) for s in cfg['strategies']}
    if index is None and {'semantic','semantic-nim','keyword'}&bases:
        index={}
        try:
            if 'semantic' in bases:index['semantic']=EmbeddingIndex(cache=path.parent/'.cache',model=cfg['embedding_model']);manifest['embedding']=index['semantic'].metadata;save()
            if 'keyword' in bases:index['keyword']=KeywordIndex();manifest['embedding_keyword']=index['keyword'].metadata;save()
            if 'semantic-nim' in bases:index['semantic-nim']=NimEmbeddingIndex(cache=path.parent/'.cache',model=cfg.get('nim_embedding_model','nvidia/nemotron-3-embed-1b'),timeout=cfg['timeout'],retries=cfg['retries']);manifest['embedding_nim']=index['semantic-nim'].metadata;save()
        except (OSError,ValueError,APIError) as exc:manifest.update(status='blocked',blocker=f'Embedding setup: {exc}');save();return manifest
    if not manifest['preflight_done']:
        try:
            check=client.chat([{'role':'user','content':'Use get_order for O100.'}],[TOOLS['get_order'].schema],phase='compatibility',force={'type':'function','function':{'name':'get_order'}})
            tc=check['choices'][0]['message'].get('tool_calls',[])
            if not tc or tc[0]['function']['name']!='get_order':raise APIError('Model did not return the forced compatibility tool call')
            from jsonschema import Draft202012Validator
            Draft202012Validator(TOOLS['get_order'].parameters).validate(json.loads(tc[0]['function']['arguments']))
            manifest['preflight_done']=True;save()
        except (APIError,LimitReached,ValueError,KeyError,ValidationError) as exc:manifest.update(status='blocked',blocker=str(exc));save();return manifest
    done={x['job_key'] for x in manifest['results']};tasks={t['id']:t for t in manifest['tasks']}
    pending=[(f'{j["task_id"]}:{j["strategy"]}:{j["repetition"]}',j) for j in manifest['plan']]
    pending=[(k,j) for k,j in pending if k not in done]
    stop=threading.Event();lock=threading.Lock()
    def work(item):
        key,job=item
        if stop.is_set():return
        if manifest['api_requests']>=cfg['max_requests']:stop.set();return
        client.context={'job_key':key}
        row=execute(tasks[job['task_id']],job['strategy'],client,cfg,index)
        with client.lock:attempts=[e for e in manifest['attempts'] if e.get('job_key')==key]
        row.update(job_key=key,repetition=job['repetition'],run_id=manifest['run_id'],seed=cfg['seed'],model=cfg['model'],configuration=cfg,attempt_numbers=[e['request_number'] for e in attempts],tokens={'input':sum((e.get('usage') or {}).get('prompt_tokens',0) for e in attempts),'output':sum((e.get('usage') or {}).get('completion_tokens',0) for e in attempts)},usage_unknown_attempts=sum(e.get('usage') is None for e in attempts))
        with lock:
            # A budget-interrupted task is restartable with an explicitly increased total cap.
            if row['error']=='request_limit':manifest.setdefault('interrupted_results',[]).append(row);manifest['interrupted_result']=row;stop.set();return
            manifest['results'].append(row);store.append('results.jsonl',row)
            print(f'{len(manifest["results"])}/{len(manifest["plan"])} {key}: completed={row["grading"]["completed"]} requests={manifest["api_requests"]}',flush=True)
    if workers<=1:
        for item in pending:
            work(item)
            if stop.is_set():break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(work,pending))
    manifest['status']='request_limit' if stop.is_set() else 'completed'
    if isinstance(index,dict) and 'semantic-nim' in index:
        nim=index['semantic-nim'];manifest['embedding_nim'].update(total_requests=nim.requests,query_seconds=nim.query_seconds)
    save();return manifest
