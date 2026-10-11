"""Run every case under every strategy with mocked tool execution; results go to an append-only, resumable JSONL."""
import hashlib,json,os,random,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from jsonschema import Draft202012Validator
from .http import APIError
from .grading import grade
from . import strategies as S

INSTRUCTIONS='''You are an assistant that completes requests by calling the tools you have been given.
- Use tools to look things up instead of guessing; never invent IDs, numbers or outcomes.
- If a request is ambiguous or missing a detail you cannot look up, ask a short clarifying question before changing anything.
- If none of your tools can do what is asked, say so plainly instead of attempting a workaround.
- Do everything the request asks, and nothing it did not ask for. Finish with a short reply describing the outcome.'''

def fingerprint(catalog,cases):
    return hashlib.sha256(json.dumps([catalog,cases],sort_keys=True).encode()).hexdigest()[:16]

def mock_result(case,mocks,name,count):
    """The stub a tool call returns: case mocks, then file mocks, else {"ok": true}."""
    for table in (case.get('mocks') or {},mocks or {}):
        if name in table:
            value=table[name]
            if isinstance(value,dict) and '$sequence' in value:
                seq=value['$sequence'];return seq[min(count,len(seq)-1)] if seq else {'ok':True}
            return value
    return {'ok':True}

def run_case(case,strategy,ctx,mocks=None,max_steps=8,instructions=None):
    start=time.perf_counter();calls=[];final=None;error=None;names=[];info={};turns=0
    tokens_in=0;tokens_out=0;unknown_usage=False;counts={}
    validators={n:Draft202012Validator(t['parameters']) for n,t in ctx.tools.items()}
    phase='discovery'
    try:
        names,info=S.discover(strategy,case['prompt'],ctx);phase='execution'
        if info.get('categories') is not None:
            turns+=1
            if info.get('routing_input_tokens') is None:unknown_usage=True
            tokens_in+=info.get('routing_input_tokens') or 0;tokens_out+=info.get('routing_output_tokens') or 0
        exposed=set(names);search=S.searching(strategy);searches=[]
        messages=[{'role':'system','content':instructions or INSTRUCTIONS},{'role':'user','content':case['prompt']}]
        for step in range(max_steps):
            tools=[ctx.tools[n] for n in names]+([S.search_tool(len(ctx.tools),ctx.categories)] if search else [])
            turn=ctx.agent.step(messages,tools);turns+=1
            if turn.input_tokens is None:unknown_usage=True
            tokens_in+=turn.input_tokens or 0;tokens_out+=turn.output_tokens or 0
            messages.append({'role':'assistant','content':turn.text,'tool_calls':turn.tool_calls,'raw':turn.raw})
            if not turn.tool_calls:final=turn.text or '';break
            for c in turn.tool_calls:
                if search and c.name=='search_tools':
                    query=c.arguments.get('query') if isinstance(c.arguments,dict) else None
                    if isinstance(query,str) and query.strip():
                        output,added=S.search(strategy,query,names,ctx);exposed.update(added);searches.append({'step':step,'query':query,'added':added})
                    else:output={'error':'search_tools needs a non-empty "query" string'}
                else:
                    entry={'step':step,'name':c.name,'arguments':c.arguments if isinstance(c.arguments,dict) else None,'valid':False}
                    if not isinstance(c.arguments,dict):output={'error':'arguments were not a JSON object'}
                    elif c.name not in ctx.tools:output={'error':f'unknown tool {c.name}'}
                    elif c.name not in names:output={'error':f'tool {c.name} is not available'};entry['note']='not_available'
                    else:
                        problems=[p.message for p in validators[c.name].iter_errors(c.arguments)]
                        if problems:output={'error':'invalid arguments: '+'; '.join(problems)[:300]};entry['note']='schema_invalid'
                        else:
                            entry['valid']=True;output=mock_result(case,mocks,c.name,counts.get(c.name,0));counts[c.name]=counts.get(c.name,0)+1
                    calls.append(entry)
                messages.append({'role':'tool','tool_call_id':c.id,'name':c.name,'content':json.dumps(output)})
        else:error='step_limit'
        if search:info['searches']=searches
    except APIError as exc:error='api_error';info['error']=str(exc)[:500]
    except ValueError as exc:error='routing_error' if phase=='discovery' else 'agent_error';info['error']=str(exc)[:500]
    except Exception as exc:error='agent_error';info['error']=f'{type(exc).__name__}: {exc}'[:500]  # a broken adapter fails the job, not the run
    exposed=sorted(set(names)|set(info.get('initial',[])))
    return {'case_id':case['id'],'type':case['type'],'category':case.get('category'),'strategy':strategy,
            'shortlist_size':len(info.get('initial',[])),'exposed_size':len(exposed),'discovery':info,'calls':calls,'final':final,'error':error,
            'model_turns':turns,'input_tokens':None if unknown_usage else tokens_in,'output_tokens':None if unknown_usage else tokens_out,
            'seconds':round(time.perf_counter()-start,3),'grade':grade(case,calls,final,error,exposed)}

class Store:
    """results.jsonl is append-only; a job is done once a line for its key exists. run.json records the setup."""
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.path=self.out/'results.jsonl';self.lock=threading.Lock()
    def rows(self):
        rows=[]
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except ValueError:pass  # a torn last line after a crash is ignored and its job re-run
        return rows
    def append(self,row):
        with self.lock,open(self.path,'a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())

def key(row):return f'{row["case_id"]}|{row["strategy"]}|{row["repetition"]}'

def run(catalog,cases,strategies,ctx,out,meta,repetitions=1,concurrency=4,max_steps=8,seed=0,retry_errors=False,log=print):
    store=Store(out);fp=fingerprint(catalog,cases);run_json=store.out/'run.json'
    setup={'fingerprint':fp,'strategies':list(strategies),'repetitions':repetitions,'max_steps':max_steps,'k':ctx.k,'search_k':ctx.search_k,'category_limit':ctx.category_limit,**meta}
    if run_json.exists():
        old=json.loads(run_json.read_text())
        if old.get('fingerprint')!=fp:raise SystemExit(f'{out} holds results for a different catalog/cases; use a new --out')
        for k in ('model','provider','agent','max_steps'):
            if old.get(k)!=setup.get(k):raise SystemExit(f'{out} was run with {k}={old.get(k)!r}; use a new --out to change it')
        setup['strategies']=list(dict.fromkeys(old['strategies']+list(strategies)));setup['repetitions']=max(old['repetitions'],repetitions)
    setup['updated']=datetime.now(timezone.utc).isoformat(timespec='seconds');run_json.write_text(json.dumps(setup,indent=1))
    done={key(r) for r in store.rows() if not (retry_errors and r.get('error')=='api_error')}
    rng=random.Random(seed);jobs=[]
    for rep in range(repetitions):
        for case in cases['cases']:
            order=list(strategies);rng.shuffle(order)  # interleave strategies so drift in the endpoint hits all of them alike
            jobs+=[(case,s,rep) for s in order if f'{case["id"]}|{s}|{rep}' not in done]
    log(f'{len(jobs)} jobs to run ({len(done)} already done) with concurrency {concurrency}')
    def work(job):
        case,s,rep=job;row=run_case(case,s,ctx,cases.get('mocks'),max_steps,catalog.get('instructions'));row['repetition']=rep
        store.append(row);return row
    finished=0
    with ThreadPoolExecutor(max(1,concurrency)) as pool:
        for fut in as_completed([pool.submit(work,j) for j in jobs]):
            row=fut.result();finished+=1
            mark='ok ' if row['grade']['success'] else (row['grade']['reason'] or 'fail')
            log(f'[{finished}/{len(jobs)}] {row["strategy"]:<16} {row["case_id"]:<12} {mark}')
    return store.rows()
