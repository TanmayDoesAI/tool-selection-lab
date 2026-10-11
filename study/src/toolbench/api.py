"""Explicit retries with a durable, run-wide pre-request budget reservation."""
import json,os,random,re,threading,time
import httpx

class LimitReached(RuntimeError):pass
class APIError(RuntimeError):pass

def sanitize(value):
    text=json.dumps(value,ensure_ascii=False)
    for name in ('GROQ_API_KEY','NIM_API_KEY'):
        key=os.environ.get(name)
        if key:text=text.replace(key,'[REDACTED]')
    text=re.sub(r'(gsk_|nvapi-)[A-Za-z0-9_-]+','[REDACTED]',text)
    text=re.sub(r'Bearer [A-Za-z0-9_.-]+','Bearer [REDACTED]',text)
    return json.loads(text)

def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(sanitize(data),indent=2)+'\n'); tmp.replace(path)

PROVIDERS={
    'groq':{'name':'Groq','url':'https://api.groq.com/openai/v1/chat/completions','key':'GROQ_API_KEY','max_tokens':'max_completion_tokens','default_model':'openai/gpt-oss-20b'},
    'nim':{'name':'NVIDIA NIM','url':'https://integrate.api.nvidia.com/v1/chat/completions','key':'NIM_API_KEY','max_tokens':'max_tokens','default_model':'nvidia/nemotron-3-super-120b-a12b'},
}

class GroqClient:
    """OpenAI-compatible chat client; provider (Groq by default, or NVIDIA NIM) comes from the run config."""
    def __init__(self,manifest,save,transport=None,sleep=time.sleep,log=None):
        # log(event) appends an attempt snapshot durably; without it the whole manifest is saved instead.
        self.manifest=manifest; self.save=save; self.sleep=sleep; self.log=log
        self.http=httpx.Client(timeout=manifest['config']['timeout'],transport=transport,trust_env=True)
        self.lock=threading.RLock(); self.local=threading.local()
        # One rate-limit cooldown shared by every worker thread: a 429 pauses all of them, not just one.
        self.cooldown_until=0.0
    @property
    def context(self):return getattr(self.local,'context',{})
    @context.setter
    def context(self,value):self.local.context=value
    def persist(self,event):
        with self.lock:
            if self.log:self.log(event)
            else:self.save()
    def wait_for_cooldown(self,provider,request_number):
        """Sleep out any shared cooldown this thread has not already waited for."""
        with self.lock:target=self.cooldown_until
        waited=getattr(self.local,'waited_until',0.0)
        if target<=waited:return
        remaining=target-time.monotonic();self.local.waited_until=target
        while remaining>0:
            chunk=min(remaining,30)
            print(f'{provider["name"]} retry backoff: {chunk:.1f}s (request {request_number})',flush=True)
            self.sleep(chunk);remaining-=chunk
    def chat(self,messages,tools=None,phase='execution',json_mode=False,force=None):
        cfg=self.manifest['config']; provider=PROVIDERS[cfg.get('provider','groq')]; key=os.getenv(provider['key'])
        if not key:raise APIError(f'{provider["key"]} is not configured')
        body={'model':cfg['model'],'messages':messages,'temperature':0,provider['max_tokens']:1500}
        if tools:body.update(tools=tools,tool_choice=force or 'auto',parallel_tool_calls=False)
        if json_mode:body['response_format']={'type':'json_object'}
        # Execution transcripts are stored once per result; attempts keep tool names, not 100 schemas.
        compact={'tools':[t['function']['name'] for t in tools or []],'n_messages':len(messages),'json_mode':json_mode,'force':force}
        if phase!='execution':compact['messages']=messages
        for retry in range(cfg['retries']+1):
            self.wait_for_cooldown(provider,self.manifest['api_requests']+1)
            with self.lock:
                if self.manifest['api_requests']>=cfg['max_requests']: raise LimitReached('run-wide API request limit reached')
                self.manifest['api_requests']+=1
                event={'request_number':self.manifest['api_requests'],'phase':phase,'retry':retry,'model':cfg['model'],**self.context,'request':sanitize(compact),'usage':None,'outcome':'reserved'}
                self.manifest['attempts'].append(event); self.persist(event) # durable reservation before I/O
            start=time.perf_counter(); response=None
            try:
                response=self.http.post(provider['url'],headers={'Authorization':f'Bearer {key}'},json=body)
                event['seconds']=time.perf_counter()-start; event['http_status']=response.status_code
                if response.is_success:
                    result=response.json(); event.update(outcome='success',usage=result.get('usage'),response=sanitize(result));self.persist(event); return result
                event.update(outcome='error',error=sanitize(response.json()) if response.headers.get('content-type','').startswith('application/json') else {'message':'non-JSON API error'})
                transient=response.status_code in (408,429,500,502,503,504,529)
            except (httpx.TimeoutException,httpx.TransportError) as exc:
                event.update(seconds=time.perf_counter()-start,outcome='error',error={'type':type(exc).__name__});transient=True
            self.persist(event)
            if not transient or retry==cfg['retries']:raise APIError(f'{provider["name"]} request failed: {event.get("http_status",event["error"].get("type"))}')
            header=response.headers.get('retry-after') if response is not None else None
            try:delay=float(header) if header is not None else None
            except ValueError:delay=None
            # No Retry-After: exponential backoff (2, 4, 8 ... s) capped at max_backoff, plus jitter.
            if delay is None:delay=min(cfg.get('max_backoff',60),2**(retry+1))+random.uniform(0,1)
            elif delay>cfg.get('max_backoff',10): raise APIError(f'Rate-limit wait {delay:.1f}s exceeds configured max_backoff; this task stopped')
            event['backoff_seconds']=max(0,delay);self.persist(event)
            with self.lock:self.cooldown_until=max(self.cooldown_until,time.monotonic()+max(0,delay))
            self.wait_for_cooldown(provider,event['request_number'])
        raise AssertionError('unreachable')
