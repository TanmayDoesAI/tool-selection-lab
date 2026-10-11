"""The Agent interface and the built-in adapters (OpenAI-compatible chat, Anthropic Messages, an offline fake).

Every adapter speaks one neutral message format, so strategies, the run loop and grading never see provider details:
  {'role': 'system'|'user', 'content': str}
  {'role': 'assistant', 'content': str|None, 'tool_calls': [ToolCall, ...], 'raw': <provider payload, optional>}
  {'role': 'tool', 'tool_call_id': str, 'name': str, 'content': str}
Tools are neutral too: {'name', 'description', 'parameters'} (parameters is a JSON Schema object).
"""
import importlib,importlib.util,json,os,re,sys
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Protocol,runtime_checkable
import httpx
from .http import post_json,APIError

@dataclass
class ToolCall:
    id:str
    name:str
    arguments:Any  # a dict when the model sent valid JSON, else the raw string

@dataclass
class Turn:
    text:str|None=None
    tool_calls:list=field(default_factory=list)
    input_tokens:int|None=None
    output_tokens:int|None=None
    raw:Any=None  # provider-specific assistant payload to send back verbatim (e.g. Anthropic thinking blocks)

@runtime_checkable
class Agent(Protocol):
    """Implement this to test your own harness: one model turn given the conversation so far and the tools on offer."""
    def step(self,messages:list[dict],tools:list[dict])->Turn:...

def parse_args(raw):
    if isinstance(raw,dict):return raw
    try:value=json.loads(raw or '{}')
    except (TypeError,ValueError):return raw
    return value if isinstance(value,dict) else raw

# ---------- providers ----------
PROVIDERS={
    'openai':{'base_url':'https://api.openai.com/v1','key':'OPENAI_API_KEY','model':'gpt-4.1-mini','embed_model':'text-embedding-3-small','max_tokens_field':'max_completion_tokens','temperature':None},
    'anthropic':{'base_url':'https://api.anthropic.com/v1','key':'ANTHROPIC_API_KEY','model':'claude-haiku-5-5','embed_model':None},
    'nim':{'base_url':'https://integrate.api.nvidia.com/v1','key':'NIM_API_KEY','model':'openai/gpt-oss-20b','embed_model':'nvidia/nemotron-3-embed-1b','max_tokens_field':'max_tokens','temperature':0},
    'openai-compatible':{'base_url':None,'key':'TOOLPICK_API_KEY','model':None,'embed_model':None,'max_tokens_field':'max_tokens','temperature':0},
}

def provider_settings(provider,model=None,base_url=None):
    """Resolve model, base URL and API key: explicit value, then TOOLPICK_MODEL / TOOLPICK_BASE_URL, then the provider default."""
    if provider not in PROVIDERS:raise ValueError(f'unknown provider {provider!r}; choose from {", ".join(PROVIDERS)}')
    p=PROVIDERS[provider]
    base=(base_url or os.getenv('TOOLPICK_BASE_URL') or p['base_url'] or '').rstrip('/')
    model=model or os.getenv('TOOLPICK_MODEL') or p['model']
    if not base:raise ValueError(f'{provider}: set --base-url or TOOLPICK_BASE_URL (e.g. http://localhost:11434/v1 for Ollama)')
    if not model:raise ValueError(f'{provider}: set --model or TOOLPICK_MODEL')
    key=os.getenv(p['key'])
    if not key and provider!='openai-compatible':raise ValueError(f'{provider}: set the {p["key"]} environment variable')
    return {'provider':provider,'model':model,'base_url':base,'api_key':key}

class OpenAIChat:
    """Any OpenAI-compatible /chat/completions endpoint: OpenAI, NVIDIA NIM, Groq, Together, vLLM, Ollama, LM Studio ..."""
    def __init__(self,model,base_url,api_key=None,provider='openai-compatible',temperature='default',max_tokens=2048,timeout=120,retries=6,max_backoff=60,transport=None,sleep=None):
        p=PROVIDERS.get(provider,PROVIDERS['openai-compatible'])
        self.model=model;self.url=base_url.rstrip('/')+'/chat/completions';self.max_tokens=max_tokens
        self.temperature=p.get('temperature') if temperature=='default' else temperature
        self.max_tokens_field=p.get('max_tokens_field','max_tokens')
        self.headers={'Authorization':f'Bearer {api_key}'} if api_key else {}
        self.http=httpx.Client(timeout=timeout,transport=transport);self.retry={'retries':retries,'max_backoff':max_backoff,**({'sleep':sleep} if sleep else {})}
    @staticmethod
    def to_wire(messages):
        out=[]
        for m in messages:
            if m['role']=='assistant':
                msg={'role':'assistant','content':m.get('content')}
                if m.get('tool_calls'):msg['tool_calls']=[{'id':c.id,'type':'function','function':{'name':c.name,'arguments':c.arguments if isinstance(c.arguments,str) else json.dumps(c.arguments)}} for c in m['tool_calls']]
                out.append(msg)
            elif m['role']=='tool':out.append({'role':'tool','tool_call_id':m['tool_call_id'],'content':m['content']})
            else:out.append({'role':m['role'],'content':m['content']})
        return out
    def step(self,messages,tools):
        body={'model':self.model,'messages':self.to_wire(messages),self.max_tokens_field:self.max_tokens}
        if self.temperature is not None:body['temperature']=self.temperature
        if tools:body.update(tools=[{'type':'function','function':t} for t in tools],tool_choice='auto')
        data=post_json(self.http,self.url,self.headers,body,**self.retry)
        msg=data['choices'][0]['message'];usage=data.get('usage') or {}
        # Some servers leak chat-template tokens into tool names (seen with gpt-oss: "get_order<|channel|>commentary").
        calls=[ToolCall(c.get('id') or f'call_{i}',c['function']['name'].split('<|')[0].strip(),parse_args(c['function'].get('arguments'))) for i,c in enumerate(msg.get('tool_calls') or [])]
        return Turn(msg.get('content'),calls,usage.get('prompt_tokens'),usage.get('completion_tokens'))

class AnthropicMessages:
    """Anthropic Messages API with native tool use (tool_use / tool_result blocks)."""
    def __init__(self,model,api_key,base_url='https://api.anthropic.com/v1',max_tokens=8192,timeout=300,retries=6,max_backoff=60,transport=None,sleep=None):
        self.model=model;self.url=base_url.rstrip('/')+'/messages';self.max_tokens=max_tokens
        self.headers={'x-api-key':api_key or '','anthropic-version':'2023-06-01'}
        self.http=httpx.Client(timeout=timeout,transport=transport);self.retry={'retries':retries,'max_backoff':max_backoff,**({'sleep':sleep} if sleep else {})}
    @staticmethod
    def to_wire(messages):
        system=[];out=[]
        for m in messages:
            if m['role']=='system':system.append(m['content']);continue
            if m['role']=='assistant':
                # Send Anthropic's own content back unchanged (it may hold thinking blocks that must be replayed as-is).
                blocks=m.get('raw') if isinstance(m.get('raw'),list) else [*([{'type':'text','text':m['content']}] if m.get('content') else []),*({'type':'tool_use','id':c.id,'name':c.name,'input':c.arguments if isinstance(c.arguments,dict) else {}} for c in m.get('tool_calls',[]))]
                out.append({'role':'assistant','content':blocks});continue
            if m['role']=='tool':
                block={'type':'tool_result','tool_use_id':m['tool_call_id'],'content':m['content']}
                if out and out[-1]['role']=='user' and isinstance(out[-1]['content'],list) and out[-1]['content'] and out[-1]['content'][0].get('type')=='tool_result':out[-1]['content'].append(block)
                else:out.append({'role':'user','content':[block]})
                continue
            out.append({'role':'user','content':m['content']})
        return '\n\n'.join(system),out
    def step(self,messages,tools):
        system,wire=self.to_wire(messages)
        body={'model':self.model,'max_tokens':self.max_tokens,'messages':wire}
        if system:body['system']=system
        if tools:body.update(tools=[{'name':t['name'],'description':t['description'],'input_schema':t['parameters']} for t in tools],tool_choice={'type':'auto'})
        data=post_json(self.http,self.url,self.headers,body,**self.retry)
        content=data.get('content') or [];usage=data.get('usage') or {}
        text='\n'.join(b['text'] for b in content if b.get('type')=='text') or None
        calls=[ToolCall(b['id'],b['name'],b.get('input') if isinstance(b.get('input'),dict) else {}) for b in content if b.get('type')=='tool_use']
        inp=sum(usage.get(k) or 0 for k in ('input_tokens','cache_creation_input_tokens','cache_read_input_tokens')) if usage else None
        return Turn(text,calls,inp,usage.get('output_tokens'),raw=content)

# ---------- offline fake model ----------
WORD=re.compile(r'[a-z0-9]+')
STOP=set('a an the to of for and or in on at by with from is are be it its this that what which who how do does my our me i we you please can could would should all any'.split())
def words(text):return [w for w in WORD.findall(re.sub(r'([a-z])([A-Z])',r'\1 \2',str(text)).replace('_',' ').lower()) if w not in STOP]

class FakeAgent:
    """Deterministic, offline stand-in for a model (used by --dry-run and the tests). It calls the visible tool whose
    name and description best overlap the request, searches first when it has search_tools, routes by word overlap,
    and answers in text when nothing matches. It is a plumbing check, not a benchmark of anything."""
    def __init__(self,threshold=2):self.threshold=threshold
    def step(self,messages,tools):
        prompt=next(m['content'] for m in messages if m['role']=='user');q=set(words(prompt))
        size=len(json.dumps([{k:v for k,v in m.items() if k!='tool_calls'} for m in messages]))+len(json.dumps(tools))
        turn=lambda text=None,calls=():Turn(text,list(calls),size//4,20)
        names={t['name'] for t in tools};n=sum(m['role']=='assistant' for m in messages)
        if 'choose_categories' in names:
            spec=next(t for t in tools if t['name']=='choose_categories')
            options=re.findall(r'^- (\S+): (.*)$',spec['description'],re.M)
            ranked=sorted(options,key=lambda o:-len(q&set(words(o[0]+' '+o[1]))))
            return turn(calls=[ToolCall('route','choose_categories',{'categories':[ranked[0][0]] if ranked else []})])
        if 'search_tools' in names and n==0:return turn(calls=[ToolCall('s0','search_tools',{'query':prompt})])
        if any(m['role']=='tool' and m.get('name')!='search_tools' for m in messages):return turn('Done.')
        best=None;score=0
        for t in tools:
            if t['name']=='search_tools':continue
            s=len(q&set(words(t['name'])))*2+len(q&set(words(t['description'])))
            if s>score:best,score=t,s
        if best is None or score<self.threshold:return turn("Sorry, I can't do that with the tools I have.")
        ids=re.findall(r'\b[A-Z]{1,4}\d+\b',prompt)
        args={p:(ids[0] if ids else 'x') for p in best['parameters'].get('required',[])}
        return turn(calls=[ToolCall('c1',best['name'],args)])

# ---------- loading ----------
def load_custom(spec):
    """'package.module:Class' or 'path/to/file.py:Class' -> an instance (constructed with no arguments)."""
    target,_,name=spec.rpartition(':')
    if not target or not name:raise ValueError('--agent must look like module:Class or path/to/file.py:Class')
    if target.endswith('.py') or Path(target).exists():
        path=Path(target).resolve();mod_name=f'toolpick_custom_{path.stem}'
        module_spec=importlib.util.spec_from_file_location(mod_name,path);module=importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name]=module;module_spec.loader.exec_module(module)
    else:module=importlib.import_module(target)
    agent=getattr(module,name)()
    if not callable(getattr(agent,'step',None)):raise TypeError(f'{spec} has no step(messages, tools) method')
    return agent

def make_agent(provider,model=None,base_url=None,agent=None,**kw):
    """Build the agent for a run: a custom adapter, the offline fake, or a built-in provider."""
    if agent:return load_custom(agent),{'agent':agent}
    if provider=='fake':return FakeAgent(),{'provider':'fake','model':'fake'}
    s=provider_settings(provider,model,base_url)
    if provider=='anthropic':a=AnthropicMessages(s['model'],s['api_key'],s['base_url'],**{k:v for k,v in kw.items() if k in ('timeout','retries','max_backoff')})
    else:a=OpenAIChat(s['model'],s['base_url'],s['api_key'],provider,**kw)
    return a,{k:s[k] for k in ('provider','model','base_url')}

__all__=['Agent','Turn','ToolCall','OpenAIChat','AnthropicMessages','FakeAgent','PROVIDERS','make_agent','load_custom','APIError']
