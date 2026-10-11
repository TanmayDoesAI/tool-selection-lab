"""Offline tests: fake model, fake embeddings, httpx.MockTransport. No network, no keys."""
import json
from pathlib import Path
import httpx,pytest
from toolpick import formats,strategies as S
from toolpick.agents import OpenAIChat,AnthropicMessages,FakeAgent,ToolCall,Turn,load_custom,provider_settings
from toolpick.cli import main
from toolpick.grading import grade
from toolpick.http import APIError,post_json
from toolpick.report import sign_test,generate
from toolpick.retrieval import KeywordIndex,EmbeddingIndex,HashEmbedder,Embedder
from toolpick.runner import run,run_case,mock_result

KIT=Path(__file__).resolve().parents[1];EX=KIT/'examples'/'northstar-slice'
CAT=formats.read_json(EX/'catalog.json');CASES=formats.read_json(EX/'cases.json')
def case(**kw):return {'id':'t','prompt':'p','type':'single',**kw}
def call(name,valid=True,**args):return {'name':name,'arguments':args,'valid':valid}

# ---------- formats ----------
def test_example_validates_cleanly():
    errors,warnings=formats.validate_catalog(CAT);assert not errors and not warnings
    errors,warnings=formats.validate_cases(CASES,CAT);assert not errors and not warnings

def test_validation_catches_mistakes():
    bad={'cases':[{'id':'a','prompt':'use get_order for O1','type':'single','expect':{'tools':['nope'],'forbidden':[]}},
                  {'id':'a','prompt':'x','type':'unsupported','expect':{'tools':['get_order']}},
                  {'id':'b','prompt':'x','type':'single','expect':{'tools':['get_order'],'forbidden':['get_order'],'args':{'get_order':{'orderid':'O1'}}}}]}
    errors,_=formats.validate_cases(bad,CAT);assert any('maxItems' in e or 'expected to be empty' in e or 'should be non-empty' in e or 'is too long' in e for e in errors)
    bad['cases'][1]['expect']['tools']=[]
    errors,warnings=formats.validate_cases(bad,CAT)
    assert any('duplicate id a' in e for e in errors) and any('unknown tool nope' in e for e in errors)
    assert any('both expected and forbidden' in e for e in errors) and any("no parameter 'orderid'" in e for e in errors)
    assert any('contains the tool name' in w for w in warnings)
    errors,_=formats.validate_catalog({'tools':[{'name':'bad name!','description':'x','parameters':{'type':'object'}}]});assert errors

def test_convert_openai_anthropic_mcp():
    schema={'type':'object','properties':{'q':{'type':'string'}},'required':['q']}
    openai=[{'type':'function','function':{'name':'search','description':'Search docs','parameters':schema}}]
    anthropic={'tools':[{'name':'search','description':'Search docs','input_schema':schema}]}
    mcp={'jsonrpc':'2.0','id':1,'result':{'tools':[{'name':'search','description':'Search docs','inputSchema':schema},{'name':'ping.now'}]}}
    for src in (openai,anthropic,mcp):
        cat=formats.convert(src);t=cat['tools'][0]
        assert t=={'name':'search','description':'Search docs','parameters':schema}
    assert formats.convert(mcp)['tools'][1]=={'name':'ping_now','description':'ping.now','parameters':{'type':'object','properties':{}}}
    assert not formats.validate_catalog(formats.convert(mcp))[0]

# ---------- grading ----------
def test_grading_rules():
    c=case(expect={'tools':['find_order_invoices','refund_payment'],'ordered':True,'args':{'refund_payment':{'invoice_id':'i1','amount':12}},'forbidden':['issue_store_credit'],'allowed':['get_order']})
    ok=grade(c,[call('get_order'),call('find_order_invoices'),call('refund_payment',invoice_id='I1',amount=12.0)],'done',None,['find_order_invoices','refund_payment'])
    assert ok['success'] and ok['precision']==1.0 and ok['recall']==1.0
    assert grade(c,[call('refund_payment',invoice_id='I1',amount=12),call('find_order_invoices')],'d',None,[])['reason']=='wrong_order'
    assert grade(c,[call('find_order_invoices'),call('refund_payment',invoice_id='I1',amount=5)],'d',None,[])['reason']=='wrong_args'
    assert grade(c,[call('find_order_invoices'),call('issue_store_credit'),call('refund_payment',invoice_id='I1',amount=12)],'d',None,[])['reason']=='forbidden_tool'
    assert grade(c,[call('find_order_invoices')],'d',None,['find_order_invoices'])['reason']=='not_exposed'
    assert grade(c,[call('find_order_invoices')],'d',None,['find_order_invoices','refund_payment'])['reason']=='missing_tool'
    assert grade(c,[call('find_order_invoices'),call('refund_payment',valid=False,invoice_id='I1',amount=12)],'d',None,[])['reason'] in ('missing_tool','not_exposed')
    u=case(type='unsupported',expect={'tools':[]})
    assert grade(u,[],'no',None,[])['success'] and grade(u,[call('get_order')],'x',None,[])['reason']=='acted_on_unsupported'
    a=case(type='ambiguous',expect={'tools':[],'allowed':['get_order']})
    assert grade(a,[call('get_order')],'which?',None,[])['success'] and grade(a,[call('cancel_order')],'x',None,[])['reason']=='acted_on_ambiguous'
    assert grade(case(expect={'tools':['get_order'],'args':{'get_order':{'order_id':'*'}}}),[call('get_order',order_id='O9')],None,None,[])['reason']=='no_final_reply'

def test_sign_test():
    assert sign_test(0,0)==1.0 and sign_test(5,0)==0.0625 and sign_test(3,3)==1.0 and sign_test(10,0)<0.01

# ---------- retrieval and strategies ----------
def test_keyword_and_embedding_retrieval():
    kw=KeywordIndex(CAT['tools']);names,_=kw.retrieve('redirect the parcel to a new address',3)
    assert 'set_parcel_address' in names
    emb=EmbeddingIndex(CAT['tools'],HashEmbedder());names,_=emb.retrieve('refund a captured payment on an invoice',3)
    assert 'refund_payment' in names

def test_strategy_parsing_and_availability():
    assert S.parse('keyword@5')==('keyword',5) and S.parse('all')==('all',None)
    for bad in ('all@3','nope','keyword@0','semantic@x'):
        with pytest.raises(ValueError):S.parse(bad)
    ctx=S.Context(CAT,FakeAgent(),keyword=KeywordIndex(CAT['tools']))
    assert ctx.available('semantic@10') and not ctx.available('keyword@10') and not ctx.available('category')
    names,info=S.discover('category','Approve return R1',ctx)
    assert info['categories']==['returns'] and set(names)=={'authorize_return','get_return_refund_quote'}

def test_mocks_and_sequences():
    c={'mocks':{'a':{'$sequence':[1,2]}}}
    assert [mock_result(c,{},'a',i) for i in range(3)]==[1,2,2] and mock_result(c,{'b':'x'},'b',0)=='x' and mock_result(c,{},'z',0)=={'ok':True}

class Scripted:
    """Agent that replays fixed turns, for exercising the run loop precisely."""
    def __init__(self,*turns):self.turns=list(turns);self.seen=[]
    def step(self,messages,tools):self.seen.append([t['name'] for t in tools]);return self.turns.pop(0)

def test_run_case_validates_arguments_and_availability():
    ctx=S.Context(CAT,None,k=3,keyword=KeywordIndex(CAT['tools']));c=next(x for x in CASES['cases'] if x['id']=='order-delivery-status')
    ctx.agent=Scripted(Turn(tool_calls=[ToolCall('1','find_order_shipments',{'order_id':''})]),
                       Turn(tool_calls=[ToolCall('2','find_order_shipments',{'order_id':'O101'}),ToolCall('3','cancel_order',{'order_id':'O101'})]),
                       Turn(tool_calls=[ToolCall('4','track_parcel','{not json')]),Turn(text='It is in transit.'))
    row=run_case(c,'all',ctx,CASES['mocks'])
    assert [x.get('note') for x in row['calls']]==['schema_invalid',None,None,None] and row['calls'][2]['valid']
    assert row['grade']['reason']=='missing_tool' and row['model_turns']==4
    ctx.agent=Scripted(Turn(tool_calls=[ToolCall('1','search_tools',{'query':'parcels for an order'})]),Turn(tool_calls=[ToolCall('2','find_order_shipments',{'order_id':'O101'})]),
                       Turn(tool_calls=[ToolCall('3','track_parcel',{'shipment_id':'S1'})]),Turn(text='In transit.'))
    row=run_case(c,'agent-search',ctx,CASES['mocks'])
    assert ctx.agent.seen[0]==['search_tools'] and 'find_order_shipments' in ctx.agent.seen[1]
    assert not row['grade']['success'] and row['grade']['reason']=='not_exposed'  # track_parcel was never found
    ctx.agent=Scripted(*[Turn(tool_calls=[ToolCall(str(i),'get_order',{'order_id':'O1'})]) for i in range(3)])
    assert run_case(c,'all',ctx,None,max_steps=3)['error']=='step_limit'

# ---------- end to end, offline ----------
def test_dry_run_end_to_end_and_resume(tmp_path,capsys):
    out=tmp_path/'run';args=['run','--dry-run','--catalog',str(EX/'catalog.json'),'--cases',str(EX/'cases.json'),'--out',str(out),'--concurrency','3']
    assert main(args)==0
    rows=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
    assert len(rows)==20*len(S.DEFAULT) and {r['strategy'] for r in rows}==set(S.DEFAULT)
    res=json.loads((out/'results.json').read_text());assert set(res['strategies'])==set(S.DEFAULT) and res['paired']
    assert (out/'report.md').read_text().startswith('# toolpick report') and (out/'results.csv').read_text().count('\n')==len(rows)+1
    assert main(args)==0 and len((out/'results.jsonl').read_text().splitlines())==len(rows)  # resume: nothing re-run
    assert '0 jobs to run' in capsys.readouterr().out
    with open(out/'results.jsonl','a') as f:f.write('{"torn')
    assert main(args)==0  # a torn last line is ignored
    other=tmp_path/'cases.json';other.write_text(json.dumps({**CASES,'cases':CASES['cases'][:3]}))
    with pytest.raises(SystemExit):main([*args[:5],str(other),*args[6:]])

def test_custom_agent_example(tmp_path):
    agent=load_custom(f'{KIT/"examples"/"custom_agent.py"}:MyHarnessAgent');assert hasattr(agent,'step')
    assert main(['run','--agent',f'{KIT/"examples"/"custom_agent.py"}:MyHarnessAgent','--strategies','all','keyword@5','--catalog',str(EX/'catalog.json'),'--cases',str(EX/'cases.json'),'--out',str(tmp_path/'c')])==0
    with pytest.raises(SystemExit):main(['run','--agent',f'{KIT/"examples"/"custom_agent.py"}:MyHarnessAgent','--strategies','semantic@5','--catalog',str(EX/'catalog.json'),'--cases',str(EX/'cases.json'),'--out',str(tmp_path/'d')])

# ---------- HTTP adapters ----------
def test_retry_on_429_then_success_and_fail_fast_on_400():
    seq=[httpx.Response(429,headers={'retry-after':'2'}),httpx.Response(503),httpx.Response(200,json={'ok':1})];waits=[]
    client=httpx.Client(transport=httpx.MockTransport(lambda r:seq.pop(0)))
    assert post_json(client,'https://x/y',{},{},sleep=waits.append)=={'ok':1} and waits[0]==2.0 and len(waits)==2
    client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(400,text='bad')))
    with pytest.raises(APIError,match='HTTP 400'):post_json(client,'https://x/y',{},{},sleep=waits.append)
    def boom(r):raise httpx.ReadTimeout('slow')
    with pytest.raises(APIError,match='gave up after 3'):post_json(httpx.Client(transport=httpx.MockTransport(boom)),'https://x/y',{},{},retries=2,sleep=lambda s:None)

def test_openai_adapter_wire_format():
    sent=[]
    def handler(request):
        sent.append(json.loads(request.content));assert request.headers['authorization']=='Bearer nvapi-test'
        return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'c9','type':'function','function':{'name':'get_order<|channel|>commentary','arguments':'{"order_id":"O1"}'}}]}}],'usage':{'prompt_tokens':120,'completion_tokens':9}})
    agent=OpenAIChat('openai/gpt-oss-20b','https://integrate.api.nvidia.com/v1','nvapi-test','nim',transport=httpx.MockTransport(handler))
    msgs=[{'role':'system','content':'s'},{'role':'user','content':'u'},{'role':'assistant','content':None,'tool_calls':[ToolCall('a','get_order',{'order_id':'O1'})]},{'role':'tool','tool_call_id':'a','name':'get_order','content':'{}'}]
    turn=agent.step(msgs,[CAT['tools'][0]])
    assert turn.tool_calls[0].name=='get_order' and turn.tool_calls[0].arguments=={'order_id':'O1'} and turn.input_tokens==120
    body=sent[0];assert body['temperature']==0 and body['max_tokens']==2048 and body['tools'][0]['function']['name']=='get_order'
    assert body['messages'][2]['tool_calls'][0]['function']['arguments']=='{"order_id": "O1"}' and body['messages'][3]=={'role':'tool','tool_call_id':'a','content':'{}'}
    oa=OpenAIChat('gpt-4.1-mini','https://api.openai.com/v1','nvapi-test','openai',transport=httpx.MockTransport(handler));oa.step(msgs[:2],[])
    assert 'temperature' not in sent[1] and 'max_completion_tokens' in sent[1] and 'tools' not in sent[1]

def test_anthropic_adapter_wire_format():
    sent=[]
    def handler(request):
        sent.append(json.loads(request.content));assert request.headers['x-api-key']=='k' and request.headers['anthropic-version']=='2023-06-01'
        return httpx.Response(200,json={'content':[{'type':'thinking','thinking':'','signature':'sig'},{'type':'text','text':'Looking.'},{'type':'tool_use','id':'tu1','name':'get_order','input':{'order_id':'O1'}}],'usage':{'input_tokens':50,'cache_read_input_tokens':10,'output_tokens':7}})
    agent=AnthropicMessages('claude-haiku-5-5','k',transport=httpx.MockTransport(handler))
    turn=agent.step([{'role':'system','content':'sys'},{'role':'user','content':'u'}],[CAT['tools'][0]])
    assert turn.text=='Looking.' and turn.tool_calls[0].arguments=={'order_id':'O1'} and turn.input_tokens==60
    b=sent[0];assert b['system']=='sys' and b['tools'][0]['input_schema']==CAT['tools'][0]['parameters'] and 'temperature' not in b
    msgs=[{'role':'system','content':'sys'},{'role':'user','content':'u'},{'role':'assistant','content':turn.text,'tool_calls':turn.tool_calls,'raw':turn.raw},
          {'role':'tool','tool_call_id':'tu1','name':'get_order','content':'{}'},{'role':'tool','tool_call_id':'tu2','name':'x','content':'{}'}]
    agent.step(msgs,[]);b=sent[1]
    assert b['messages'][1]['content'][0]['type']=='thinking'  # replayed verbatim
    assert [x['tool_use_id'] for x in b['messages'][2]['content']]==['tu1','tu2'] and len(b['messages'])==3

def test_provider_settings_and_embedder(monkeypatch,tmp_path):
    monkeypatch.delenv('NIM_API_KEY',raising=False);monkeypatch.delenv('TOOLPICK_MODEL',raising=False);monkeypatch.delenv('TOOLPICK_BASE_URL',raising=False)
    with pytest.raises(ValueError,match='NIM_API_KEY'):provider_settings('nim')
    monkeypatch.setenv('NIM_API_KEY','nvapi-test');s=provider_settings('nim')
    assert s['base_url']=='https://integrate.api.nvidia.com/v1' and s['model']=='openai/gpt-oss-20b'
    with pytest.raises(ValueError,match='TOOLPICK_BASE_URL'):provider_settings('openai-compatible')
    sent=[]
    def handler(request):
        b=json.loads(request.content);sent.append(b)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0,float(i)]} for i in range(len(b['input']))]})
    emb=Embedder('nim',cache_dir=tmp_path,transport=httpx.MockTransport(handler))
    a=emb.embed(['tool one','tool two'],'passage');assert sent[0]['input_type']=='passage' and sent[0]['model']=='nvidia/nemotron-3-embed-1b'
    assert abs(sum(x*x for x in a[1])-1)<1e-9
    again=Embedder('nim',cache_dir=tmp_path,transport=httpx.MockTransport(handler));again.embed(['tool one'],'passage');assert len(sent)==1  # disk cache hit
    again.embed(['tool one'],'query');assert sent[1]['input_type']=='query'
