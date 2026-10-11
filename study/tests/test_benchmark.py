import json
from copy import deepcopy
import httpx
import pytest
from toolbench.tools import TOOLS
from toolbench.tasks import load,fixture,verify_freeze
from toolbench.grading import grade
from toolbench.world import fresh
from toolbench.api import GroqClient,LimitReached,APIError,atomic_json,sanitize
from toolbench.strategies import discover,EmbeddingIndex,NimEmbeddingIndex
from toolbench.runner import create_run,run,execute
from toolbench.report import generate

@pytest.fixture
def cfg():return {'model':'test-only','timeout':1,'retries':2,'max_requests':120,'seed':1741,'repetitions':1,'strategies':['all'],'split':'dev','top_k':10,'category_limit':2,'max_steps':6,'embedding_model':'en_core_web_md'}

@pytest.fixture
def task():return next(t for t in load('dev') if t['id']=='orders-01')

def client(tmp_path,cfg,handler,monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','test-secret')
    m={'config':cfg,'api_requests':0,'attempts':[]}
    c=GroqClient(m,lambda:atomic_json(tmp_path/'m.json',m),transport=httpx.MockTransport(handler),sleep=lambda _:None)
    return c,m

def response(message):return httpx.Response(200,json={'choices':[{'message':message}],'usage':{'prompt_tokens':10,'completion_tokens':5}})

def toolcall(name='get_order',args=None):return {'role':'assistant','content':None,'tool_calls':[{'id':'call1','type':'function','function':{'name':name,'arguments':json.dumps(args or {'order_id':'O100'})}}]}

@pytest.mark.parametrize('name',list(TOOLS))
def test_all_tools_execute(name):
    state=fresh()
    if name=='restock_return':state['returns']['R1'].update(received=True,inspection='resellable')
    assert TOOLS[name].execute(state,TOOLS[name].example) is not None

def test_fixture_isolation():
    one=fresh();two=fresh();TOOLS['refund_payment'].execute(one,{'invoice_id':'I1','amount':12})
    assert two['invoices']['I1']['refunded']==0

def test_business_guards():
    s=fresh()
    with pytest.raises(ValueError):TOOLS['refund_payment'].execute(s,{'invoice_id':'I1','amount':25})
    with pytest.raises(ValueError):TOOLS['reserve_stock'].execute(s,{'sku':'BAG','quantity':3})
    with pytest.raises(ValueError):TOOLS['cancel_order'].execute(s,{'order_id':'O101'})
    with pytest.raises(ValueError):TOOLS['restock_return'].execute(s,{'return_id':'R1'})
    assert s==fresh()

def test_schema_blocks_before_mutation():
    from jsonschema import ValidationError
    s=fresh()
    with pytest.raises(ValidationError):TOOLS['refund_payment'].execute(s,{'invoice_id':'I1','amount':-1})
    assert s==fresh()

def test_all_strategy_same_schema(cfg):
    names,info=discover('all','private request',None,cfg)
    assert len(names)==100 and info['size']==100

def test_category_only_sees_request_and_descriptions(cfg):
    class Router:
        def chat(self,messages,**kw):
            assert messages[1]['content']=='Where is the parcel?'
            assert 'reference' not in messages[0]['content']
            assert kw['phase']=='routing'
            return {'choices':[{'message':{'content':'{"categories":["shipping","orders"]}'}}]}
    names,info=discover('category','Where is the parcel?',Router(),cfg)
    assert len(names)==20 and set(TOOLS[n].category for n in names)=={'shipping','orders'}

def test_router_cap_rejected(cfg):
    class Router:
        def chat(self,*a,**k):return {'choices':[{'message':{'content':'{"categories":["shipping","orders","billing"]}'}}]}
    with pytest.raises(ValueError):discover('category','request',Router(),cfg)

def test_semantic_cache_and_topk(tmp_path,cfg):
    first=EmbeddingIndex(tmp_path);second=EmbeddingIndex(tmp_path)
    assert not first.cache_hit and second.cache_hit
    names,info=discover('semantic','Refund a captured invoice payment',None,cfg,second)
    assert len(names)==10 and 'refund_payment' in names
    assert first.fingerprint==second.fingerprint

def test_frozen_holdout():
    assert verify_freeze()['sha256']['test.json']
    assert len(load('dev'))==60 and len(load('test'))==40
    assert all(t['split']=='dev' for t in load('dev') if t['smoke'])

def test_alternative_tool_sequence_and_wrong_facts():
    t=next(t for t in load('dev') if t['id']=='inventory-02');s=fixture(t)
    calls=[{'name':'get_stock','arguments':{'sku':'MUG'},'ok':True,'schema_valid':True}]
    assert grade(t,s,calls,'You can sell 16 mugs right now.',list(TOOLS))['completed']
    wrong=grade(t,s,calls,'You can sell 20 mugs right now.',list(TOOLS))
    assert not wrong['completed'] and 'missing_facts' in wrong['failures']

def test_state_grader_detects_unrelated_changes():
    t=next(t for t in load('dev') if t['id']=='billing-05');s=fixture(t)
    TOOLS['refund_payment'].execute(s,{'invoice_id':'I1','amount':12})
    calls=[{'name':'refund_payment','arguments':{'invoice_id':'I1','amount':12},'ok':True,'schema_valid':True}]
    assert grade(t,s,calls,'Refunded $12 on invoice I1.',list(TOOLS))['completed']
    s['customers']['C1']['credit']=12
    assert not grade(t,s,calls,'Refunded $12 on invoice I1.',list(TOOLS))['completed']

def test_clarification_cannot_guess_and_change():
    t=next(t for t in load('dev') if t['id']=='clarify-01');s=fixture(t)
    calls=[{'name':'set_customer_address','arguments':{},'mutates':True,'ok':False,'schema_valid':False}]
    assert grade(t,s,[],'There are two customers called Alex Lee. Which one do you mean?',list(TOOLS))['completed']
    assert not grade(t,s,[],'I have updated the address.',list(TOOLS))['completed']  # did not ask
    TOOLS['set_customer_address'].execute(s,{'customer_id':'C1','address':'21 Pine Road'})
    done=[{'name':'set_customer_address','arguments':{},'mutates':True,'ok':True,'schema_valid':True}]
    assert not grade(t,s,done,'Updated C1. Did you mean C2?',list(TOOLS))['completed']  # changed state, then asked

def test_prose_answer_needs_only_requested_facts():
    # The old prompt/grader demanded every field of the record; "how much mug stock" now needs only what was asked.
    t=next(t for t in load('haiku') if t['id']=='hA-07');s=fixture(t)
    calls=[{'name':'get_stock','arguments':{'sku':'MUG'},'ok':True,'schema_valid':True}]
    from toolbench.grading import expectations
    assert expectations()['hA-07'] and grade(t,s,calls,'We have 20 ceramic mugs on hand (4 reserved).',list(TOOLS))['completed']

def test_unsupported_declines_without_changes():
    t=next(t for t in load('dev') if t['type']=='unsupported');s=fixture(t)
    assert grade(t,s,[],"Sorry, I can't do that with the tools I have.",list(TOOLS))['completed']
    assert not grade(t,s,[],'',list(TOOLS))['completed']

def test_empty_result_must_be_reported():
    from toolbench.grading import expectations
    tid=next(k for k,v in expectations().items() if any(p.endswith('@empty') for p in v))
    t=next(t for split in ('dev','test','haiku') for t in load(split) if t['id']==tid)
    from toolbench.facts import gold_reads
    assert any(o==[] for o in gold_reads(t).values())

def test_request_limit_counts_retries_and_reserves_durably(tmp_path,cfg,monkeypatch):
    cfg['max_requests']=2
    def handler(request):
        saved=json.loads((tmp_path/'m.json').read_text())
        assert saved['api_requests']>=1 and saved['attempts'][-1]['outcome']=='reserved'
        return httpx.Response(429,json={'error':{'message':'limited'}})
    c,m=client(tmp_path,cfg,handler,monkeypatch)
    with pytest.raises(LimitReached):c.chat([{'role':'user','content':'test'}],phase='routing')
    assert m['api_requests']==2 and len(m['attempts'])==2
    assert all(e['phase']=='routing' for e in m['attempts'])

def test_nontransient_no_retry(tmp_path,cfg,monkeypatch):
    c,m=client(tmp_path,cfg,lambda r:httpx.Response(401,json={'error':{'message':'bad key'}}),monkeypatch)
    with pytest.raises(APIError):c.chat([])
    assert m['api_requests']==1

def test_timeout_retries(tmp_path,cfg,monkeypatch):
    def handler(r):raise httpx.ReadTimeout('offline timeout')
    c,m=client(tmp_path,cfg,handler,monkeypatch)
    with pytest.raises(APIError):c.chat([])
    assert m['api_requests']==3 and all(e['usage'] is None for e in m['attempts'])

def test_sanitization(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','gsk_example123')
    assert 'gsk_' not in json.dumps(sanitize({'message':'gsk_example123 Bearer abc'}))

def test_resume_keeps_completed_jobs_and_request_budget(tmp_path,cfg,task,monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','test-secret')
    cfg['max_requests']=2
    m=create_run(tmp_path,cfg,[task]);state=fresh();output=TOOLS['get_order'].execute(state,{'order_id':'O100'})
    def handler(request):
        body=json.loads(request.content)
        if body.get('tool_choice')!='auto':return response(toolcall())
        if body['messages'][-1]['role']=='tool':return response({'role':'assistant','content':json.dumps({'status':'completed','answer':output})})
        return response(toolcall())
    run(tmp_path,m,transport=httpx.MockTransport(handler))
    assert m['status']=='request_limit' and not m['results'] and m['api_requests']==2
    from toolbench.runner import load_run
    m=load_run(tmp_path);m['config']['max_requests']=4
    run(tmp_path,m,transport=httpx.MockTransport(handler))
    assert m['status']=='completed' and m['api_requests']==4 and len(m['results'])==1
    assert m['results'][0]['grading']['completed']
    run(tmp_path,m,transport=httpx.MockTransport(handler))
    assert m['api_requests']==4 and len(m['results'])==1
    summary=generate(tmp_path,m)
    assert summary['tokens']['input']==40 and summary['recorded']==1
    assert 'orphaned' in (tmp_path/'report.html').read_text()

def test_step_bound_and_unavailable_tool(tmp_path,cfg,task,monkeypatch):
    cfg['max_steps']=2
    c,m=client(tmp_path,cfg,lambda r:response(toolcall('refund_payment',{'invoice_id':'I1','amount':12})),monkeypatch)
    class Index:
        def retrieve(self,*a):return ['get_order'],[]
    row=execute(task,'semantic',c,cfg,Index())
    assert row['error']=='step_limit' and m['api_requests']==2
    assert row['state']['invoices']['I1']['refunded']==0
    assert all(not x['ok'] for x in row['calls'])

def test_outcome_completion_not_bound_to_reference_names():
    t=next(t for t in load('dev')+load('test') if t['id']=='analytics-03');s=fixture(t)
    # Sum MUG line quantities after observing both customers' orders.
    calls=[{'name':'list_customer_orders','arguments':{'customer_id':cid},'ok':True,'schema_valid':True} for cid in ('C1','C2')]
    score=grade(t,s,calls,'We have sold 2 mugs in total.',list(TOOLS))
    assert score['completed'] and not score['reference_evidence_coverage']
    # Choice score remains independent; this is why factual derivation needs human audit.
    assert not score['tool_choice']

def test_api_error_not_incorrect_selection():
    t=next(t for t in load('dev') if t['id']=='orders-01')
    score=grade(t,fixture(t),[],None,list(TOOLS),'api_error')
    assert 'api_error' in score['failures'] and 'incorrect_selection' not in score['failures']

def test_long_retry_wait_chunked_and_bounded(tmp_path,cfg,monkeypatch):
    cfg['max_backoff']=60
    calls=[]
    def handler(r):
        if not calls: return httpx.Response(429,headers={'retry-after':'45'},json={'error':{'message':'limited'}})
        return response({'role':'assistant','content':'ok'})
    c,m=client(tmp_path,cfg,handler,monkeypatch);c.sleep=lambda seconds:calls.append(seconds)
    c.chat([])
    assert calls==[30,pytest.approx(15,abs=0.01)] and m['api_requests']==2

def test_resume_blocks_changed_protocol(tmp_path,cfg,task,monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','test-secret')
    m=create_run(tmp_path,cfg,[task]);m['protocol_fingerprint']['tools.py']='changed'
    run(tmp_path,m,transport=httpx.MockTransport(lambda r:pytest.fail('Network forbidden after protocol mismatch')))
    assert m['status']=='blocked' and m['api_requests']==0

def test_unknown_semantic_arguments_are_not_claimed_correct():
    t=next(t for t in load('dev') if t['id']=='inventory-02')
    score=grade(t,fixture(t),[{'name':'inventory_summary','arguments':{},'ok':True,'schema_valid':True}],'I cannot do that.',list(TOOLS))
    assert score['schema_valid_all'] is True and score['arguments_correct'] is None

def nim_transport(log):
    def handler(request):
        body=json.loads(request.content);log.append(body)
        assert request.headers['authorization']=='Bearer nvapi-test'
        vecs=[[1.0,0.0] if ('refund' in t.lower() and 'payment' in t.lower()) else [0.0,1.0] for t in body['input']]
        return httpx.Response(200,json={'data':[{'index':i,'embedding':v} for i,v in enumerate(vecs)]})
    return httpx.MockTransport(handler)

def test_nim_index_caches_passages_and_queries_separately(tmp_path,cfg,monkeypatch):
    monkeypatch.setenv('NIM_API_KEY','nvapi-test');log=[]
    first=NimEmbeddingIndex(tmp_path,transport=nim_transport(log))
    assert not first.cache_hit and all(b['input_type']=='passage' for b in log) and sum(len(b['input']) for b in log)==100
    second=NimEmbeddingIndex(tmp_path,transport=nim_transport(log))
    assert second.cache_hit and len(log)==4 and second.fingerprint==first.fingerprint
    names,info=discover('semantic-nim','Refund a captured payment',None,cfg,{'semantic-nim':second})
    assert log[-1]['input_type']=='query' and len(names)==10 and 'refund_payment' in names
    assert 'nvapi-test' not in json.dumps(second.metadata)

def test_nim_missing_key_and_http_error(tmp_path,monkeypatch):
    monkeypatch.delenv('NIM_API_KEY',raising=False)
    with pytest.raises(APIError,match='NIM_API_KEY'):NimEmbeddingIndex(tmp_path)
    monkeypatch.setenv('NIM_API_KEY','nvapi-test');calls=[]
    def fail(request):calls.append(1);return httpx.Response(401,json={'error':'nvapi-test bad'})
    with pytest.raises(APIError) as exc:NimEmbeddingIndex(tmp_path,transport=httpx.MockTransport(fail),sleep=lambda s:None)
    assert len(calls)==1 and 'nvapi-test' not in str(exc.value)

def test_nim_key_is_redacted(monkeypatch):
    from toolbench.api import sanitize
    monkeypatch.setenv('NIM_API_KEY','nvapi-test-secretvalue')
    assert 'secretvalue' not in json.dumps(sanitize({'x':'nvapi-test-secretvalue','y':'nvapi-other_key-1'}))

def test_nim_chat_provider_uses_nim_endpoint_and_key(tmp_path,cfg,monkeypatch):
    monkeypatch.setenv('NIM_API_KEY','nvapi-chat');cfg['provider']='nim';seen=[]
    def handler(request):
        body=json.loads(request.content);seen.append(body)
        assert request.url.host=='integrate.api.nvidia.com' and request.headers['authorization']=='Bearer nvapi-chat'
        return response({'role':'assistant','content':'{"status":"completed","answer":{}}'})
    c,m=client(tmp_path,cfg,handler,monkeypatch)
    c.chat([{'role':'user','content':'hi'}])
    assert seen[0]['max_tokens']==1500 and 'max_completion_tokens' not in seen[0]
    assert 'nvapi-chat' not in json.dumps(m)
    monkeypatch.delenv('NIM_API_KEY')
    with pytest.raises(APIError,match='NIM_API_KEY'):c.chat([{'role':'user','content':'hi'}])

def test_strategy_topk_suffix(cfg):
    from toolbench.strategies import parse_strategy
    assert parse_strategy('semantic-nim@5')==('semantic-nim',5) and parse_strategy('all')==('all',None)
    for bad in ('all@5','semantic@0','semantic@101','nope','category@2'):
        with pytest.raises(ValueError):parse_strategy(bad)
    class Idx:
        def retrieve(self,request,k):return [f't{i}' for i in range(k)],[]
    names,_=discover('semantic-nim@3','x',None,cfg,{'semantic-nim':Idx()})
    assert names==['t0','t1','t2']
    names,_=discover('semantic-nim','x',None,cfg,{'semantic-nim':Idx()})
    assert len(names)==cfg['top_k']

def test_keyword_index_ranks_lexical_match():
    from toolbench.strategies import KeywordIndex
    names,_=KeywordIndex().retrieve('refund captured payment invoice',5)
    assert 'refund_payment' in names[:3] and len(names)==5

def test_agent_search_grows_shortlist_between_steps(tmp_path,cfg,monkeypatch):
    cfg['max_steps']=6;task={'id':'m','type':'multi_step','split':'dev','request':'Refund 12 dollars on O100','fixture_patch':{},
        'grade':{'status':'completed','facts':None,'state':{'invoices':{'I1':{**fresh()['invoices']['I1'],'refunded':12.0}}},'evidence_groups':[['find_order_invoices'],['refund_payment']],'argument_constraints':{}}}
    class Idx:
        def retrieve(self,q,k):return (['find_order_invoices'] if 'invoice' in q else ['refund_payment']),[]
    seen=[];script=[('search_tools',{'query':'find invoice for order'}),('refund_payment',{'invoice_id':'I1','amount':12}),
        ('find_order_invoices',{'order_id':'O100'}),('search_tools',{'query':'refund a payment'}),('refund_payment',{'invoice_id':'I1','amount':12}),None]
    def handler(request):
        body=json.loads(request.content);seen.append(sorted(t['function']['name'] for t in body['tools']))
        step=script[len(seen)-1]
        return response(toolcall(*step) if step else {'role':'assistant','content':'{"status":"completed","answer":"refunded"}'})
    c,m=client(tmp_path,cfg,handler,monkeypatch)
    row=execute(task,'agent-search@1',c,cfg,{'semantic-nim':Idx()})
    assert seen[0]==['search_tools'] and 'find_order_invoices' in seen[1] and 'refund_payment' not in seen[1]
    assert not row['calls'][0]['ok']  # refund attempted before it was found
    assert [s['added'] for s in row['discovery']['searches']]==[['find_order_invoices'],['refund_payment']]
    assert row['grading']['completed'] and row['discovery']['size']==2 and all(c['name']!='search_tools' for c in row['calls'])

def test_parallel_run_jsonl_storage_and_resume(tmp_path,cfg,monkeypatch):
    from toolbench.runner import load_run
    monkeypatch.setenv('GROQ_API_KEY','test-secret');cfg['strategies']=['all'];cfg['max_requests']=50
    tasks=[t for t in load('dev') if t['type']=='direct'][:6]
    def handler(request):
        body=json.loads(request.content)
        if body.get('tool_choice',{}) not in ('auto',None) and isinstance(body['tool_choice'],dict):return response(toolcall('get_order',{'order_id':'O100'}))
        return response({'role':'assistant','content':'{"status":"completed","answer":{}}'})
    m=create_run(tmp_path,cfg,tasks);run(tmp_path,m,transport=httpx.MockTransport(handler),concurrency=4)
    assert m['status']=='completed' and len(m['results'])==6
    ck=json.loads((tmp_path/'checkpoint.json').read_text())
    assert 'attempts' not in ck and 'results' not in ck and ck['storage']=='jsonl'
    again=load_run(tmp_path)
    assert len(again['results'])==6 and again['api_requests']==m['api_requests']==7
    assert sorted(r['job_key'] for r in again['results'])==sorted(r['job_key'] for r in m['results'])
    assert all(len(r['attempt_numbers'])==1 for r in again['results'])  # attempts attributed per job across threads
    assert all('tools' in e['request'] and 'messages' not in e['request'] for e in again['attempts'] if e['phase']=='execution')
    assert 'test-secret' not in (tmp_path/'attempts.jsonl').read_text()
    run(tmp_path,again,transport=httpx.MockTransport(handler),concurrency=4)
    assert again['api_requests']==7 and len(load_run(tmp_path)['results'])==6

def test_hybrid_starts_with_shortlist_and_can_search(cfg):
    from toolbench.strategies import SEARCH_TOOL,agent_search
    class Idx:
        def retrieve(self,q,k):return [f'tool{i}' for i in range(k)],[]
    names,info=discover('hybrid','refund',None,cfg,{'semantic-nim':Idx()})
    assert len(names)==10
    class Idx2:
        def retrieve(self,q,k):return ['refund_payment','get_order'][:k],[]
    names=['get_order'];out,rec=agent_search('hybrid@2','refund a payment',names,cfg,{'semantic-nim':Idx2()})
    assert rec['added']==['refund_payment'] and 'search again' in out['note'] and 'only the ones you have found' in SEARCH_TOOL['function']['description']

def test_template_token_leak_in_tool_name_is_normalised(tmp_path,cfg,task,monkeypatch):
    replies=[toolcall('get_order<|channel|>commentary',{'order_id':'O100'}),{'role':'assistant','content':'Order O100 is processing, total $24.'}]
    c,m=client(tmp_path,cfg,lambda r:response(replies.pop(0)),monkeypatch)
    row=execute(task,'all',c,cfg,None)
    assert row['calls'][0]['name']=='get_order' and row['calls'][0]['ok'] and row['calls'][0]['raw_name'].endswith('commentary')

def test_category_laya_uses_cached_route_by_request_text(cfg):
    t=next(t for t in load('haiku') if t['reference'])
    names,info=discover('category-laya',t['request'],None,cfg)
    assert 1<=len(info['categories'])<=2 and len(names)==10*len(info['categories']) and info['seconds']>0.5
    with pytest.raises(ValueError):discover('category-laya','a request nobody routed',None,cfg)

def test_rate_limit_cooldown_is_shared_across_workers(tmp_path,cfg,monkeypatch):
    import threading
    cfg['retries']=6;seen=[]
    def handler(r):
        seen.append(1)
        return httpx.Response(429,json={'error':'slow down'}) if len(seen)==1 else response({'role':'assistant','content':'ok'})
    c,m=client(tmp_path,cfg,handler,monkeypatch);waits={'a':[],'b':[]}
    c.sleep=lambda s:waits['a'].append(s);c.chat([])          # worker A: 429 -> exponential backoff (2 s + jitter)
    assert len(waits['a'])==1 and 2<=waits['a'][0]<=3
    def worker_b():
        c.sleep=lambda s:waits['b'].append(s);c.chat([])
    t=threading.Thread(target=worker_b);t.start();t.join()
    assert waits['b'] and waits['b'][0]>1.5                    # worker B also waits out A's cooldown
    assert m['api_requests']==3

def test_backoff_grows_exponentially_without_retry_after(tmp_path,cfg,monkeypatch):
    cfg['retries']=4;cfg['max_backoff']=60;sleeps=[]
    c,m=client(tmp_path,cfg,lambda r:httpx.Response(503,json={}),monkeypatch);c.sleep=sleeps.append
    with pytest.raises(APIError):c.chat([])
    assert [int(x) for x in sleeps]==[2,4,8,16] and m['api_requests']==5

def test_overloaded_529_is_retried(tmp_path,cfg,monkeypatch):
    cfg['retries']=2;sleeps=[]
    c,m=client(tmp_path,cfg,lambda r:httpx.Response(529,json={'error':{'message':'Service temporarily overloaded'}}),monkeypatch);c.sleep=sleeps.append
    with pytest.raises(APIError):c.chat([])
    assert len(sleeps)==2 and m['api_requests']==3

def test_nim_query_embeddings_cached_across_instances(tmp_path,cfg,monkeypatch):
    monkeypatch.setenv('NIM_API_KEY','nvapi-test');log=[]
    first=NimEmbeddingIndex(tmp_path,transport=nim_transport(log))
    a,_=first.retrieve('Refund a captured payment',10);calls=len(log)
    b,_=first.retrieve('Refund a captured payment',3)            # same request, smaller k: no new call
    assert len(log)==calls and b==a[:3] and first.cache_hits==1
    second=NimEmbeddingIndex(tmp_path,transport=nim_transport(log))  # new process / other model run
    c,_=second.retrieve('Refund a captured payment',10)
    assert len(log)==calls and c==a
