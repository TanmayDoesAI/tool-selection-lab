import argparse,json,os,fcntl
from pathlib import Path
from jsonschema import Draft202012Validator
from .tools import TOOLS,CATEGORIES
from .tasks import load,fixture,verify_freeze
from .grading import grade
from .runner import create_run,run,load_run,RunStore
from .report import generate
from .api import atomic_json,PROVIDERS
from .strategies import parse_strategy

def validate():
    verify_freeze();assert len(TOOLS)==100 and len({t.name for t in TOOLS.values()})==100
    assert all(sum(t.category==c for t in TOOLS.values())==10 for c in CATEGORIES)
    for t in TOOLS.values():Draft202012Validator.check_schema(t.parameters);Draft202012Validator(t.parameters).validate(t.example)
    tasks=load('dev')+load('test');assert len(tasks)==100 and len({t['id'] for t in tasks})==100
    for task in tasks:
        assert task['split'] in ('dev','test')
        assert all(name in TOOLS for name,args in task['reference'])
    print('Validated 100 tools, 10 categories, 60 development + 40 frozen held-out tasks.')

def reference_reply(task,outputs):
    """A plain-language reply a perfect agent could give: it asks, declines, or reports every looked-up value."""
    status=task['grade']['status']
    if status=='clarification':return 'Which customer, order and amount do you mean?'
    if status=='unsupported':return "Sorry, that isn't something I can do with the tools I have."
    return 'Done. Here is what I found: '+json.dumps(outputs)+('. Nothing was found.' if any(o in ([],{}) for o in outputs) else '')

def offline():
    validate()
    for task in load('dev')+load('test'):
        state=fixture(task);calls=[];outputs=[]
        for name,args in task['reference']:
            output=TOOLS[name].execute(state,args);outputs.append(output)
            calls.append({'name':name,'arguments':args,'ok':True,'schema_valid':True,'mutates':TOOLS[name].mutates})
        reply=reference_reply(task,outputs)
        score=grade(task,state,calls,reply,list(TOOLS))
        assert score['completed'],(task['id'],score)
    for tool in TOOLS.values():
        state=fixture({'fixture_patch':{}})
        if tool.name=='restock_return':state['returns']['R1'].update(received=True,inspection='resellable')
        tool.execute(state,tool.example)
    print('Offline reference outcomes passed for all 100 tasks; all 100 tool examples executed. No inference calls.')

def main():
    p=argparse.ArgumentParser(description='Controlled synthetic retail tool-discovery benchmark')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate');sub.add_parser('offline')
    report=sub.add_parser('report');report.add_argument('run_dir',type=Path)
    resume=sub.add_parser('resume');resume.add_argument('run_dir',type=Path);resume.add_argument('--max-requests',type=int);resume.add_argument('--concurrency',type=int)
    index=sub.add_parser('index');index.add_argument('--cache',default='.cache')
    for name in ('smoke','benchmark'):
        r=sub.add_parser(name);r.add_argument('--out',type=Path,default=Path('runs')/name);r.add_argument('--provider',choices=list(PROVIDERS),default=os.getenv('TOOLBENCH_PROVIDER','groq'));r.add_argument('--model');r.add_argument('--split',choices=['dev','test','haiku'],default='dev');r.add_argument('--tasks',type=int);r.add_argument('--repetitions',type=int,default=1);r.add_argument('--strategies',nargs='+',default=['all','category','semantic'],help="all, category, semantic[@k], semantic-nim[@k]");r.add_argument('--category-limit',type=int,default=2);r.add_argument('--top-k',type=int,default=10);r.add_argument('--max-steps',type=int,default=6);r.add_argument('--max-requests',type=int,default=120);r.add_argument('--seed',type=int,default=1741);r.add_argument('--timeout',type=float,default=30);r.add_argument('--retries',type=int,default=6);r.add_argument('--max-backoff',type=float,default=60);r.add_argument('--embedding-model',default='en_core_web_md');r.add_argument('--nim-embedding-model',default=os.getenv('NIM_EMBEDDING_MODEL','nvidia/nemotron-3-embed-1b'));r.add_argument('--dry-run',action='store_true');r.add_argument('--concurrency',type=int,default=1,help='parallel jobs (operational; not part of the protocol)')
    args=p.parse_args()
    if args.command=='validate':return validate()
    if args.command=='offline':return offline()
    if args.command=='index':
        from .strategies import EmbeddingIndex
        return print(json.dumps(EmbeddingIndex(args.cache).metadata,indent=2))
    if args.command in ('report','resume'):
        path=args.run_dir;manifest=load_run(path)
        if args.command=='report':return print(json.dumps(generate(path,manifest),indent=2))
        if args.max_requests is not None:
            if args.max_requests<manifest['api_requests']:p.error('total cap cannot be below requests already consumed')
            manifest['config']['max_requests']=args.max_requests
        if manifest['freeze']!=verify_freeze():p.error('Frozen task identity differs from checkpoint')
    else:
        for s in args.strategies:
            try:parse_strategy(s)
            except ValueError as exc:p.error(str(exc))
        if not 1<=args.category_limit<=10 or not 1<=args.top_k<=100 or not 1<=args.max_steps<=30 or args.max_requests<1 or args.repetitions<1 or not 0<=args.retries<=10 or args.timeout<=0 or args.max_backoff<0:p.error('Invalid experiment bounds')
        if args.command=='smoke':
            if args.split!='dev' or args.repetitions!=1 or args.max_requests>120 or len(set(args.strategies))!=len(args.strategies) or args.tasks is not None:p.error('smoke is fixed: six dev tasks, distinct strategies, one repetition, <=120 requests')
            tasks=[t for t in load('dev') if t['smoke']]
        else:
            tasks=load(args.split)
            if args.tasks is not None:
                if args.tasks<1:p.error('--tasks must be positive')
                tasks=tasks[:args.tasks]
        if args.model is None:args.model=os.getenv('GROQ_MODEL') if args.provider=='groq' and os.getenv('GROQ_MODEL') else PROVIDERS[args.provider]['default_model']
        cfg=vars(args).copy();cfg.pop('command');path=cfg.pop('out');cfg.pop('dry_run');cfg.pop('tasks')
        plan={'tasks':len(tasks),'strategies':len(args.strategies),'repetitions':args.repetitions,'observations':len(tasks)*len(args.strategies)*args.repetitions,'max_api_requests_including_retries_routing_preflight':args.max_requests,'worst_case_attempts':(args.retries+1)*(1+len(tasks)*args.repetitions*(len(args.strategies)*args.max_steps+('category' in args.strategies))),'task_ids':[t['id'] for t in tasks]}
        if args.dry_run:return print(json.dumps(plan,indent=2))
        path.mkdir(parents=True,exist_ok=True)
        if (path/'checkpoint.json').exists():p.error('Run exists; use resume or choose a new --out')
        manifest=create_run(path,cfg,tasks)
    # Reject concurrent writers; checkpoint and pre-request reservations are atomic.
    with (path/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another runner owns this run directory')
        try:manifest=run(path,manifest,concurrency=getattr(args,'concurrency',None))
        except KeyboardInterrupt:manifest['status']='interrupted';RunStore(path).save_meta(manifest)
        print(json.dumps(generate(path,manifest),indent=2))
        if manifest['status']=='blocked':print('BLOCKER:',manifest.get('blocker'))
if __name__=='__main__':main()
