"""toolpick: which tool-discovery strategy works best for your agent and your tools?"""
import argparse,json,os,re,sys
from pathlib import Path
from . import formats,strategies as S
from .agents import make_agent,PROVIDERS
from .retrieval import KeywordIndex,EmbeddingIndex,Embedder,HashEmbedder
from .runner import run
from .report import generate,rank

def cmd_convert(a):
    data=formats.convert(formats.read_json(a.source))
    text=json.dumps(data,indent=2)+'\n'
    if a.out:Path(a.out).write_text(text);print(f'wrote {len(data["tools"])} tools to {a.out}',file=sys.stderr)
    else:sys.stdout.write(text)

def cmd_validate(a):
    try:cat,cases,warnings=formats.load(a.catalog,a.cases)
    except ValueError as exc:
        print('INVALID\n'+str(exc));return 1
    for w in warnings:print('warning:',w)
    n=len(cases['cases']) if cases else 0
    print(f'OK: {len(cat["tools"])} tools'+(f', {n} cases' if cases else '')+(f', {len(warnings)} warnings' if warnings else ''))
    return 1 if a.strict and warnings else 0

def embedder_for(a):
    if a.dry_run:return HashEmbedder()
    p=a.embed_provider
    if p=='auto':p=a.provider if a.provider in ('openai','nim') and not a.agent else 'none'
    if p=='none':return None
    return Embedder(p,a.embed_model,a.embed_base_url,cache_dir=Path(a.out)/'.cache',timeout=a.timeout,retries=a.retries)

def cmd_run(a):
    cat,cases,warnings=formats.load(a.catalog,a.cases)
    for w in warnings:print('warning:',w,file=sys.stderr)
    if a.mocks:cases['mocks']={**cases.get('mocks',{}),**formats.read_json(a.mocks)}
    provider='fake' if a.dry_run else a.provider
    kw={'timeout':a.timeout,'retries':a.retries}
    if a.temperature is not None:kw['temperature']=None if a.temperature=='none' else float(a.temperature)
    agent,meta=make_agent(provider,a.model,a.base_url,a.agent,**kw)
    if a.out is None:a.out=str(Path('toolpick-runs')/re.sub(r'[^A-Za-z0-9.-]+','-',meta.get('model') or Path(a.agent).stem))
    emb=embedder_for(a)
    ctx=S.Context(cat,agent,a.k,a.search_k,a.category_limit,KeywordIndex(cat['tools']),EmbeddingIndex(cat['tools'],emb) if emb else None)
    chosen=a.strategies or list(S.DEFAULT)
    for s in chosen:S.parse(s)
    usable=[]
    for s in chosen:
        why=ctx.available(s)
        if why and a.strategies:raise SystemExit(f'cannot run {s}: {why}')
        if why:print(f'skipping {s}: {why}',file=sys.stderr)
        else:usable.append(s)
    meta.update(embeddings=f'{emb.provider}:{emb.model}' if emb else None,search_index='semantic' if emb else 'keyword')
    run(cat,cases,usable,ctx,a.out,meta,a.repetitions,a.concurrency,a.max_steps,a.seed,a.retry_errors)
    return cmd_report(argparse.Namespace(run_dir=a.out))

def cmd_report(a):
    arms,_=generate(a.run_dir)
    print(f'\n{"strategy":<18}{"success":>10}{"precision":>11}{"recall":>8}{"med. tokens":>13}')
    for s in rank(arms):
        x=arms[s];print(f'{s:<18}{x["success"]:>5}/{x["n"]:<4}{x["precision"] if x["precision"] is not None else "-":>11}{x["recall"] if x["recall"] is not None else "-":>8}{x["median_input_tokens"] if x["median_input_tokens"] is not None else "-":>13}')
    print(f'\nreport: {Path(a.run_dir)/"report.md"}  (also results.json, results.csv)')
    return 0

def main(argv=None):
    p=argparse.ArgumentParser(prog='toolpick',description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('convert',help='turn OpenAI / Anthropic / MCP tool JSON into a toolpick catalog')
    c.add_argument('source');c.add_argument('-o','--out')
    v=sub.add_parser('validate',help='check catalog.json (and cases.json) against the schemas')
    v.add_argument('catalog');v.add_argument('cases',nargs='?');v.add_argument('--strict',action='store_true',help='treat warnings as errors')
    r=sub.add_parser('run',help='run every case under each strategy, then write the report')
    r.add_argument('--catalog',required=True);r.add_argument('--cases',required=True);r.add_argument('--out',help='run directory (default toolpick-runs/<model>)')
    r.add_argument('--provider',default=os.getenv('TOOLPICK_PROVIDER','openai'),choices=list(PROVIDERS))
    r.add_argument('--model',help='default: TOOLPICK_MODEL, then the provider example model');r.add_argument('--base-url',help='default: TOOLPICK_BASE_URL, then the provider URL')
    r.add_argument('--agent',help='your own adapter instead of a provider: module:Class or path/to/file.py:Class')
    r.add_argument('--dry-run',action='store_true',help='offline: fake model and fake embeddings, no network or keys')
    r.add_argument('--strategies',nargs='+',help=f'default: {" ".join(S.DEFAULT)} (semantic/category skipped when unavailable)')
    r.add_argument('--k',type=int,default=10,help='shortlist size for keyword/semantic/hybrid');r.add_argument('--search-k',type=int,default=5,help='results per search_tools call')
    r.add_argument('--category-limit',type=int,default=2);r.add_argument('--max-steps',type=int,default=8);r.add_argument('--repetitions',type=int,default=1)
    r.add_argument('--concurrency',type=int,default=4);r.add_argument('--timeout',type=float,default=120);r.add_argument('--retries',type=int,default=6)
    r.add_argument('--temperature',help='number, or "none" to omit it (default: 0 for NIM/compatible, omitted for OpenAI and Anthropic)')
    r.add_argument('--embed-provider',default='auto',choices=['auto','none','openai','nim','openai-compatible'],help='embeddings for semantic@k (auto: same as --provider when it is openai or nim)')
    r.add_argument('--embed-model');r.add_argument('--embed-base-url')
    r.add_argument('--mocks',help='JSON file: tool name -> stub result (default {"ok": true})')
    r.add_argument('--retry-errors',action='store_true',help='on resume, re-run jobs that ended in api_error');r.add_argument('--seed',type=int,default=0)
    rp=sub.add_parser('report',help='rebuild report.md / results.json / results.csv for a run directory');rp.add_argument('run_dir')
    a=p.parse_args(argv)
    try:return {'convert':cmd_convert,'validate':cmd_validate,'run':cmd_run,'report':cmd_report}[a.command](a) or 0
    except ValueError as exc:print('error:',exc,file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
