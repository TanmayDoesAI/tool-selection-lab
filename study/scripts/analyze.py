"""Paired analysis across one or more runs: per-arm completion, exact sign tests between arms, and the sweet spot.
  uv run python scripts/analyze.py runs/A runs/B [--out analysis.json]
Arms are labelled model|strategy so several models' runs can be compared on the same tasks.
"""
import argparse,json,statistics
from collections import defaultdict,Counter
from itertools import combinations
from math import comb
from pathlib import Path
from toolbench.runner import load_run

def mcnemar(b,c):
    """Exact two-sided sign test on discordant pairs."""
    n=b+c
    if n==0:return 1.0
    return min(1.0,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n)

def load(paths):
    arms=defaultdict(dict);meta={}
    for path in paths:
        m=load_run(path);model=m['config']['model'].split('/')[-1]
        for r in m['results']:
            api_failed=r['error']=='api_error' or 'api_error' in r['grading']['failures']
            arms[f'{model}|{r["strategy"]}'][(r['task_id'],r['repetition'])]={'ok':r['grading']['completed'],'api_failed':api_failed,'type':r['type'],
                'input_tokens':r['tokens']['input'],'shortlist':len(r['shortlist']),'failures':r['grading']['failures'],'retrieval_ok':'retrieval_failure' not in r['grading']['failures']}
        meta[str(path)]={'run_id':m['run_id'],'status':m['status'],'model':m['config']['model'],'split':m['config']['split'],'strategies':m['config']['strategies']}
    return arms,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--out');p.add_argument('--margin',type=int,default=2,help='sweet spot: cheapest arm within this many tasks of the best')
    a=p.parse_args();arms,meta=load(a.runs);summary={}
    print(f'{"arm":45} {"done":>7} {"healthy":>9} {"med_in_tok":>10} {"shortlist":>9}  failures')
    for arm,rows in sorted(arms.items()):
        v=list(rows.values());healthy=[x for x in v if not x['api_failed']]
        s={'completed':sum(x['ok'] for x in v),'n':len(v),'api_healthy_completed':sum(x['ok'] for x in healthy),'api_healthy_n':len(healthy),
           'median_input_tokens':statistics.median(x['input_tokens'] for x in v) if v else None,'median_shortlist':statistics.median(x['shortlist'] for x in v) if v else None,
           'failures':dict(Counter(f for x in v for f in x['failures'])),'by_type':{}}
        for kind in sorted({x['type'] for x in v}):
            g=[x for x in v if x['type']==kind];s['by_type'][kind]=f'{sum(x["ok"] for x in g)}/{len(g)}'
        summary[arm]=s
        print(f'{arm:45} {s["completed"]:>3}/{s["n"]:<3} {s["api_healthy_completed"]:>4}/{s["api_healthy_n"]:<4} {s["median_input_tokens"] or 0:>10.0f} {s["median_shortlist"] or 0:>9.0f}  {s["failures"]}')
    print('\nby type:')
    for arm,s in sorted(summary.items()):print(f'  {arm:45} {s["by_type"]}')
    print('\npaired (same task+repetition, both API-healthy): arm1 vs arm2 -> only1 / only2, exact p')
    pairs={}
    for x,y in combinations(sorted(arms),2):
        keys=[k for k in arms[x] if k in arms[y] and not arms[x][k]['api_failed'] and not arms[y][k]['api_failed']]
        if not keys:continue
        b=sum(arms[x][k]['ok'] and not arms[y][k]['ok'] for k in keys);c=sum(arms[y][k]['ok'] and not arms[x][k]['ok'] for k in keys)
        pairs[f'{x} vs {y}']={'n':len(keys),'only_first':b,'only_second':c,'p':mcnemar(b,c)}
        print(f'  {x:40} vs {y:40} n={len(keys):3} {b:>2}/{c:<2} p={mcnemar(b,c):.3f}')
    print(f'\nsweet spot per model (cheapest arm within {a.margin} tasks of the best, by median input tokens):')
    spots={}
    for model in sorted({k.split('|')[0] for k in summary}):
        mine={k:v for k,v in summary.items() if k.startswith(model+'|')};best=max(v['completed'] for v in mine.values())
        ok=[k for k,v in mine.items() if v['completed']>=best-a.margin]
        spots[model]=min(ok,key=lambda k:mine[k]['median_input_tokens'] or 0)
        print(f'  {model}: best={best} -> {spots[model]} ({mine[spots[model]]["completed"]} done, {mine[spots[model]]["median_input_tokens"]:.0f} median input tokens)')
    if a.out:Path(a.out).write_text(json.dumps({'runs':meta,'arms':summary,'paired':pairs,'sweet_spot':spots},indent=1))

if __name__=='__main__':main()
