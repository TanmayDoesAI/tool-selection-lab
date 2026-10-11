"""Summarise a run: per-strategy success, selection precision/recall, tokens, latency and paired sign tests.
Writes report.md, results.json and results.csv into the run directory."""
import csv,json,statistics
from collections import Counter,defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from .runner import Store,key

def sign_test(b,c):
    """Exact two-sided sign (McNemar) test on discordant pairs: b = only A passed, c = only B passed."""
    n=b+c
    return 1.0 if n==0 else min(1.0,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n)

def mean(xs):
    xs=[x for x in xs if x is not None];return round(sum(xs)/len(xs),3) if xs else None
def median(xs,digits=1):
    xs=[x for x in xs if x is not None]
    if not xs:return None
    m=round(statistics.median(xs),digits);return int(m) if digits==0 else m

def summarise(rows):
    latest={}
    for r in rows:latest[key(r)]=r  # a re-run job (--retry-errors) replaces its earlier row
    rows=list(latest.values());by=defaultdict(list)
    for r in rows:by[r['strategy']].append(r)
    arms={}
    for s,rs in by.items():
        g=[r['grade'] for r in rs];types=defaultdict(list)
        for r in rs:types[r['type']].append(r['grade']['success'])
        arms[s]={'n':len(rs),'success':sum(x['success'] for x in g),'success_rate':round(sum(x['success'] for x in g)/len(rs),3),
                 'precision':mean(x['precision'] for x in g),'recall':mean(x['recall'] for x in g),
                 'median_input_tokens':median((r['input_tokens'] for r in rs),0),'median_seconds':median((r['seconds'] for r in rs),2),
                 'median_shortlist':median((r['shortlist_size'] for r in rs),0),'by_type':{t:f'{sum(v)}/{len(v)}' for t,v in sorted(types.items())},
                 'failures':dict(Counter(x['reason'] for x in g if x['reason']).most_common())}
    outcome={(r['case_id'],r['repetition'],r['strategy']):r['grade']['success'] for r in rows}
    units=sorted({(c,rep) for c,rep,_ in outcome})
    pairs=[]
    for a,b in combinations(sorted(arms),2):
        both=[u for u in units if (*u,a) in outcome and (*u,b) in outcome]
        only_a=sum(outcome[(*u,a)] and not outcome[(*u,b)] for u in both);only_b=sum(outcome[(*u,b)] and not outcome[(*u,a)] for u in both)
        pairs.append({'a':a,'b':b,'paired':len(both),'only_a':only_a,'only_b':only_b,'p':round(sign_test(only_a,only_b),4)})
    return rows,arms,pairs

def rank(arms):
    """Best first: success rate, then fewer median input tokens."""
    return sorted(arms,key=lambda s:(-arms[s]['success_rate'],arms[s]['median_input_tokens'] if arms[s]['median_input_tokens'] is not None else float('inf')))

def markdown(meta,arms,pairs):
    order=rank(arms);lines=[f'# toolpick report','',f'Model: `{meta.get("model") or meta.get("agent")}` via `{meta.get("provider","custom")}` · catalog fingerprint `{meta.get("fingerprint")}` · max steps {meta.get("max_steps")}','',
        '| strategy | success | precision | recall | median input tokens | median shortlist | median seconds | top failure |','|---|---|---|---|---|---|---|---|']
    for s in order:
        a=arms[s];top=next(iter(a['failures']),'')
        lines.append(f'| {s} | {a["success"]}/{a["n"]} ({a["success_rate"]:.0%}) | {a["precision"]} | {a["recall"]} | {a["median_input_tokens"]} | {a["median_shortlist"]} | {a["median_seconds"]} | {top} |')
    types=sorted({t for a in arms.values() for t in a['by_type']})
    lines+=['','## By case type','','| strategy | '+' | '.join(types)+' |','|---|'+'---|'*len(types)]
    lines+=[f'| {s} | '+' | '.join(arms[s]['by_type'].get(t,'-') for t in types)+' |' for s in order]
    lines+=['','## Paired comparisons','','Exact two-sided sign test over cases both strategies ran. `only A` = cases A passed and B failed.','',
            '| A | B | paired | only A | only B | p |','|---|---|---|---|---|---|']
    lines+=[f'| {p["a"]} | {p["b"]} | {p["paired"]} | {p["only_a"]} | {p["only_b"]} | {p["p"]} |' for p in sorted(pairs,key=lambda p:p['p'])]
    best=order[0] if order else None
    if best:
        close=[p for p in pairs if best in (p['a'],p['b']) and p['p']>=0.05]
        lines+=['','## Reading this','',f'Highest success: **{best}**.'+(f' Not significantly different (sign test p >= 0.05) from: {", ".join(p["b"] if p["a"]==best else p["a"] for p in close)}. Among those, prefer the one with fewer tokens, or add cases to separate them.' if close else '')]
    lines+=['','Failure reasons: `not_exposed` the needed tool was never shown (a discovery miss); `missing_tool` it was shown but not called; `forbidden_tool` a near-miss tool was called; `wrong_args`, `wrong_order`; `acted_on_unsupported` / `acted_on_ambiguous` it acted when it should have declined or asked; `step_limit`, `api_error`, `routing_error`, `agent_error`.']
    return '\n'.join(lines)+'\n'

CSV_FIELDS=['case_id','type','category','strategy','repetition','success','reason','precision','recall','missing','extra','forbidden_called','not_exposed','shortlist_size','exposed_size','model_turns','input_tokens','output_tokens','seconds','error','calls','final']

def generate(out):
    out=Path(out);meta=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
    rows,arms,pairs=summarise(Store(out).rows())
    if not rows:raise SystemExit(f'no results in {out}')
    (out/'report.md').write_text(markdown(meta,arms,pairs))
    (out/'results.json').write_text(json.dumps({'run':meta,'strategies':arms,'ranking':rank(arms),'paired':pairs},indent=1))
    with open(out/'results.csv','w',newline='') as f:
        w=csv.DictWriter(f,CSV_FIELDS);w.writeheader()
        for r in sorted(rows,key=lambda r:(r['strategy'],r['case_id'],r['repetition'])):
            g=r['grade'];w.writerow({**{k:r.get(k) for k in CSV_FIELDS},**{k:g[k] for k in ('success','reason','precision','recall')},
                **{k:' '.join(g[k]) for k in ('missing','extra','forbidden_called','not_exposed')},'calls':' '.join(c['name'] for c in r['calls']),'final':(r['final'] or '')[:300]})
    return arms,pairs
