import html,json,statistics
from collections import Counter
from pathlib import Path
from .api import atomic_json

def summary(manifest):
    rows=manifest['results']; result={'run_id':manifest['run_id'],'status':manifest['status'],'planned':len(manifest['plan']),'recorded':len(rows),'missing':len(manifest['plan'])-len(rows),'api_requests':manifest['api_requests'],'tokens':{'input':sum((e.get('usage') or {}).get('prompt_tokens',0) for e in manifest['attempts']),'output':sum((e.get('usage') or {}).get('completion_tokens',0) for e in manifest['attempts'])},'unknown_usage_attempts':sum(e.get('usage') is None for e in manifest['attempts']),'strategies':{},'task_types':{}}
    for strategy in manifest['config']['strategies']:
        selected=[r for r in rows if r['strategy']==strategy];n=len(selected)
        planned=sum(j['strategy']==strategy for j in manifest['plan'])
        result['strategies'][strategy]={'completed':sum(r['grading']['completed'] for r in selected),'recorded':n,'planned':planned,'completion_rate_recorded':sum(r['grading']['completed'] for r in selected)/n if n else None,'median_seconds':statistics.median(r['timings']['end_to_end_seconds'] for r in selected) if n else None,'input_tokens':sum(r['tokens']['input'] for r in selected),'output_tokens':sum(r['tokens']['output'] for r in selected),'shortlist_sizes':[len(r['shortlist']) for r in selected],'failures':dict(Counter(x for r in selected for x in r['grading']['failures']))}
        for kind in dict.fromkeys(t['type'] for t in manifest['tasks']):
            group=[r for r in selected if r['type']==kind]
            result['task_types'][f'{strategy}/{kind}']={'completed':sum(r['grading']['completed'] for r in group),'recorded':len(group),'planned':sum(j['strategy']==strategy and next(t['type'] for t in manifest['tasks'] if t['id']==j['task_id'])==kind for j in manifest['plan'])}
    return result

def generate(path,manifest):
    data=summary(manifest);atomic_json(path/'summary.json',data)
    escape=lambda x:html.escape(str(x))
    cards=''
    for name,row in data['strategies'].items():
        med=f'{row["median_seconds"]:.2f}s' if row['median_seconds'] is not None else '—'
        cards+=f'<article><h2>{escape(name)}</h2><strong>{row["completed"]}/{row["recorded"]} completed</strong><p>{row["planned"]} planned · median {med}</p><p>{row["input_tokens"]:,} input / {row["output_tokens"]:,} output tokens</p><p>Actual shortlist sizes: {escape(row["shortlist_sizes"])}</p></article>'
    attempts={e['request_number']:e for e in manifest['attempts']}
    traces=''
    for row in manifest['results']:
        payload={**row,'model_attempts':[attempts[n] for n in row['attempt_numbers']]}
        traces+=f'<details data-strategy="{escape(row["strategy"])}" data-result="{str(row["grading"]["completed"]).lower()}"><summary>{escape(row["task_id"])} · {escape(row["strategy"])} · {"PASS" if row["grading"]["completed"] else "FAIL"} · {escape(row["grading"]["failures"])}</summary><p>{escape(row["request"])}</p><pre>{escape(json.dumps(payload,indent=2))}</pre></details>'
    pairs=''
    for task in manifest['tasks']:
        cells=''
        for strategy in manifest['config']['strategies']:
            rr=[r for r in manifest['results'] if r['task_id']==task['id'] and r['strategy']==strategy]
            cells+=f'<td>{sum(r["grading"]["completed"] for r in rr)}/{len(rr)} · '+(', '.join(f'{r["timings"]["end_to_end_seconds"]:.2f}s' for r in rr) if rr else 'missing')+'</td>'
        pairs+=f'<tr><td>{escape(task["id"])}</td><td>{escape(task["type"])}</td>{cells}</tr>'
    breakdown=''.join(f'<tr><td>{escape(k)}</td><td>{v["completed"]}</td><td>{v["recorded"]}</td><td>{v["planned"]}</td></tr>' for k,v in data['task_types'].items())
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>100 Tools, One Agent — Evidence</title>
<style>body{{font:16px system-ui;background:#f3f5f9;color:#172238;max-width:1200px;margin:auto;padding:32px}}h1{{font-size:36px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}article,details,.notice{{background:white;border:1px solid #ccd5e2;border-radius:10px;padding:18px;margin:12px 0}}.notice{{border-left:6px solid #bb7413}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:600px;overflow:auto}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d8e0eb}}.table-scroll{{overflow-x:auto;max-width:100%}}@media(max-width:600px){{body{{padding:16px}}h1{{font-size:28px}}}}summary{{cursor:pointer}}select{{padding:8px;margin:8px}}</style>
<h1>100 Tools, One Agent</h1><p>Saved live evidence · run {escape(manifest['run_id'])} · model {escape(manifest['config']['model'])}</p>
<div class="notice"><strong>Implementation validation, not a strategy ranking.</strong><p>Synthetic retailer and jointly authored tasks/tools. {len(manifest['tasks'])} unique {escape(manifest['config']['split'])} tasks · {manifest['config']['repetitions']} repetition(s). {data['recorded']}/{data['planned']} planned observations recorded, {data['missing']} missing. Status: {escape(data['status'])}.</p><p>Request budget: {data['api_requests']}/{manifest['config']['max_requests']} attempts, including preflight, routing and retries. Total measured usage: {data['tokens']['input']:,} input + {data['tokens']['output']:,} output tokens. Unknown usage on {data['unknown_usage_attempts']} attempts. Strategy cards exclude preflight and orphaned attempts; totals include them. No monetary estimate.</p><p>Static shortlists; category cap {manifest['config']['category_limit']}; semantic top-k {manifest['config']['top_k']}. Completion rates use recorded observations only; planned counts expose missing data. Word-vector baseline, small sample, author overlap and model nondeterminism limit interpretation.</p></div>
<div class="cards">{cards}</div><h2>Embedding setup, separate from warm retrieval</h2><pre>{escape(json.dumps(manifest.get('embedding',{}),indent=2))}</pre>
<h2>Task-type breakdown</h2><table><tr><th>Strategy / type</th><th>Completed</th><th>Recorded</th><th>Planned</th></tr>{breakdown}</table>
<h2>Paired task comparisons</h2><table><tr><th>Task</th><th>Type</th>{''.join('<th>'+escape(s)+'</th>' for s in manifest['config']['strategies'])}</tr>{pairs}</table>
<h2>Inspectable traces and failures</h2><label>Strategy <select id="strategy"><option value="">All</option>{''.join('<option>'+escape(s)+'</option>' for s in manifest['config']['strategies'])}</select></label><label>Outcome <select id="outcome"><option value="">All</option><option value="false">Failures</option><option value="true">Passes</option></select></label>{traces}
<h2>Preflight, orphaned attempts and blockers</h2><pre>{escape(json.dumps({'audit_notes':manifest.get('audit_notes'),'blocker':manifest.get('blocker'),'interrupted_result':manifest.get('interrupted_result'),'attempts':[e for e in manifest['attempts'] if not e.get('job_key') or e['request_number'] not in {n for r in manifest['results'] for n in r['attempt_numbers']}]},indent=2))}</pre>
<script>function filter(){{document.querySelectorAll('details').forEach(d=>d.hidden=(strategy.value&&d.dataset.strategy!==strategy.value)||(outcome.value&&d.dataset.result!==outcome.value))}}document.getElementById('strategy').onchange=filter;document.getElementById('outcome').onchange=filter;</script></html>'''
    document=document.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
    (path/'report.html').write_text(document)
    (path/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest['results']))
    return data
