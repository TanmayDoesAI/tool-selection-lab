"""Compile independently authored raw tasks into graded tasks by executing their reference traces.

Raw task: {"id","request","type","reference":[[tool,args],...],"fixture_patch"?,"question_contains"?}
Gold state changes and final facts are derived by running the reference on a fresh fixture,
never typed by the author. Usage:
  uv run python scripts/compile_tasks.py --check raw.json            # validate only
  uv run python scripts/compile_tasks.py raw1.json raw2.json --out data/haiku.json
"""
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
from jsonschema import Draft202012Validator
from toolbench.tools import TOOLS
from toolbench.tasks import fixture
from toolbench.grading import grade

TYPES={'direct','paraphrase','confusable','multi_step','clarification','unsupported'}
# Read tools whose facts are also recoverable from a broader read (same as the original authoring script).
ALIASES={'get_order_total':['get_order'],'get_order_items':['get_order'],'get_available_stock':['get_stock'],'get_loyalty_balance':['get_customer'],'estimate_delivery':['get_shipment'],'track_parcel':['get_shipment'],'get_invoice_balance':['get_invoice'],'get_refund_capacity':['get_invoice'],'get_ticket_sla':['get_ticket'],'get_product_margin':['get_product']}

def compile_task(raw,author):
    kind=raw.get('type')
    if kind not in TYPES:raise ValueError(f'type must be one of {sorted(TYPES)}')
    if not isinstance(raw.get('request'),str) or len(raw['request'].strip())<8:raise ValueError('request text missing')
    patch=raw.get('fixture_patch') or {};refs=raw.get('reference') or []
    base={'id':raw['id'],'request':raw['request'].strip(),'type':kind,'fixture_patch':patch,'reference':refs,'split':'haiku','smoke':False,'author':author}
    state=fixture(base)  # raises on unknown patch rows
    if kind in ('clarification','unsupported'):
        if refs:raise ValueError(f'{kind} tasks must have an empty reference')
        spec={'status':kind,'state':{},'evidence_groups':[],'argument_constraints':{}}
        if kind=='clarification':
            needles=raw.get('question_contains')
            if not needles or not all(isinstance(x,str) and x for x in needles):raise ValueError('clarification needs question_contains keywords')
            spec['question_contains']=needles
        return {**base,'grade':spec}
    if not refs:raise ValueError('completed tasks need at least one reference call')
    if kind=='multi_step' and len(refs)<2:raise ValueError('multi_step needs 2+ reference calls')
    before=deepcopy(state);outputs=[]
    for name,args in refs:
        if name not in TOOLS:raise ValueError(f'unknown tool {name}')
        Draft202012Validator(TOOLS[name].parameters).validate(args)
        outputs.append(TOOLS[name].execute(state,args))
    changes={}
    for table in state:
        if table=='events' or before[table]==state[table]:continue
        changes[table]={k:v for k,v in state[table].items() if before[table].get(k)!=v} if isinstance(state[table],dict) else state[table]
    groups=[];constraints={}
    for name,args in refs:
        group=[name]+ALIASES.get(name,[]);groups.append(group)
        for alias in group:constraints[alias]={k:v for k,v in args.items() if k in TOOLS[alias].parameters['properties']}
    facts=outputs[-1] if not TOOLS[refs[-1][0]].mutates else None
    if facts is None and not changes:raise ValueError('reference changes nothing and ends with a mutation; no checkable outcome')
    task={**base,'grade':{'status':'completed','facts':facts,'state':changes,'evidence_groups':groups,'argument_constraints':constraints}}
    # The reference trace itself must pass the real grader.
    calls=[{'name':n,'arguments':a,'ok':True,'schema_valid':True,'mutates':TOOLS[n].mutates} for n,a in refs]
    score=grade(task,state,calls,'Done: '+json.dumps(facts),list(TOOLS))
    # Requested-fact annotations come later (data/expects.json) and argument checks are diagnostics
    # (one constraint per tool name); here state, outcome and evidence must pass.
    if [f for f in score['failures'] if f not in ('missing_facts','bad_arguments')]:raise ValueError(f'reference does not pass grader: {score["failures"]}')
    return task

def main():
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args();tasks=[];errors=[];seen=set()
    for f in a.files:
        doc=json.loads(f.read_text());author=doc.get('author',f.stem)
        for raw in doc['tasks']:
            tid=raw.get('id')
            try:
                if not tid or tid in seen:raise ValueError('missing or duplicate id')
                seen.add(tid);tasks.append(compile_task(raw,author))
            except Exception as exc:errors.append(f'{f.name}:{tid}: {type(exc).__name__}: {str(exc).splitlines()[0]}')
    for e in errors:print('ERROR',e)
    from collections import Counter
    print(f'{len(tasks)} valid, {len(errors)} invalid; types {dict(Counter(t["type"] for t in tasks))}')
    if a.out and not a.check:
        if errors:sys.exit('Refusing to write with invalid tasks; fix or remove them first')
        a.out.write_text(json.dumps(tasks,indent=1)+'\n');print('wrote',a.out)
    sys.exit(1 if errors else 0)

if __name__=='__main__':main()
