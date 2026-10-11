"""Export the study's 100-tool catalogue and 329 haiku tasks in toolpick's format, so anyone can run our test on their own model or harness.
  uv run --frozen python scripts/export_kit.py ../kit/examples/northstar-haiku
Mocks are the real outputs of each task's reference calls, executed on the task's fixture, so multi-step flows see consistent data.
toolpick grades selection only (right tools, right arguments); the study grades outcomes (final state and facts in the reply),
so scores are comparable across strategies and models inside toolpick, not with the study's numbers.
Mapping: each evidence group's reference tool is expected and its alternatives are allowed; other reference calls are allowed;
multi_step tasks are ordered. Clarification and unsupported tasks become 'ambiguous' with read-only tools allowed: the study passes
them when nothing is changed.
"""
import json,sys
from pathlib import Path
from toolbench.tools import TOOLS,CATEGORIES
from toolbench.tasks import load,fixture
from toolbench.runner import INSTRUCTIONS

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
catalog={'name':'Northstar Retail: 100 tools (tool-selection-lab study)','instructions':INSTRUCTIONS,'categories':CATEGORIES,
    'tools':[{'name':t.name,'description':t.description,'parameters':t.parameters,'category':t.category} for t in TOOLS.values()]}
readonly=sorted(t.name for t in TOOLS.values() if not t.mutates)

def run_mocks(task):
    state=fixture(task);seq={}
    for name,args in task['reference']:seq.setdefault(name,[]).append(TOOLS[name].execute(state,args))
    for g in task['grade']['evidence_groups']:   # alternatives get the output they would give from the task's constraints
        for name in g:
            if name in seq:continue
            try:seq[name]=[TOOLS[name].execute(fixture(task),task['grade']['argument_constraints'].get(name,{}))]
            except Exception:pass
    return {k:(v[0] if len(v)==1 else {'$sequence':v}) for k,v in seq.items()}

cases=[]
for task in load('haiku'):
    spec=task['grade'];ref=[n for n,_ in task['reference']];notes=f"study type: {task['type']}"
    if spec['status']=='completed':
        expected=[];allowed=set(ref)
        for g in spec['evidence_groups']:
            pick=next((n for n in ref if n in g),g[0]);allowed|=set(g)
            if pick not in expected:expected.append(pick)
        expected.sort(key=lambda n:ref.index(n) if n in ref else len(ref))
        expect={'tools':expected}
        extra=sorted(allowed-set(expected))
        if extra:expect['allowed']=extra
        args={n:spec['argument_constraints'][n] for n in expected if spec['argument_constraints'].get(n)}
        if args:expect['args']=args
        kind='multi-step' if len(expected)>1 else 'single'
        if task['type']=='multi_step' and len(expected)>1:expect['ordered']=True
        if not expected:kind='ambiguous';expect={'tools':[],'allowed':readonly};notes+='; no evidence group, read-only allowed'
    else:
        kind='ambiguous';expect={'tools':[],'allowed':readonly}
        notes+='; pass = no state-changing call (the agent should '+('ask' if spec['status']=='clarification' else 'decline')+')'
    case={'id':task['id'],'prompt':task['request'],'type':kind,'expect':expect,'notes':notes}
    first=next((n for n in expect['tools']),None)
    if first:case['category']=TOOLS[first].category
    m=run_mocks(task)
    if m:case['mocks']=m
    cases.append(case)

(out/'catalog.json').write_text(json.dumps(catalog,indent=1)+'\n')
(out/'cases.json').write_text(json.dumps({'name':'Northstar haiku split: 329 cases (tool-selection-lab study)','cases':cases},indent=1,default=str)+'\n')
print(f'{len(catalog["tools"])} tools, {len(cases)} cases -> {out}')
