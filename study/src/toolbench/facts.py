"""Requested facts: resolve data/expects.json paths against the executed gold trace."""
import re

def gold_reads(task):
    """{step: output} for every non-mutating reference step, by re-executing the gold trace."""
    from toolbench.tools import TOOLS
    from toolbench.tasks import fixture
    state=fixture(task);reads={}
    for i,(name,args) in enumerate(task['reference'],1):
        out=TOOLS[name].execute(state,args)
        if not TOOLS[name].mutates:reads[i]=out
    return reads

def resolve_step(reads,path):
    step,_,rest=path.partition('.')
    if not step.isdigit() or int(step) not in reads:raise ValueError(f'{path}: step {step} is not a read step')
    out=reads[int(step)]
    if rest=='@empty':
        if out not in ([],{},None,0):raise ValueError(f'{path}: step output is not empty')
        return '@empty'
    return resolve(out,rest)

def resolve(facts,path):
    """Return the scalar values a path selects; raises on a bad path."""
    nodes=[facts]
    for part in re.findall(r'\[\*\]|\[\d+\]|[^.\[\]]+',path):
        out=[]
        for n in nodes:
            if part=='[*]':
                if not isinstance(n,list):raise ValueError(f'{path}: [*] on non-list')
                out+=n
            elif part.startswith('['):out.append(n[int(part[1:-1])])
            else:
                if not isinstance(n,dict) or part not in n:raise ValueError(f'{path}: no key {part!r}')
                out.append(n[part])
        nodes=out
    for v in nodes:
        if isinstance(v,(dict,list)) or v is None or isinstance(v,bool):raise ValueError(f'{path}: selects {type(v).__name__ if v is not None else "null"}, need a number or string')
    if not nodes:raise ValueError(f'{path}: selects nothing')
    return nodes
