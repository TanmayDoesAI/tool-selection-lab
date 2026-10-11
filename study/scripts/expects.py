"""Requested-fact annotations for plain-language answers.
data/expects.json maps task id -> list of paths into the outputs of the gold trace's READ steps,
prefixed by the 1-based step number: ["1.available"], ["2.items[*].sku"], ["1.[*].ticket_id"].
"<step>.@empty" means that step found nothing and the reply must say so; [] means the request asks
for no facts (e.g. a lookup only used to find an ID). Values always come from executing the gold trace.
  uv run python scripts/expects.py --check data/expects.json      # validate
  uv run python scripts/expects.py --export DIR --parts 4         # write annotation inputs
"""
import argparse,json
from pathlib import Path
from toolbench.facts import gold_reads,resolve_step

def tasks_with_facts():
    from toolbench.tasks import load
    from toolbench.tools import TOOLS
    return [t for split in ('dev','test','haiku') for t in load(split) if any(not TOOLS[n].mutates for n,_ in t['reference'])]

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',type=Path);p.add_argument('--export',type=Path);p.add_argument('--parts',type=int,default=4)
    a=p.parse_args();tasks={t['id']:t for t in tasks_with_facts()}
    if a.export:
        a.export.mkdir(parents=True,exist_ok=True);ids=sorted(tasks)
        for i in range(a.parts):
            part=[{'id':k,'request':tasks[k]['request'],'steps':[{'step':n,'tool':tasks[k]['reference'][n-1][0],'output':o} for n,o in gold_reads(tasks[k]).items()],
                   'actions_taken_after_or_between':[f'{n}:{name}' for n,(name,_) in enumerate(tasks[k]['reference'],1) if n not in gold_reads(tasks[k])]} for k in ids[i::a.parts]]
            (a.export/f'expects_in_{i}.json').write_text(json.dumps(part,indent=1))
        print(f'{len(ids)} tasks in {a.parts} parts')
    if a.check:
        ann=json.loads(a.check.read_text());errors=[]
        for k,paths in ann.items():
            if k not in tasks:errors.append(f'{k}: not a task with facts');continue
            if not isinstance(paths,list):errors.append(f'{k}: must be a list');continue
            reads=gold_reads(tasks[k])
            for path in paths:
                try:resolve_step(reads,path)
                except (ValueError,IndexError,KeyError,TypeError) as exc:errors.append(f'{k}: {exc}')
        missing=sorted(set(tasks)-set(ann))
        for e in errors:print('ERROR',e)
        print(f'{len(ann)} annotated, {len(errors)} errors, {len(missing)} of {len(tasks)} tasks not annotated in this file')
        raise SystemExit(1 if errors else 0)

if __name__=='__main__':main()
