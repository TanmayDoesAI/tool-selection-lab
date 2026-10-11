"""Offline shortlist coverage: fraction of tasks whose every evidence group has a tool in the top-k.
No chat calls. semantic-nim needs NIM_API_KEY (one embedding call per request, tool vectors cached).
  uv run python scripts/coverage.py --split dev haiku --ks 3 5 10 20 40
"""
import argparse,json
from collections import defaultdict
from toolbench.tasks import load
from toolbench.tools import TOOLS
from toolbench.strategies import EmbeddingIndex,NimEmbeddingIndex,KeywordIndex

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',nargs='+',default=['dev']);p.add_argument('--ks',nargs='+',type=int,default=[3,5,10,20,40])
    p.add_argument('--retrievers',nargs='+',default=['semantic','semantic-nim']);p.add_argument('--cache',default='runs/.cache');p.add_argument('--out')
    a=p.parse_args();idx={}
    if 'semantic' in a.retrievers:idx['semantic']=EmbeddingIndex(a.cache)
    if 'semantic-nim' in a.retrievers:idx['semantic-nim']=NimEmbeddingIndex(a.cache)
    if 'keyword' in a.retrievers:idx['keyword']=KeywordIndex()
    out={}
    for split in a.split:
        tasks=[t for t in load(split) if t['grade']['evidence_groups']]
        for name,index in idx.items():
            ranks=[]  # per task: k needed so every evidence group is covered
            for t in tasks:
                order,_=index.retrieve(t['request'],100);pos={n:i+1 for i,n in enumerate(order)}
                ranks.append((t['type'],max(min(pos[n] for n in g) for g in t['grade']['evidence_groups'])))
            row={'n':len(ranks),'median_k_needed':sorted(r for _,r in ranks)[len(ranks)//2]}
            for k in a.ks:
                row[f'@{k}']=round(sum(r<=k for _,r in ranks)/len(ranks),3)
                by=defaultdict(list)
                for kind,r in ranks:by[kind].append(r<=k)
                row[f'@{k}_by_type']={kind:f'{sum(v)}/{len(v)}' for kind,v in sorted(by.items())}
            out[f'{split}/{name}']=row
            print(f"{split:6} {name:13} n={row['n']:3} median_k={row['median_k_needed']:3} "+' '.join(f"@{k}={row[f'@{k}']:.2f}" for k in a.ks))
        cats=defaultdict(int)
        for t in tasks:cats[len({TOOLS[n].category for n,_ in t['reference']})]+=1
        print(f'{split:6} tasks by number of categories needed: {dict(sorted(cats.items()))}')
    if a.out:open(a.out,'w').write(json.dumps(out,indent=1))

if __name__=='__main__':main()
