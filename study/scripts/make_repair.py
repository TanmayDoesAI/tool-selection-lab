"""Create a repair run (pre-registration A11): the jobs of a completed core run whose result is api_error,
re-run once in a separate run directory with the same model, config, seed and code. Jobs are chosen only by
`error == api_error`, never by outcome. Then run it with `toolbench resume <repair dir>`.
  uv run python scripts/make_repair.py runs/haiku-core-gpt-oss-20b          # -> runs/haiku-repair-gpt-oss-20b
"""
import argparse,json,uuid
from datetime import datetime,timezone
from pathlib import Path
from toolbench.runner import load_run,protocol_fingerprint
from toolbench.api import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('core',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();core=load_run(a.core)
    if core['status']!='completed':raise SystemExit(f'{a.core}: core run is {core["status"]}, not completed')
    if core['protocol_fingerprint']!=protocol_fingerprint():raise SystemExit(f'{a.core}: protocol code changed since the core run')
    failed={r['job_key'] for r in core['results'] if r['error']=='api_error'}
    plan=[j for j in core['plan'] if f'{j["task_id"]}:{j["strategy"]}:{j["repetition"]}' in failed]
    out=a.out or a.core.parent/a.core.name.replace('haiku-core-','haiku-repair-',1)
    if out.exists():raise SystemExit(f'{out} already exists')
    keep=('config','freeze','tasks','protocol_fingerprint')
    manifest={k:core[k] for k in keep}|{'run_id':uuid.uuid4().hex,'created_at':datetime.now(timezone.utc).isoformat(),'kind':'repair',
        'repair_of':{'run_dir':str(a.core),'run_id':core['run_id'],'job_keys':sorted(failed)},'plan':plan,'api_requests':0,
        'preflight_done':False,'status':'created','storage':'jsonl'}
    atomic_json(out/'checkpoint.json',manifest)
    print(f'{out}: {len(plan)} api_error jobs from {a.core} ({len(core["results"])} results)')

if __name__=='__main__':main()
