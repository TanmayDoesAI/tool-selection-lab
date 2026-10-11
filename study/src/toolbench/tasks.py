import hashlib,json
from pathlib import Path
from .world import fresh
ROOT=Path(__file__).resolve().parents[2]
if not (ROOT/'data/freeze.json').exists(): ROOT=Path(__file__).resolve().parent
def verify_freeze():
    manifest=json.loads((ROOT/'data/freeze.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=digest: raise ValueError(f'Frozen task file changed: {name}')
    return manifest

def load(split='dev'):
    verify_freeze()
    return json.loads((ROOT/'data'/f'{split}.json').read_text())
def fixture(task):
    state=fresh()
    for table,entries in task.get('fixture_patch',{}).items():
        for key,values in entries.items(): state[table][key].update(values)
    return state
