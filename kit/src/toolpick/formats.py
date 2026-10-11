"""Load, convert and validate the kit's two inputs: catalog.json (your tools) and cases.json (your test prompts)."""
import json,re
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator

def schema_dir():
    here=Path(__file__).resolve().parent
    for p in (here/'schemas',here.parents[1]/'schemas'):
        if p.is_dir():return p
    raise FileNotFoundError('toolpick schemas not found')

def schema(name):return json.loads((schema_dir()/f'{name}.schema.json').read_text())
def read_json(path):return json.loads(Path(path).read_text())

# ---------- conversion from common tool formats ----------
EMPTY={'type':'object','properties':{}}

def clean_name(name):return re.sub(r'[^A-Za-z0-9_-]','_',str(name))[:64]

def to_tool(item):
    """One tool in OpenAI (chat or responses), Anthropic, MCP or canonical form -> canonical dict."""
    if not isinstance(item,dict):raise ValueError(f'not a tool object: {item!r}'[:120])
    extra={k:item[k] for k in ('category','mutates') if k in item}
    if 'function' in item:item={**item['function'],**extra}  # OpenAI chat-completions: {"type":"function","function":{...}}
    params=item.get('parameters') or item.get('input_schema') or item.get('inputSchema') or EMPTY  # OpenAI / Anthropic / MCP
    params=dict(params);params.setdefault('type','object')
    if 'name' not in item:raise ValueError(f'tool has no name: {json.dumps(item)[:120]}')
    tool={'name':clean_name(item['name']),'description':(item.get('description') or item.get('title') or '').strip() or item['name'],'parameters':params}
    for k in ('category','mutates'):
        if k in item:tool[k]=item[k]
    return tool

def convert(data):
    """Any supported JSON tool listing -> canonical catalog. Accepts a list of tools, {"tools": [...]},
    an MCP JSON-RPC response {"result": {"tools": [...]}}, or an existing toolpick/study catalog."""
    if isinstance(data,dict) and 'result' in data and isinstance(data['result'],dict):data=data['result']
    meta={}
    if isinstance(data,dict):
        meta={k:data[k] for k in ('name','instructions','categories') if k in data}
        data=data.get('tools')
    if not isinstance(data,list):raise ValueError('expected a list of tools or an object with a "tools" list')
    return {**meta,'tools':[to_tool(t) for t in data]}

# ---------- validation ----------
def _schema_errors(name,data):
    return [f'{name}: {"/".join(map(str,e.absolute_path)) or "(root)"}: {e.message}' for e in Draft202012Validator(schema(name)).iter_errors(data)]

def validate_catalog(cat):
    errors=_schema_errors('catalog',cat);warnings=[]
    if errors:return errors,warnings
    names=[t['name'] for t in cat['tools']]
    errors+=[f'catalog: duplicate tool name {n}' for n,c in Counter(names).items() if c>1]
    for t in cat['tools']:
        try:Draft202012Validator.check_schema(t['parameters'])
        except Exception as exc:errors.append(f'catalog: {t["name"]}: parameters is not a valid JSON Schema: {str(exc).splitlines()[0]}')
        if len(t['description'])<15:warnings.append(f'catalog: {t["name"]}: very short description; retrieval and the model both rely on it')
    cats={t.get('category') for t in cat['tools']}-{None}
    if cats and any('category' not in t for t in cat['tools']):warnings.append('catalog: some tools have no category; the category strategy can never reach them')
    for c in sorted(cats-set(cat.get('categories',{}))):warnings.append(f'catalog: category {c!r} has no description in "categories"; the router will see only its tool names')
    if not cats:warnings.append('catalog: no categories; the category strategy will be skipped')
    return errors,warnings

def validate_cases(cases,cat):
    errors=_schema_errors('cases',cases);warnings=[]
    if errors:return errors,warnings
    tools={t['name']:t for t in cat['tools']};items=cases['cases']
    errors+=[f'cases: duplicate id {i}' for i,c in Counter(c['id'] for c in items).items() if c>1]
    for name in cases.get('mocks',{}):
        if name not in tools:warnings.append(f'cases: mock for unknown tool {name}')
    for c in items:
        e=c['expect'];where=f'cases: {c["id"]}'
        for field in ('tools','allowed','forbidden'):
            for n in e.get(field,[]):
                if n not in tools:errors.append(f'{where}: expect.{field} names unknown tool {n}')
        both=set(e['tools'])&set(e.get('forbidden',[]))
        if both:errors.append(f'{where}: tools both expected and forbidden: {sorted(both)}')
        for n,args in e.get('args',{}).items():
            if n not in e['tools']:errors.append(f'{where}: expect.args for {n}, which is not in expect.tools');continue
            if n in tools:
                props=tools[n]['parameters'].get('properties',{})
                for a in args:
                    if props and a not in props:errors.append(f'{where}: {n} has no parameter {a!r}')
        for n in c.get('mocks',{}):
            if n not in tools:warnings.append(f'{where}: mock for unknown tool {n}')
        for n in tools:
            if len(n)>3 and re.search(r'\b'+re.escape(n)+r'\b',c['prompt']):warnings.append(f'{where}: prompt contains the tool name {n!r}; real users rarely do this and it makes retrieval trivial')
        if c['type']=='multi-step' and not e.get('ordered') and len(set(e['tools']))<len(e['tools']):warnings.append(f'{where}: repeated tool in an unordered expectation counts once')
    types=Counter(c['type'] for c in items)
    if not 20<=len(items)<=30:warnings.append(f'cases: {len(items)} cases; 20-30 is the recommended range (fewer gives noisy comparisons)')
    for t in ('single','multi-step','unsupported'):
        if not types[t]:warnings.append(f'cases: no {t} cases')
    cats={t.get('category') for t in cat['tools']}-{None}
    covered={tools[n].get('category') for c in items for n in c['expect']['tools'] if n in tools}
    for m in sorted(cats-covered):warnings.append(f'cases: no case expects a tool from category {m!r}')
    if not any(c['expect'].get('forbidden') for c in items):warnings.append('cases: no case lists forbidden near-miss tools')
    return errors,warnings

def load(catalog_path,cases_path=None):
    """Load and validate; raises ValueError with every error at once."""
    cat=read_json(catalog_path);errors,warnings=validate_catalog(cat)
    cases=None
    if cases_path is not None and not errors:
        cases=read_json(cases_path);e,w=validate_cases(cases,cat);errors+=e;warnings+=w
    if errors:raise ValueError('\n'.join(errors))
    return cat,cases,warnings
