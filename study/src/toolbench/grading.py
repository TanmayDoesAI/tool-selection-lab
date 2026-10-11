"""Outcome grading of plain-language replies; reference traces never enter execution or discovery.
State changes are checked exactly. Requested facts (data/expects.json paths into the gold output) must
appear in the reply text, with tolerant number/string matching."""
import json,re
from functools import lru_cache
from .tasks import fixture,ROOT

@lru_cache(maxsize=1)
def expectations():
    path=ROOT/'data'/'expects.json'
    return json.loads(path.read_text()) if path.exists() else {}

def number_patterns(v):
    v=float(v);forms={f'{v:.2f}',f'{v:g}'}|({str(int(v)),f'{int(v):,}'} if v.is_integer() else set())
    pats=[r'(?<![\d.])'+re.escape(f).replace(r'\.','\\.')+r'(?:\.0+)?(?!\.?\d)' for f in forms]
    if 0<abs(v)<1:pats.append(r'(?<![\d.])'+re.escape(f'{v*100:g}')+r'(?:\.0+)?\s*%')  # 0.1 -> 10%
    return pats

def mentions(text,value):
    if isinstance(value,(int,float)):return any(re.search(p,text) for p in number_patterns(value))
    low=text.lower();v=str(value).lower()
    return v in low or v.replace('_',' ') in low

NONE_WORDS=re.compile(r"\b(no|none|not any|zero|0|nothing|aren't any|isn't any|there are no|couldn't find|could not find)\b",re.I)

def facts_in_reply(task,text):
    """None when the task has no annotation; else (all_present, per-path detail)."""
    from .facts import gold_reads,resolve_step
    paths=expectations().get(task['id'])
    if paths is None:return None,[]
    reads=gold_reads(task);detail=[]
    for path in paths:
        values=resolve_step(reads,path)
        ok=bool(NONE_WORDS.search(text)) if values=='@empty' else all(mentions(text,v) for v in dict.fromkeys(values))
        detail.append({'path':path,'found':ok})
    return all(d['found'] for d in detail),detail

def subset(expected,actual):
    if isinstance(expected,dict):return isinstance(actual,dict) and all(k in actual and subset(v,actual[k]) for k,v in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and len(expected)==len(actual) and all(subset(x,y) for x,y in zip(expected,actual))
    if isinstance(expected,(float,int)) and not isinstance(expected,bool):return isinstance(actual,(float,int)) and not isinstance(actual,bool) and abs(expected-actual)<1e-6
    return expected==actual

def grade(task,state,calls,final,shortlist,error=None):
    spec=task['grade']; initial=fixture(task); text=final if isinstance(final,str) else ''
    successes=[c for c in calls if c.get('ok')]
    used={c['name'] for c in successes}
    mutated=any(c.get('ok') and c.get('mutates') for c in calls)
    groups=spec['evidence_groups']
    retrieval_ok=all(set(g)&set(shortlist) for g in groups)
    choice_ok=all(set(g)&used for g in groups)
    state_ok=subset(spec['state'],state)
    # Full untouched-table check prevents a correct change hiding collateral damage.
    for table in initial:
        if table=='events':continue
        expected=spec['state'].get(table)
        if expected is None:
            state_ok=state_ok and initial[table]==state[table]
        elif isinstance(expected,dict):
            state_ok=state_ok and set(state[table])==set(initial[table])|set(expected)
            state_ok=state_ok and all(state[table][k]==v for k,v in initial[table].items() if k not in expected)
        else:state_ok=state_ok and state[table]==expected
    argument_scores=[]
    for c in calls:
        required=spec['argument_constraints'].get(c['name'])
        argument_scores.append({'name':c['name'],'schema_valid':c.get('schema_valid',False),'goal_constraints_met':subset(required,c.get('arguments')) if required is not None else None})
    facts_ok,fact_detail=(True,[]) if spec['status']!='completed' else facts_in_reply(task,text)
    if facts_ok is None:facts_ok=spec.get('facts') is None  # unannotated read task cannot pass silently
    asked='?' in text
    # Replies are plain language: outcome comes from state, reply content and whether it asks.
    if spec['status']=='clarification':
        status_ok=asked and not mutated
        choice_ok=not any(c.get('mutates') for c in calls)
    elif spec['status']=='unsupported':
        status_ok=bool(text.strip()) and not mutated
        choice_ok=not any(c.get('mutates') for c in calls)
    else:status_ok=bool(text.strip())
    keyword_ok=all(x.lower() in text.lower() for x in spec.get('question_contains',[])) if spec['status']=='clarification' else None
    # Completion follows outcomes, not reference tool names; choice is scored separately.
    evidence_ok=bool(successes) if spec['status']=='completed' else choice_ok
    complete=bool(state_ok and facts_ok and status_ok and evidence_ok and not error)
    failures=[]
    if error:failures.append(error)
    if not retrieval_ok and (shortlist or not error):failures.append('retrieval_failure')
    if retrieval_ok and not choice_ok and not error and not complete:failures.append('incorrect_selection')
    if any(not x['schema_valid'] or x['goal_constraints_met'] is False for x in argument_scores):failures.append('bad_arguments')
    if any(c.get('schema_valid') and not c.get('ok') for c in calls):failures.append('execution_failure')
    if not state_ok:failures.append('wrong_state')
    if not facts_ok:failures.append('missing_facts')
    if not status_ok:failures.append('wrong_outcome')
    return {'completed':complete,'tool_choice':bool(choice_ok),'argument_checks':argument_scores,'schema_valid_all':all(x['schema_valid'] for x in argument_scores) if calls else None,'arguments_correct':(False if any(not x['schema_valid'] or x['goal_constraints_met'] is False for x in argument_scores) else (None if any(x['goal_constraints_met'] is None for x in argument_scores) else True)) if calls else None,'state_correct':bool(state_ok),'facts_correct':bool(facts_ok),'fact_detail':fact_detail,'status_correct':bool(status_ok),'clarification_keyword':keyword_ok,'retrieval_sufficient':bool(retrieval_ok) if shortlist or not error else None,'reference_evidence_coverage':bool(choice_ok),'failures':list(dict.fromkeys(failures))}
