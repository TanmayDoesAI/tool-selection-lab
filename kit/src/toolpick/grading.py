"""Selection-level grading: did the agent call the right tools, with the required arguments, and nothing it must not?

A call is *valid* when the tool was callable at that moment and its arguments passed the tool's JSON Schema.
  single / multi-step  pass = every expected tool called validly (in order if ordered), argument checks met,
                       no forbidden tool called, and the agent finished (no error, no step limit).
                       Extra tools are reported (precision) but do not fail the case.
  ambiguous            as above, but any tool outside expect.tools + expect.allowed fails: the agent should ask.
  unsupported          pass = no tool call at all and a final reply.
"""

def same(expected,actual):
    if expected=='*':return actual is not None
    if isinstance(expected,str) and isinstance(actual,str):return expected.strip().casefold()==actual.strip().casefold()
    if isinstance(expected,(int,float)) and isinstance(actual,(int,float)) and not isinstance(expected,bool):return abs(expected-actual)<1e-9
    return expected==actual

def subsequence(needle,hay):
    it=iter(hay);return all(any(x==y for y in it) for x in needle)

def grade(case,calls,final,error,exposed):
    """calls: [{'name','arguments','valid', ...}] in order; exposed: every tool name the agent could ever see."""
    e=case['expect'];expected=e['tools'];allowed=set(e.get('allowed',[]));forbidden=set(e.get('forbidden',[]))
    valid=[c for c in calls if c['valid']];called=[c['name'] for c in valid];distinct=set(called)
    want=set(expected);hit=want&distinct
    missing=sorted(want-distinct);extra=sorted(distinct-want-allowed);forbidden_called=sorted(forbidden&{c['name'] for c in calls})
    args_ok=all(any(all(k in (c['arguments'] or {}) and same(v,c['arguments'][k]) for k,v in checks.items()) for c in valid if c['name']==tool) for tool,checks in e.get('args',{}).items())
    order_ok=subsequence(expected,called) if e.get('ordered') else True
    precision=(len(distinct&(want|allowed))/len(distinct)) if distinct else (1.0 if not want else None)
    recall=len(hit)/len(want) if want else (1.0 if not calls else 0.0)
    not_exposed=sorted(want-set(exposed))
    if error:reason=error
    elif case['type']=='unsupported':reason='acted_on_unsupported' if calls else None
    elif not_exposed and missing:reason='not_exposed'
    elif missing:reason='missing_tool'
    elif forbidden_called:reason='forbidden_tool'
    elif not args_ok:reason='wrong_args'
    elif not order_ok:reason='wrong_order'
    elif case['type']=='ambiguous' and extra:reason='acted_on_ambiguous'
    else:reason=None
    if reason is None and final is None:reason='no_final_reply'
    return {'success':reason is None,'reason':reason,'precision':precision,'recall':recall,'missing':missing,'extra':extra,
            'forbidden_called':forbidden_called,'not_exposed':not_exposed,'args_ok':args_ok,'order_ok':order_ok,
            'invalid_calls':len(calls)-len(valid)}
