"""Example: plug your own agent harness into toolpick.

toolpick drives the loop (discovery, mocked tool results, step limit, grading); your adapter only has to turn one
conversation + tool list into one model turn. Run it with:

    uv run toolpick run --agent examples/custom_agent.py:MyHarnessAgent \
        --catalog examples/northstar-slice/catalog.json --cases examples/northstar-slice/cases.json

Messages arrive in toolpick's neutral format (see toolpick/agents.py):
    {'role': 'system'|'user', 'content': str}
    {'role': 'assistant', 'content': str|None, 'tool_calls': [ToolCall(id, name, arguments)]}
    {'role': 'tool', 'tool_call_id': str, 'name': str, 'content': str}   # content is JSON text
Tools arrive as {'name', 'description', 'parameters'} with parameters a JSON Schema.
Return a Turn: text (the final reply) and/or tool_calls, plus token counts if you have them (None if not).
"""
import re
from toolpick import Turn,ToolCall

def my_harness_turn(history,tool_specs):
    """Stand-in for your real harness (LangGraph node, in-house SDK, a hosted agent ...). This toy version calls the
    tool whose name shares the most words with the request, once, then replies. Replace it with your own call."""
    request=next(m['content'] for m in history if m['role']=='user').lower()
    if any(m['role']=='tool' for m in history):return {'reply':'Done.'}
    scored=sorted(tool_specs,key=lambda t:-sum(w in request for w in t['name'].split('_')))
    if not scored or not any(w in request for w in scored[0]['name'].split('_')):return {'reply':"I can't help with that."}
    tool=scored[0];ids=re.findall(r'\b[A-Z]{1,4}\d+\b',next(m['content'] for m in history if m['role']=='user'))
    return {'call':tool['name'],'args':{p:(ids[0] if ids else 'unknown') for p in tool['parameters'].get('required',[])}}

class MyHarnessAgent:
    """The adapter: translate toolpick's messages/tools to your harness and its answer back to a Turn."""
    def step(self,messages,tools):
        history=[{'role':m['role'],'content':m.get('content') or ''} for m in messages]
        out=my_harness_turn(history,tools)
        if 'call' in out:return Turn(tool_calls=[ToolCall(id='call-1',name=out['call'],arguments=out['args'])])
        return Turn(text=out['reply'])
